"""
PHM 시뮬레이션 엔진

AGV fleet의 상태를 시간 단위로 진행하며 MILP 스케줄러와 연동한다.

핵심 설계:
  - AGV.rul_mean: 시뮬레이션 기준 잔존 수명 (시간)
  - AGV.rul_samples: MC Dropout 100개 샘플 — MILP P_fail 계산용
  - LSTM ModelWrapper가 주입되면 실제 RUL 예측값 사용
  - 없으면 선형 감소 mock 모드로 동작 (하위 호환)

상태 머신:
  NORMAL → WARNING(RUL_low ≤ 72h) → CRITICAL(RUL_low ≤ 24h)
         → WAITING(MILP 스케줄 도달) or FAILED(RUL ≤ 0)
         → MAINTENANCE (K 슬롯 배정) → NORMAL (정비 완료)
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

_PROJ_ROOT = Path(__file__).parent.parent.parent
_SIM_DIR = _PROJ_ROOT / 'data' / 'simulation'

# 정비 소요 시간 (계획 / 긴급)
# 베어링 교체: 분해 → 프레스핏 → 조립 → 시운전 포함 현실적 소요시간
_MAINT_PLANNED_H = 8   # 계획 정비: 8시간
_MAINT_EMERGENCY_H = 16 # 긴급 정비: 16시간 (비계획 분해수리)

# 정비 완료 후 새 베어링 수명 가정 (시뮬레이션 시간 단위: hour)
_NEW_BEARING_RUL_MEAN = 3000.0
_NEW_BEARING_RUL_STD = 300.0


def _load_scenario(agv_id: int):
    """시나리오 CSV 로드. 없으면 None."""
    path = _SIM_DIR / f"agv_{agv_id:02d}_scenario.csv"
    if not path.exists():
        return None
    try:
        df = (pd.read_csv(path)
              .drop_duplicates(subset=['timestamp_idx'])
              .sort_values('timestamp_idx')
              .reset_index(drop=True))
        return df
    except Exception as exc:
        print(f"[Engine] 시나리오 로드 실패 (AGV #{agv_id}): {exc}")
        return None


class AGV:
    """단일 AGV 상태 머신.

    Args:
        agv_id       : AGV 식별자 (0~9)
        initial_rul  : 시뮬레이션 초기 RUL (시간)
        scenario_df  : PRONOSTIA 기반 특징 데이터 (센서 트렌드 + RUL)
        model_wrapper: ModelWrapper 인스턴스 (없으면 mock 모드)
    """

    def __init__(self, agv_id: int, initial_rul: float = None,
                 scenario_df=None, model_wrapper=None):
        self.id = agv_id
        self._scenario = scenario_df
        self._model = model_wrapper
        self._scenario_pos = 0.0

        # --- 데이터 기반 초기 RUL 결정 ---
        if initial_rul is None and self._scenario is not None and 'RUL' in self._scenario.columns:
            initial_rul = self._scenario['RUL'].iloc[0]
        
        # 기본값 (데이터 없을 시)
        initial_rul = float(initial_rul) if initial_rul is not None else 600.0
        
        self.rul_mean = initial_rul
        self.rul_low = initial_rul * 0.9
        self.rul_high = initial_rul * 1.1
        self.state = 'NORMAL'
        self.maintenance_time_left = 0
        self.scheduled_maintenance_time = None
        self.wait_time = 0

        # 시나리오와 시뮬레이션 속도 동기화
        self._sync_step_rate(initial_rul)

        self._refresh_samples()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _get_seq_df(self):
        """모델 입력용 최근 seq_len 행 반환."""
        if self._scenario is None:
            return None
        seq_len = self._model.seq_len if self._model else 10
        end = min(int(self._scenario_pos) + 1, len(self._scenario))
        start = max(0, end - seq_len)
        return self._scenario.iloc[start:end]

    def _sync_step_rate(self, life_hours: float):
        """시나리오 전체 길이를 주어진 수명 시간에 맞춰 1시간 단위로 매핑."""
        n_rows = len(self._scenario) if self._scenario is not None else 1
        self._step_rate = n_rows / max(float(life_hours), 1.0)

    def _reset_to_new_bearing(self):
        """정비 완료 후 새 베어링의 초기 열화 구간으로 되돌린다."""
        self.state = 'NORMAL'
        self.rul_mean = _NEW_BEARING_RUL_MEAN
        self.rul_low = _NEW_BEARING_RUL_MEAN - _NEW_BEARING_RUL_STD
        self.rul_high = _NEW_BEARING_RUL_MEAN + _NEW_BEARING_RUL_STD
        self._scenario_pos = 0.0
        self._sync_step_rate(_NEW_BEARING_RUL_MEAN)
        self.rul_samples = list(
            np.random.normal(
                _NEW_BEARING_RUL_MEAN,
                _NEW_BEARING_RUL_STD,
                100,
            ).clip(0)
        )

    def _refresh_samples(self):
        """MC Dropout으로 rul_samples 갱신.

        ModelWrapper.predict() 가 있으면 실제 LSTM RUL 예측값을 사용.
        없으면 현재 rul_mean 기준 정규분포 mock.
        """
        seq_df = self._get_seq_df()

        # --- LSTM 실제 예측 ---
        if (self._model is not None
                and self._model.loaded
                and seq_df is not None
                and len(seq_df) >= self._model.seq_len):
            result = self._model.predict(seq_df)
            if result is not None:
                rul_mean, rul_low, rul_high, samples = result
                # 시나리오 RUL과 LSTM 예측을 블렌딩
                # (초기엔 시나리오 우선, 점차 LSTM 비중 증가)
                alpha = min(1.0, int(self._scenario_pos) / max(self._model.seq_len * 3, 1))
                self.rul_mean = (1 - alpha) * self.rul_mean + alpha * rul_mean
                self.rul_low = rul_low
                self.rul_high = rul_high
                self.rul_samples = [max(0.0, s) for s in samples]
                return

        # --- Mock 모드: 정규분포 샘플링 ---
        cv = (self._model.get_uncertainty_cv(seq_df)
              if (self._model and seq_df is not None and len(seq_df) >= self._model.seq_len)
              else 0.10)
        std = max(self.rul_mean * cv, 5.0)
        samples = list(np.random.normal(self.rul_mean, std, 100).clip(0))
        self.rul_samples = samples
        self.rul_low = float(np.percentile(samples, 5))
        self.rul_high = float(np.percentile(samples, 95))

    # ------------------------------------------------------------------
    # 공개 메서드
    # ------------------------------------------------------------------

    def get_recent_sensor_data(self, n: int = 100):
        """최근 n개 센서 특징 데이터(DataFrame) 반환."""
        if self._scenario is None:
            return None
        end = min(int(self._scenario_pos) + 1, len(self._scenario))
        start = max(0, end - n)
        return self._scenario.iloc[start:end].copy()

    def update_state(self, current_time: int):
        """시뮬레이션 1시간 진행."""

        # --- 정비 중 ---
        if self.state == 'MAINTENANCE':
            self.maintenance_time_left -= 1
            if self.maintenance_time_left <= 0:
                self._reset_to_new_bearing()
            return

        # --- 대기 중 (FAILED / WAITING) ---
        if self.state in ['FAILED', 'WAITING']:
            self.wait_time += 1
            return

        # --- 정상 열화 ---
        self.rul_mean = max(0.0, self.rul_mean - 1.0)

        if self._scenario is not None:
            self._scenario_pos = min(
                self._scenario_pos + self._step_rate,
                len(self._scenario) - 1,
            )

        # MC Dropout 갱신 (매 5스텝, 성능 최적화)
        if current_time % 5 == 0:
            self._refresh_samples()
        else:
            self.rul_samples = [max(0.0, s - 1.0) for s in self.rul_samples]
            self.rul_low = max(0.0, self.rul_low - 1.0)
            self.rul_high = max(0.0, self.rul_high - 1.0)

        # --- 상태 판정 (5th percentile 기준 — 보수적 추정) ---
        rul_low = self.rul_low

        if self.rul_mean <= 0 or rul_low <= 0:
            self.state = 'FAILED'
        elif (self.scheduled_maintenance_time is not None
              and self.scheduled_maintenance_time <= current_time):
            self.state = 'WAITING'
            self.scheduled_maintenance_time = None
        else:
            if rul_low <= 24:
                self.state = 'CRITICAL'
            elif rul_low <= 72:
                self.state = 'WARNING'
            else:
                self.state = 'NORMAL'


# 기본 10대용 초기 RUL (하위 호환 유지)
_DEFAULT_RULS_10 = [280, 220, 175, 145, 120, 100, 75, 50, 28, 10]


def _generate_initial_ruls(n: int, seed: int = 42) -> list[float]:
    """n대 AGV의 초기 RUL을 현실적인 분포로 생성.

    분포 비율:
      10% → CRITICAL (10~50h)
      20% → WARNING  (50~150h)
      40% → NORMAL   (150~600h)
      30% → FRESH    (600~2000h)
    """
    if n <= 10:
        return _DEFAULT_RULS_10[:n]

    rng = np.random.default_rng(seed)
    n_crit  = max(1, int(n * 0.10))
    n_warn  = max(1, int(n * 0.20))
    n_norm  = max(1, int(n * 0.40))
    n_fresh = n - n_crit - n_warn - n_norm

    ruls = np.concatenate([
        rng.uniform(10,   50,   n_crit),
        rng.uniform(50,   150,  n_warn),
        rng.uniform(150,  600,  n_norm),
        rng.uniform(600,  2000, n_fresh),
    ])
    rng.shuffle(ruls)
    return ruls.tolist()


class SimulationEngine:
    """AGV PHM 시뮬레이션 엔진.

    Args:
        num_agvs     : AGV 대수 (기본값 10, 100 등 자유 설정)
        model_wrapper: ModelWrapper 인스턴스 (Streamlit app.py에서 주입)
        initial_ruls : AGV별 초기 RUL (시간) 리스트. None이면 자동 생성
    """

    def __init__(self, num_agvs: int = 10, model_wrapper=None, initial_ruls=None, K: int = 2):
        ruls = initial_ruls if initial_ruls is not None else _generate_initial_ruls(num_agvs)
        self.K = K  # 동시 정비 최대 대수 (MILP scheduler.K와 일치시킬 것)
        self.agvs = [
            AGV(
                agv_id=i,
                initial_rul=ruls[i] if i < len(ruls) else 600.0,
                scenario_df=_load_scenario(i),
                model_wrapper=model_wrapper,
            )
            for i in range(num_agvs)
        ]
        self.current_time = 0
        self.cost_accumulated = 0.0
        self.total_wait_time = 0

        # 히스토리 (백테스팅, 대시보드 시각화용)
        self.cost_history: list[float] = []
        self.failure_count = 0
        self.planned_maint_count = 0

        # ── 추가 지표 추적 ──────────────────────────────
        self.total_downtime = 0       # 전 AGV 합산 정지 시간 (FAILED+MAINT+WAITING AGV·h)
        self.total_maint_time = 0     # 정비 중 시간만 (MAINTENANCE AGV·h)
        self.max_concurrent_down = 0  # 동시 정지 최대 대수
        self.maint_slot_full_time = 0 # K 슬롯 모두 포화된 시간 누적 (h)
        # 경보/고장 이벤트 시각 (PHM Score, 조기 경보 적중률 계산용)
        self.warning_events: list[int] = []   # WARNING/CRITICAL 진입 시각
        self.failure_events: list[int] = []   # FAILED 진입 시각

    def step(self):
        """시뮬레이션 1시간 진행."""
        self.current_time += 1

        # 1. 개별 AGV 상태 업데이트
        for agv in self.agvs:
            old_state = agv.state
            agv.update_state(self.current_time)

            if old_state != 'FAILED' and agv.state == 'FAILED':
                self.cost_accumulated += 300   # 비계획 고장 비용
                self.failure_count += 1
                self.failure_events.append(self.current_time)

            # 경보 이벤트 기록 (NORMAL → WARNING/CRITICAL 진입)
            if old_state == 'NORMAL' and agv.state in ('WARNING', 'CRITICAL'):
                self.warning_events.append(self.current_time)

        # 2. 대기 시간 누적 (FAILED 또는 WAITING 상태)
        for agv in self.agvs:
            if agv.state in ['FAILED', 'WAITING']:
                self.total_wait_time += 1

        # 3. 정비 슬롯 관리 및 배정
        maintenance_count = sum(1 for a in self.agvs if a.state == 'MAINTENANCE')
        available_slots = self.K - maintenance_count

        if available_slots > 0:
            waiting_agvs = [a for a in self.agvs
                            if a.state in ['FAILED', 'WAITING']]
            # 우선순위: 고장 임박도(rul_low)가 높은 순서 (긴급도 우선)
            # RUL이 동일하다면(예: 둘 다 FAILED) 대기 시간이 긴 순서
            waiting_agvs.sort(
                key=lambda a: (a.rul_low, -a.wait_time)
            )
            for agv in waiting_agvs[:available_slots]:
                if agv.state == 'WAITING':
                    self.cost_accumulated += 50   # 계획 정비 비용
                    self.planned_maint_count += 1
                    agv.maintenance_time_left = _MAINT_PLANNED_H
                else:  # FAILED
                    agv.maintenance_time_left = _MAINT_EMERGENCY_H
                agv.state = 'MAINTENANCE'
                agv.wait_time = 0

        # 4. 추가 지표 누적
        down_states = ('FAILED', 'MAINTENANCE', 'WAITING')
        down_count = sum(1 for a in self.agvs if a.state in down_states)
        maint_count_now = sum(1 for a in self.agvs if a.state == 'MAINTENANCE')
        self.total_downtime += down_count
        self.total_maint_time += maint_count_now
        self.max_concurrent_down = max(self.max_concurrent_down, down_count)
        if maint_count_now >= self.K:
            self.maint_slot_full_time += 1

        # 히스토리 기록
        self.cost_history.append(self.cost_accumulated)

    def set_schedule(self, schedule_dict: dict):
        """MILP 최적화 결과를 각 AGV에 적용.

        Args:
            schedule_dict: {agv_id: delay_hours (int) or None}
        """
        for agv_id, start_delay in schedule_dict.items():
            if start_delay is not None:
                self.agvs[agv_id].scheduled_maintenance_time = (
                    self.current_time + start_delay
                )

    def get_in_progress_ids(self) -> set:
        """현재 정비 진행 중인 AGV id 집합 반환 (MILP 재최적화용)."""
        return {a.id for a in self.agvs if a.state == 'MAINTENANCE'}

    def get_in_progress_dict(self) -> dict:
        """현재 정비 진행 중인 AGV id와 남은 정비 시간(hours) 반환 (MILP 용량 계산용)."""
        return {a.id: a.maintenance_time_left for a in self.agvs if a.state == 'MAINTENANCE'}

    def get_summary(self) -> dict:
        """현재 시뮬레이션 요약 통계 반환."""
        return {
            'current_time': self.current_time,
            'cost_accumulated': self.cost_accumulated,
            'failure_count': self.failure_count,
            'planned_maint_count': self.planned_maint_count,
            'total_wait_time': self.total_wait_time,
            'agv_states': {a.id: a.state for a in self.agvs},
        }

    def get_full_metrics(self) -> dict:
        """신뢰성·정비효율·경제성 지표를 모두 포함한 딕셔너리 반환."""
        from src.simulation.metrics import (
            calc_mtbf, calc_mttr, calc_availability,
            calc_unplanned_failure_rate, calc_pf_ratio,
            calc_slot_utilization, calc_avg_wait_per_event,
            calc_early_warning_rate,
        )
        t = self.current_time
        num_agvs = len(self.agvs)
        total_events = self.failure_count + self.planned_maint_count

        return {
            'mode'             : 'PHM-MILP',
            'cost'             : round(self.cost_accumulated, 1),
            'failure_count'    : self.failure_count,
            'planned_count'    : self.planned_maint_count,
            'availability'     : calc_availability(t, self.total_downtime, num_agvs),
            'mtbf'             : calc_mtbf(t, self.failure_count),
            'mttr'             : calc_mttr(self.total_maint_time, total_events),
            'ufr'              : calc_unplanned_failure_rate(self.failure_count,
                                                              self.planned_maint_count),
            'pf_ratio'         : calc_pf_ratio(self.planned_maint_count, self.failure_count),
            'slot_utilization' : calc_slot_utilization(self.total_maint_time, t, K=self.K),
            'avg_wait'         : calc_avg_wait_per_event(self.total_wait_time, total_events),
            'total_downtime'   : self.total_downtime,
            'maint_slot_full_time': self.maint_slot_full_time,
            'max_concurrent_down': self.max_concurrent_down,
            'early_warning_rate': calc_early_warning_rate(
                self.warning_events, self.failure_events, lookahead_h=72.0),
            'mean_rul': round(float(np.mean([a.rul_mean for a in self.agvs])), 1),
        }
