"""
PHM 시뮬레이션 엔진

AGV fleet의 상태를 시간 단위로 진행하며 ILP 스케줄러와 연동한다.

핵심 설계 — ground truth(실제 고장)와 예측(LSTM)의 분리:
  - AGV.true_rul    : 데이터 기반 ground-truth 잔존 수명 (시간). 매 시간 1씩 감소하며
                      true_rul <= 0 일 때만 '실제 고장(FAILED)'이 발생한다. B1/B2 baseline과
                      동일한 ground truth → 모든 전략이 같은 고장 시각표 위에서 비교된다.
  - AGV.rul_mean / rul_low / rul_high / rul_samples : LSTM(MC Dropout) '예측값'.
                      B3 임계치, ILP P_fail 등 '의사결정'에만 사용한다.
                      예측은 결코 물리적 고장을 일으키지 않는다.

  - LSTM ModelWrapper와 시나리오 데이터는 필수다(mock 폴백 제거됨).

상태 머신:
  NORMAL (가동/열화 중)
         → WAITING(ILP 스케줄 도달) or FAILED(true_rul ≤ 0)
         → MAINTENANCE (K 슬롯 배정) → NORMAL (정비 완료, 새 베어링)
"""
import os
from functools import lru_cache
import numpy as np
import pandas as pd
from pathlib import Path

_PROJ_ROOT = Path(__file__).parent.parent.parent
_SIM_DIR = _PROJ_ROOT / 'data' / 'simulation'

# 정비 소요 시간 (계획 / 긴급)
# 베어링 교체: 분해 → 프레스핏 → 조립 → 시운전 포함 현실적 소요시간
_MAINT_PLANNED_H = 8   # 계획 정비: 8시간
_MAINT_EMERGENCY_H = 16 # 긴급 정비: 16시간 (비계획 분해수리)

# 정비 완료 후 새 베어링의 ground-truth 실제 수명 (시간).
# B1/B2 baseline(comparator.py)의 정비 후 RUL 리셋값과 동일한 가정.
_NEW_BEARING_TRUE_LIFE_H = 3000.0


@lru_cache(maxsize=None)
def _load_scenario(agv_id: int):
    """시나리오 CSV 로드. 없으면 None.

    여러 엔진(PHM/B3)·베이스라인(B1/B2)이 동일 파일을 반복 읽으므로 캐싱한다.
    반환 DataFrame은 읽기 전용으로만 사용된다(AGV는 _scenario_pos만 자체 추적).
    """
    path = _SIM_DIR / f"agv_{agv_id:02d}_scenario.csv"
    if not path.exists():
        return None
    try:
        return _select_longest_run(pd.read_csv(path))
    except Exception as exc:
        print(f"[Engine] 시나리오 로드 실패 (AGV #{agv_id}): {exc}")
        return None


def _select_longest_run(df):
    """한 시나리오에 섞여 있을 수 있는 여러 run-to-failure 중 '가장 긴(완주) 런'만 남긴다.

    동명 베어링의 두 런(FEMTO Test_set/Full_Test_Set)이 한 bearing_name으로 병합되면
    한 파일에 두 런이 섞인다(원인 분석: agent_log 04). 각 런은
    RUL = (그 런의 max_idx) - timestamp_idx 이므로 life_id = RUL + timestamp_idx 가 런 전체에서
    상수다. 이를 기준으로 런을 분리해 life_id가 가장 큰(= 수명이 가장 긴 완주) 런만 남기면,
    RUL이 깨끗한 단조 감소가 되고 RUL[0] = 실제 신품 수명이 된다.

    병합이 없는 시나리오(증강본 포함)는 life_id가 단일값이라 전체가 그대로 유지된다.
    """
    if 'RUL' in df.columns and 'timestamp_idx' in df.columns:
        life_id = df['RUL'] + df['timestamp_idx']
        df = df[life_id == life_id.max()]
    return (df.drop_duplicates(subset=['timestamp_idx'], keep='first')
              .sort_values('timestamp_idx')
              .reset_index(drop=True))


def scenario_initial_rul(agv_id: int) -> float:
    """AGV의 데이터 기반 초기 ground-truth RUL(= 시나리오 RUL[0])을 반환한다.

    모든 전략(PHM/B3=SimulationEngine, B1/B2=BaselineSimulator)이 동일한 시나리오
    파일에서 ground truth를 도출하도록 하는 단일 진입점. 임의로 RUL을 지정하는
    기능은 없다 — ground truth는 오직 데이터(시나리오)에서만 나온다.
    """
    df = _load_scenario(agv_id)
    if df is None or 'RUL' not in df.columns:
        raise ValueError(
            f"AGV #{agv_id}: 시나리오/RUL 컬럼이 없어 초기 RUL을 결정할 수 없습니다. "
            f"먼저 시나리오를 생성하세요 (python -m src.simulation.generate_scenarios)."
        )
    return float(df['RUL'].iloc[0])


class AGV:
    """단일 AGV 상태 머신.

    초기 ground-truth RUL은 항상 시나리오 RUL[0]에서 도출한다(임의 지정 불가).

    Args:
        agv_id       : AGV 식별자 (0~9)
        scenario_df  : PRONOSTIA 기반 특징 데이터 (센서 트렌드 + RUL) — 필수
        model_wrapper: ModelWrapper 인스턴스 (LSTM) — 필수
    """

    def __init__(self, agv_id: int, scenario_df=None, model_wrapper=None):
        if model_wrapper is None or not getattr(model_wrapper, 'loaded', False):
            raise ValueError(
                f"AGV #{agv_id}: model_wrapper(LSTM)가 필수입니다. mock 모드는 제거되었습니다."
            )
        if scenario_df is None or 'RUL' not in scenario_df.columns:
            raise ValueError(
                f"AGV #{agv_id}: RUL 컬럼을 가진 scenario_df가 필수입니다. "
                f"먼저 data/simulation 시나리오를 생성하세요 "
                f"(python -m src.simulation.generate_scenarios)."
            )

        self.id = agv_id
        self._scenario = scenario_df
        self._model = model_wrapper
        self._scenario_pos = 0.0

        # --- ground-truth 초기 RUL: 항상 시나리오 RUL[0] (데이터 기반) ---
        initial_rul = float(self._scenario['RUL'].iloc[0])
        self.true_rul = initial_rul   # ground truth (실제 고장 결정)

        self.state = 'NORMAL'
        self.maintenance_time_left = 0
        self.scheduled_maintenance_time = None
        self.wait_time = 0

        # 시나리오와 시뮬레이션 속도 동기화 (true 수명 기준)
        self._sync_step_rate(initial_rul)

        # 예측값 (LSTM) — _refresh_samples에서 채움
        self.rul_mean = initial_rul
        self.rul_low = initial_rul
        self.rul_high = initial_rul
        self.rul_samples = []
        self._refresh_samples()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _get_seq_df(self):
        """모델 입력용 최근 seq_len 행 반환.

        워밍업(아직 seq_len행이 안 쌓인 초기 구간)에는 시나리오 첫 seq_len행을
        예측 시드로 사용해 LSTM이 항상 동작하도록 한다.
        """
        seq_len = self._model.seq_len
        end = min(int(self._scenario_pos) + 1, len(self._scenario))
        start = end - seq_len
        if start < 0:
            # 워밍업: 시나리오 첫 seq_len행으로 초기 예측 시드
            return self._scenario.iloc[0:seq_len]
        return self._scenario.iloc[start:end]

    def _sync_step_rate(self, life_hours: float):
        """시나리오 전체 길이를 주어진 수명 시간에 맞춰 1시간 단위로 매핑."""
        n_rows = len(self._scenario) if self._scenario is not None else 1
        self._step_rate = n_rows / max(float(life_hours), 1.0)

    def _reset_to_new_bearing(self):
        """정비 완료 후 새 베어링 상태로 되돌린다.

        ground truth(true_rul)는 해당 AGV 시나리오를 처음부터 재생한 새 베어링의
        실제 수명(= 시나리오 RUL[0], 데이터 기반)으로 리셋한다. 같은 시나리오를
        다시 읽으므로 LSTM 예측이 계속 ground truth와 정렬된다. 예측값은 리셋
        직후 _refresh_samples로 AGV 자신의 새 베어링 시나리오에서 다시 산출한다.
        """
        self.state = 'NORMAL'
        self._scenario_pos = 0.0

        # ground truth: 새 베어링 실제 수명 (해당 AGV 시나리오의 전체 수명)
        if 'RUL' in self._scenario.columns:
            new_life = float(self._scenario['RUL'].iloc[0])
        else:
            new_life = _NEW_BEARING_TRUE_LIFE_H
        self.true_rul = new_life
        self._sync_step_rate(new_life)

        # 예측: AGV 자신의 새 베어링 시나리오(첫 구간)로 재예측
        self._refresh_samples()

    def _refresh_samples(self):
        """MC Dropout으로 예측값(rul_mean/low/high/samples)을 갱신한다 (단일 AGV).

        항상 LSTM 예측을 사용한다(mock 폴백 제거). 예측값은 의사결정 전용이며
        ground truth(true_rul)에는 영향을 주지 않는다. 엔진은 다수 AGV를 한 번에
        처리하는 배치 경로(SimulationEngine.step)를 사용한다.
        """
        seq_df = self._get_seq_df()
        self._apply_prediction(self._model.predict(seq_df))

    def _apply_prediction(self, result):
        """예측 결과 (rul_mean, rul_low, rul_high, samples)를 반영한다."""
        if result is None:
            raise RuntimeError(f"AGV #{self.id}: LSTM 예측에 실패했습니다.")
        rul_mean, rul_low, rul_high, samples = result
        self.rul_mean = rul_mean
        self.rul_low = rul_low
        self.rul_high = rul_high
        self.rul_samples = [max(0.0, s) for s in samples]

    def _carry_forward(self):
        """LSTM 재실행 없이 예측값을 1시간만큼 감쇠 (refresh 사이 비용 절감)."""
        self.rul_mean = max(0.0, self.rul_mean - 1.0)
        self.rul_samples = [max(0.0, s - 1.0) for s in self.rul_samples]
        self.rul_low = max(0.0, self.rul_low - 1.0)
        self.rul_high = max(0.0, self.rul_high - 1.0)

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

    def _advance(self, current_time: int) -> bool:
        """상태별 진행 + ground truth 카운트다운.

        Returns:
            이번 스텝에 예측 갱신/상태 판정이 필요한 'active'(정상 열화) 상태면 True.
            정비중/대기중/고장 상태는 자체 처리 후 False.
        """
        # --- 정비 중 ---
        if self.state == 'MAINTENANCE':
            self.maintenance_time_left -= 1
            if self.maintenance_time_left <= 0:
                self._reset_to_new_bearing()
            return False

        # --- 대기 중 (FAILED / WAITING) ---
        if self.state in ['FAILED', 'WAITING']:
            self.wait_time += 1
            return False

        # --- 정상 열화: ground truth 카운트다운 ---
        self.true_rul = max(0.0, self.true_rul - 1.0)
        self._scenario_pos = min(
            self._scenario_pos + self._step_rate,
            len(self._scenario) - 1,
        )
        return True

    def _judge_state(self, current_time: int):
        """상태 판정. 실제 고장은 ground truth(true_rul)로만 결정한다.

        WARNING/CRITICAL 경보 상태는 의사결정에 쓰이지 않아 제거되었다(스케줄링은
        rul_samples(ILP)/rul_mean(B3 임계치)을 직접 사용). 상태는
        NORMAL / WAITING / FAILED / MAINTENANCE 네 가지만 존재한다.
        """
        if self.true_rul <= 0:
            self.state = 'FAILED'
        elif (self.scheduled_maintenance_time is not None
              and self.scheduled_maintenance_time <= current_time):
            self.state = 'WAITING'
            self.scheduled_maintenance_time = None
        else:
            self.state = 'NORMAL'

    def update_state(self, current_time: int):
        """단일 AGV 1시간 진행 (standalone/테스트용).

        SimulationEngine은 다수 AGV 예측을 한 번에 처리하는 배치 경로(step)를 쓴다.
        """
        if not self._advance(current_time):
            return
        if current_time % 5 == 0:
            self._refresh_samples()
        else:
            self._carry_forward()
        self._judge_state(current_time)


class SimulationEngine:
    """AGV PHM 시뮬레이션 엔진.

    각 AGV의 초기 ground-truth RUL은 항상 자신의 시나리오 RUL[0]에서 도출된다
    (임의 지정 불가). 모든 전략이 같은 시나리오 파일을 읽으므로 동일 ground truth가
    자동으로 보장된다.

    Args:
        num_agvs     : AGV 대수 (기본값 10, 100 등 자유 설정)
        model_wrapper: ModelWrapper 인스턴스 (필수, mock 모드 제거됨)
    """

    def __init__(self, num_agvs: int = 10, model_wrapper=None, K: int = 2):
        if model_wrapper is None or not getattr(model_wrapper, 'loaded', False):
            raise ValueError(
                "SimulationEngine: model_wrapper(LSTM)가 필수입니다. mock 모드는 제거되었습니다. "
                "ModelWrapper()를 생성해 주입하세요."
            )

        self.K = K  # 동시 정비 최대 대수 (MILP scheduler.K와 일치시킬 것)
        self.model_wrapper = model_wrapper

        # 각 AGV의 시나리오 데이터 로드
        scenario_dfs_dict = {i: _load_scenario(i) for i in range(num_agvs)}

        # 정비 후 새 베어링 RUL 샘플 계산 (한 번만 수행, 모든 AGV에서 동일)
        self.new_bearing_rul_samples = None
        if num_agvs > 0 and scenario_dfs_dict[0] is not None:
            # 첫 번째 AGV의 초기 시나리오로 새 베어링 RUL 예측
            new_bearing_df = scenario_dfs_dict[0].iloc[:model_wrapper.seq_len]
            if len(new_bearing_df) >= model_wrapper.seq_len:
                result = model_wrapper.predict(new_bearing_df)
                if result:
                    _, _, _, rul_samples = result
                    self.new_bearing_rul_samples = rul_samples
        if self.new_bearing_rul_samples is None:
            raise RuntimeError(
                "SimulationEngine: 새 베어링 RUL 예측에 실패했습니다. "
                "data/simulation 시나리오와 LSTM 가중치를 확인하세요."
            )

        self.agvs = [
            AGV(
                agv_id=i,
                scenario_df=scenario_dfs_dict[i],  # 초기 RUL은 AGV가 시나리오에서 도출
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
        # 고장 이벤트 시각 (PHM Score 계산용)
        self.failure_events: list[int] = []   # FAILED 진입 시각

    def step(self):
        """시뮬레이션 1시간 진행."""
        self.current_time += 1
        ct = self.current_time

        # 1. 개별 AGV 상태 업데이트 (예측은 active AGV 전체를 한 번에 배치 처리)
        old_states = {a.id: a.state for a in self.agvs}
        active = [a for a in self.agvs if a._advance(ct)]

        if ct % 5 == 0 and active:
            # 예측 갱신 스텝: active AGV들의 최근 시퀀스를 1회 배치 forward로 예측
            dfs = [a._get_seq_df() for a in active]
            results = self.model_wrapper.predict_many(dfs)
            for a, res in zip(active, results):
                a._apply_prediction(res)
        else:
            for a in active:
                a._carry_forward()

        for a in active:
            a._judge_state(ct)

        # 이벤트 집계 (고장/경보)
        for agv in self.agvs:
            old_state = old_states[agv.id]
            if old_state != 'FAILED' and agv.state == 'FAILED':
                self.cost_accumulated += 300   # 비계획 고장 비용
                self.failure_count += 1
                self.failure_events.append(ct)

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
            # 우선순위: 고장 임박도(true_rul)가 높은 순서 (긴급도 우선)
            # RUL이 동일하다면(예: 둘 다 FAILED) 대기 시간이 긴 순서
            waiting_agvs.sort(
                key=lambda a: (a.true_rul, -a.wait_time)
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
        """ILP 최적화 결과를 각 AGV에 적용.

        Args:
            schedule_dict: {agv_id: delay_hours (int) or None}
        """
        for agv_id, start_delay in schedule_dict.items():
            if start_delay is not None:
                self.agvs[agv_id].scheduled_maintenance_time = (
                    self.current_time + start_delay
                )

    def get_in_progress_ids(self) -> set:
        """현재 정비 진행 중인 AGV id 집합 반환 (ILP 재최적화용)."""
        return {a.id for a in self.agvs if a.state == 'MAINTENANCE'}

    def get_in_progress_dict(self) -> dict:
        """현재 정비 진행 중인 AGV id와 남은 정비 시간(hours) 반환 (ILP 용량 계산용)."""
        return {a.id: a.maintenance_time_left for a in self.agvs if a.state == 'MAINTENANCE'}

    def get_scenario_dfs(self) -> dict:
        """각 AGV의 시나리오 DataFrame 반환 (ILP 정비 후 RUL 예측용)."""
        return {a.id: a._scenario for a in self.agvs}

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
        )
        t = self.current_time
        num_agvs = len(self.agvs)
        total_events = self.failure_count + self.planned_maint_count

        return {
            'mode'             : 'PHM-ILP',
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
            'mean_rul': round(float(np.mean([a.true_rul for a in self.agvs])), 1),
        }
