"""
PHM Baseline 시뮬레이터

PHM 시스템이 없는 2가지 대안적 정비 전략을 시뮬레이션하여
MILP 기반 PHM과 정량 비교한다.

전략:
  B1 NO_MAINT  : 정비 없이 고장 날 때까지 운영 (현행 최악 케이스)
  B2 TBM       : 고정 주기(tbm_interval h)마다 예방 정비 (전통적 TBM)

공통 가정:
  - AGV 10대, 초기 RUL = engine._DEFAULT_RULS 와 일치
  - 계획 정비 비용  : 50 만원/건  (8h 소요)
  - 비계획 고장 비용: 300 만원/건 (16h 소요)
  - 동시 정비 최대  : K=2대 (프로젝트 기본 정비 베이)
"""
import numpy as np


_MAINT_PLANNED_H   = 8   # 계획 정비 소요 시간 (engine과 일치)
_MAINT_EMERGENCY_H = 16  # 긴급 정비 소요 시간 (engine과 일치)
_C_PLANNED         = 50  # 계획 정비 비용 (만원)
_C_FAILURE         = 300 # 비계획 고장 비용 (만원)
_DEFAULT_RULS      = [280, 220, 175, 145, 120, 100, 75, 50, 28, 10]  # engine._DEFAULT_RULS와 동일
_INIT_RUL          = lambda i: _DEFAULT_RULS[i] if i < len(_DEFAULT_RULS) else 280.0


class BaselineSimulator:
    """PHM 미적용 Baseline 시뮬레이터.

    Args:
        num_agvs     : AGV 대수 (기본 10)
        mode         : 'NO_MAINT' | 'TBM'
        tbm_interval : TBM 모드 — 정비 주기 (h), 기본 500h
        K            : 동시 정비 최대 슬롯 수 (기본 2)
    """

    MODES = ('NO_MAINT', 'TBM')

    def __init__(self, num_agvs: int = 10, mode: str = 'NO_MAINT',
                 tbm_interval: int = 500, cbm_threshold: int = 48,
                 K: int = 2, initial_ruls: list = None):
        # Explicitly defined __init__ with mode
        assert mode in self.MODES, f"mode must be one of {self.MODES}"
        self.num_agvs = num_agvs
        self.mode = mode
        self.tbm_interval = tbm_interval
        self.cbm_threshold = cbm_threshold
        self.K = K
        self.initial_ruls = initial_ruls

        self._reset()

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 상태 초기화
    # ──────────────────────────────────────────────────────────────────────────

    def _reset(self):
        n = self.num_agvs
        if self.initial_ruls is not None:
            self.ruls = [float(r) for r in self.initial_ruls]
        else:
            self.ruls = [float(_INIT_RUL(i)) for i in range(n)]
        self.states           = ['NORMAL'] * n   # NORMAL / MAINTENANCE / FAILED
        self.maintenance_left = [0] * n
        self.current_time     = 0

        # ── 비용 추적 ──
        self.cost_accumulated  = 0.0
        self.failure_count     = 0
        self.planned_maint_count = 0

        # ── 지표 추적 ──
        self.total_downtime    = 0   # 정지 시간 누적 (AGV·h)
        self.total_maint_time  = 0   # 정비 중 시간 누적 (AGV·h)
        self.total_wait_time   = 0   # 대기 시간 누적 (AGV·h, K 슬롯 포화 시)
        self.max_concurrent_down = 0 # 동시 정지 최대 대수

        # ── 이력 ──
        self.cost_history: list[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # 스텝 실행
    # ──────────────────────────────────────────────────────────────────────────

    def step(self):
        """1시간 진행."""
        self.current_time += 1

        # 1. 정비 중인 AGV 카운트다운
        for i in range(self.num_agvs):
            if self.states[i] == 'MAINTENANCE':
                self.maintenance_left[i] -= 1
                if self.maintenance_left[i] <= 0:
                    self.states[i] = 'NORMAL'
                    self.ruls[i] = 3000.0   # 정비 후 RUL 리셋

        # 2. 정상/고장 AGV RUL 감소 및 고장 판정
        for i in range(self.num_agvs):
            if self.states[i] == 'MAINTENANCE':
                continue
            self.ruls[i] = max(0.0, self.ruls[i] - 1.0)
            if self.ruls[i] <= 0 and self.states[i] != 'FAILED':
                self.states[i] = 'FAILED'
                self.failure_count += 1
                self.cost_accumulated += _C_FAILURE

        # 3. 모드별 정비 트리거
        if self.mode == 'TBM':
            self._trigger_tbm()
        # NO_MAINT: 정비 없음

        # 4. 고장난 AGV 긴급 정비 처리 (B1 포함: 고장 후 자동 복구)
        self._assign_emergency_slots()

        # 5. 정지 시간 누적
        down = sum(1 for s in self.states if s in ('FAILED', 'MAINTENANCE'))
        self.total_downtime += down
        self.total_maint_time += sum(1 for s in self.states if s == 'MAINTENANCE')
        self.max_concurrent_down = max(self.max_concurrent_down, down)

        self.cost_history.append(self.cost_accumulated)

    # ──────────────────────────────────────────────────────────────────────────
    # 모드별 정비 트리거
    # ──────────────────────────────────────────────────────────────────────────

    def _trigger_tbm(self):
        """B2 TBM: 고정 주기마다 NORMAL 상태 AGV를 계획 정비 예약."""
        if self.current_time % self.tbm_interval != 0:
            return
        for i in range(self.num_agvs):
            if self.states[i] == 'NORMAL':
                # 슬롯 여유 확인
                maint_count = sum(1 for s in self.states if s == 'MAINTENANCE')
                if maint_count < self.K:
                    self.states[i] = 'MAINTENANCE'
                    self.maintenance_left[i] = _MAINT_PLANNED_H
                    self.planned_maint_count += 1
                    self.cost_accumulated += _C_PLANNED

    def _assign_emergency_slots(self):
        """FAILED 상태 AGV를 슬롯 여유 시 긴급 정비 배정 (모든 모드 공통)."""
        maint_count = sum(1 for s in self.states if s == 'MAINTENANCE')
        available = self.K - maint_count
        if available <= 0:
            return
        for i in range(self.num_agvs):
            if available <= 0:
                break
            if self.states[i] == 'FAILED':
                self.states[i] = 'MAINTENANCE'
                self.maintenance_left[i] = _MAINT_EMERGENCY_H
                available -= 1

    # ──────────────────────────────────────────────────────────────────────────
    # 지표 반환
    # ──────────────────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """현재 시뮬레이션 성과 지표 딕셔너리 반환."""
        from src.simulation.metrics import (
            calc_mtbf, calc_mttr, calc_availability,
            calc_unplanned_failure_rate, calc_pf_ratio,
            calc_slot_utilization, calc_avg_wait_per_event,
            calc_cost_savings_rate, calc_annual_savings,
        )
        t = self.current_time
        total_events = self.failure_count + self.planned_maint_count

        return {
            'mode'            : self.mode,
            'cost'            : round(self.cost_accumulated, 1),
            'failure_count'   : self.failure_count,
            'planned_count'   : self.planned_maint_count,
            'availability'    : calc_availability(t, self.total_downtime, self.num_agvs),
            'mtbf'            : calc_mtbf(t, self.failure_count),
            'mttr'            : calc_mttr(self.total_maint_time, total_events),
            'ufr'             : calc_unplanned_failure_rate(self.failure_count,
                                                             self.planned_maint_count),
            'pf_ratio'        : calc_pf_ratio(self.planned_maint_count, self.failure_count),
            'slot_utilization': calc_slot_utilization(self.total_maint_time, t, self.K),
            'avg_wait'        : calc_avg_wait_per_event(self.total_wait_time, total_events),
            'total_downtime'  : self.total_downtime,
            'mean_rul'        : round(float(np.mean(self.ruls)), 1),
        }

    def get_summary(self) -> dict:
        """app.py 호환 요약 딕셔너리."""
        return {
            'current_time'     : self.current_time,
            'cost_accumulated' : self.cost_accumulated,
            'failure_count'    : self.failure_count,
            'planned_maint_count': self.planned_maint_count,
            'states'           : list(self.states),
        }
