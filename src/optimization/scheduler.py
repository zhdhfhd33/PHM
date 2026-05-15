"""
MILP 기반 AGV 유지보수 스케줄러

계획서 Layer 3 정식화:
  결정 변수 : x[i, t] ∈ {0, 1}  — AGV i의 정비를 시각 t에 시작할지 여부
  계획 지평 : T = 24h (1일), 시간 단위 이산 슬롯 (총 24개)
  목적 함수 : P_fail 기반 기대 비용 최소화
  제약 1    : 각 AGV 최대 1회 정비
  제약 2    : 동시 정비 ≤ K=2대
  재실행 주기: 매 24시간 롤링 호라이즌

목적 함수 계수:
  coeff[i, t] = C_planned + (C_failure - C_planned)*P_fail[i][t]
                - C_failure * P_fail[i][T]

  - coeff < 0  → t시각에 정비하면 지평 T 내 방치보다 비용 절감 (정비 권장)
  - coeff ≥ 0  → 아직 정비할 필요 없음 (솔버가 x=0 선택)
"""
import pulp
from .cost_model import calculate_p_fail, calculate_coefficients


def solve_maintenance_schedule(
    rul_samples,
    in_progress=None,
    N_AGV: int = 10,
    T: int = 24,          # 계획서 기준 24h (1일 롤링 호라이즌)
    d: int = 8,           # 정비 소요 시간: engine._MAINT_PLANNED_H와 일치
    K: int = 2,           # 동시 정비 최대 대수 (프로젝트 기본: 2대)
    C_planned: int = 50,  # 계획 정비 비용 (만원)
    C_failure: int = 300, # 비계획 고장 비용 (만원)
):
    """AGV 유지보수 최적 스케줄 계산 (MILP / CBC 솔버).

    Args:
        rul_samples  : {agv_id: [MC Dropout RUL 샘플 리스트 (단위: 시간)]}
        in_progress  : 현재 정비 진행 중인 AGV id와 남은 시간 딕셔너리 ({id: remaining_time}) 또는 집합 (set)
        N_AGV        : AGV 대수
        T            : 계획 지평 (시간) — 기본값 24h (1일)
        d            : 정비 소요 시간 (시간)
        K            : 동시 정비 최대 대수
        C_planned    : 계획 정비 비용 (만원)
        C_failure    : 비계획 고장 비용 (만원)

    Returns:
        schedule     : {agv_id: 정비 시작 딜레이 (시간) 또는 None}
        obj_value    : 목적 함수 값 (기대 비용, 만원)

    Note:
        schedule[i] = t  → 현재 시각으로부터 t시간 후에 정비 시작
        schedule[i] = None → 이번 지평 내 정비 불필요
    """
    if in_progress is None:
        in_progress = {}
    elif isinstance(in_progress, set):
        # 하위 호환성 유지: set으로 들어오면 기본 d를 남은 시간으로 가정 (혹은 이미 차있다고 가정)
        in_progress = {agv_id: d for agv_id in in_progress}

    T_start = T - d + 1  # 정비를 시작할 수 있는 마지막 슬롯

    # 0. 시각 t에서의 가용 슬롯 계산 (K_available)
    K_available = {}
    for t in range(T):
        used_slots = sum(1 for rem_time in in_progress.values() if rem_time > t)
        K_available[t] = max(0, K - used_slots)

    # 1. P_fail 및 목적 함수 계수 계산
    P_fail = calculate_p_fail(rul_samples, N_AGV, T)
    coeff = calculate_coefficients(P_fail, N_AGV, T_start, T, C_planned, C_failure)

    # 2. 최적화 문제 정의 (최소화)
    prob = pulp.LpProblem("AGV_Maintenance_Scheduling_24h", pulp.LpMinimize)

    # 결정 변수: x[i, t] ∈ {0, 1}
    x = pulp.LpVariable.dicts(
        "x",
        [(i, t) for i in range(N_AGV) for t in range(T_start)],
        cat='Binary'
    )

    # 3. 목적 함수
    prob += pulp.lpSum(
        x[i, t] * coeff[i, t]
        for i in range(N_AGV)
        for t in range(T_start)
    )

    # 4. 제약 조건

    # 제약 1: 각 AGV 최대 1회 정비
    #   - 이미 정비 중인 AGV는 제외 (in_progress)
    #   - coeff[i,t] ≥ 0 인 AGV는 솔버가 자동으로 x=0 선택 (패스)
    for i in range(N_AGV):
        if i in in_progress:
            # 정비 진행 중 → 재배정 금지
            prob += pulp.lpSum(x[i, t] for t in range(T_start)) <= 0
        else:
            prob += pulp.lpSum(x[i, t] for t in range(T_start)) <= 1

    # 제약 2: 임의의 시각 t에서 동시 정비 대수 ≤ K_available[t]
    #   시각 t에 정비 중인 AGV: 시작 시각 t' ∈ [t-d+1, t] 범위
    for t in range(T):
        prob += pulp.lpSum(
            x[i, t_prime]
            for i in range(N_AGV) if i not in in_progress
            for t_prime in range(max(0, t - d + 1), min(t + 1, T_start))
        ) <= K_available[t]

    # 5. 풀이 (CBC 솔버, 로그 비활성화)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # 6. 결과 추출
    schedule = {}
    for i in range(N_AGV):
        schedule[i] = None
        for t in range(T_start):
            val = pulp.value(x[i, t])
            if val is not None and val > 0.5:
                schedule[i] = t
                break

    return schedule, pulp.value(prob.objective)



def threshold_schedule(agvs: list, threshold: float, K: int, in_progress: set) -> dict:
    """B3 Pred-CBM: rul_mean 기반 단순 임계치 스케줄러.

    Args:
        agvs       : SimulationEngine.agvs 리스트 (AGV 객체, .id/.rul_mean/.state 필요)
        threshold  : 정비 트리거 임계치 (시간). rul_mean < threshold이면 스케줄
        K          : 동시 정비 최대 슬롯 수
        in_progress: 현재 정비 중인 AGV id 집합 (set) — 제외 대상

    Returns:
        {agv_id: 0} — delay=0: 즉시 스케줄. solve_maintenance_schedule과 동일 형식.
    """
    candidates = [
        agv for agv in agvs
        if agv.id not in in_progress
        and agv.state not in ('MAINTENANCE', 'FAILED', 'WAITING')
        and agv.rul_mean < threshold
    ]
    candidates.sort(key=lambda a: a.rul_mean)  # 긴급도 우선 (rul_mean 오름차순)
    return {agv.id: 0 for agv in candidates[:K]}
