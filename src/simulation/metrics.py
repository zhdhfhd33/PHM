"""
PHM 공통 성능 평가지표 계산 모듈

신뢰성(Reliability), 정비 효율(Maintenance Efficiency),
경제성(Economic), 예측 품질(Prediction Quality) 지표를 산출한다.
"""
import numpy as np
import pandas as pd
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# A. 신뢰성 (Reliability) 지표
# ──────────────────────────────────────────────────────────────────────────────

def calc_mtbf(total_time: float, failure_count: int) -> Optional[float]:
    """MTBF (Mean Time Between Failures) — 고장 간 평균 가동 시간 (h).

    Args:
        total_time    : 총 시뮬레이션 시간 (h)
        failure_count : 비계획 고장 발생 횟수

    Returns:
        MTBF (h). 고장이 0건이면 None.
    """
    if failure_count <= 0:
        return None
    return round(total_time / failure_count, 1)


def calc_mttr(total_maint_time: float, maint_count: int) -> Optional[float]:
    """MTTR (Mean Time To Repair) — 평균 수리(정비) 소요 시간 (h).

    Args:
        total_maint_time : 총 정비 소요 시간 (h) — MAINTENANCE 상태 누적
        maint_count      : 총 정비 이벤트 수 (계획 + 비계획)

    Returns:
        MTTR (h). 정비 이벤트가 0건이면 None.
    """
    if maint_count <= 0:
        return None
    return round(total_maint_time / maint_count, 2)


def calc_availability(total_time: float, total_downtime: float,
                       num_agvs: int = 10) -> float:
    """시스템 가용률 (%) — 전체 AGV·h 중 실제 가동 비율.

    Args:
        total_time   : 시뮬레이션 총 시간 (h)
        total_downtime: 전 AGV 합산 정지 시간 (FAILED+MAINTENANCE+WAITING AGV·h)
        num_agvs     : AGV 대수

    Returns:
        가용률 (%). 0~100.
    """
    total_possible = total_time * num_agvs
    if total_possible <= 0:
        return 100.0
    avail = (total_possible - total_downtime) / total_possible * 100
    return round(max(0.0, min(100.0, avail)), 2)


def calc_unplanned_failure_rate(failure_count: int,
                                 planned_count: int) -> Optional[float]:
    """비계획 고장률 (UFR, %) — 전체 정비 중 비계획 비율.

    Args:
        failure_count : 비계획 고장 횟수
        planned_count : 계획 정비 횟수

    Returns:
        UFR (%). 정비 이벤트 없으면 None.
    """
    total = failure_count + planned_count
    if total <= 0:
        return None
    return round(failure_count / total * 100, 1)


def calc_pf_ratio(planned_count: int, failure_count: int) -> str:
    """PF 비율 (Planned:Failure) — 문자열 형태로 반환.

    높을수록 예지보전 효과가 좋음 (예: '7:3' → 계획 정비 70%).
    """
    return f"{planned_count}:{failure_count}"


# ──────────────────────────────────────────────────────────────────────────────
# B. 정비 효율 (Maintenance Efficiency) 지표
# ──────────────────────────────────────────────────────────────────────────────

def calc_slot_utilization(total_maint_time: float, total_time: float,
                           K: int = 2) -> float:
    """정비 슬롯 이용률 (%) — K 슬롯 중 실제 활용된 비율.

    Args:
        total_maint_time : 총 정비 시간 누적 (AGV·h)
        total_time       : 시뮬레이션 총 시간 (h)
        K                : 동시 정비 최대 슬롯 수

    Returns:
        슬롯 이용률 (%).
    """
    max_possible = total_time * K
    if max_possible <= 0:
        return 0.0
    return round(min(100.0, total_maint_time / max_possible * 100), 1)


def calc_avg_wait_per_event(total_wait_time: float,
                              total_events: int) -> Optional[float]:
    """정비 이벤트당 평균 대기 시간 (h).

    Args:
        total_wait_time : 총 대기 시간 (AGV·h)
        total_events    : 총 정비 이벤트 수

    Returns:
        평균 대기 (h). 이벤트 없으면 None.
    """
    if total_events <= 0:
        return None
    return round(total_wait_time / total_events, 2)


# ──────────────────────────────────────────────────────────────────────────────
# C. 경제성 (Economic) 지표
# ──────────────────────────────────────────────────────────────────────────────

def calc_cost_savings_rate(phm_cost: float, baseline_cost: float) -> Optional[float]:
    """비용 절감률 (%).

    Args:
        phm_cost      : PHM 시스템 누적 비용 (만원)
        baseline_cost : Baseline 누적 비용 (만원)

    Returns:
        절감률 (%). Baseline이 0이면 None.
    """
    if baseline_cost <= 0:
        return None
    return round((baseline_cost - phm_cost) / baseline_cost * 100, 1)


def calc_annual_savings(cost_savings: float, sim_time_hours: float) -> Optional[float]:
    """연간 환산 절감액 (만원/년).

    Args:
        cost_savings    : 누적 절감액 (만원)
        sim_time_hours  : 시뮬레이션 총 시간 (h)

    Returns:
        연간 절감액 (만원). 시뮬 시간이 0이면 None.
    """
    if sim_time_hours <= 0:
        return None
    daily_savings = cost_savings / (sim_time_hours / 24)
    return round(daily_savings * 365, 0)


def calc_roi(cost_savings: float, phm_system_cost: float) -> Optional[float]:
    """ROI (Return on Investment, %).

    PHM 시스템 운영 비용 대비 절감액 비율.

    Args:
        cost_savings     : PHM 도입으로 절감된 누적 비용 (만원)
        phm_system_cost  : PHM 시스템 운영 비용 (만원, 예: 라이선스+인건비)

    Returns:
        ROI (%). PHM 비용이 0이면 None.
    """
    if phm_system_cost <= 0:
        return None
    return round((cost_savings - phm_system_cost) / phm_system_cost * 100, 1)


# ──────────────────────────────────────────────────────────────────────────────
# D. 예측 품질 (Prediction Quality) 지표
# ──────────────────────────────────────────────────────────────────────────────

def calc_phm_score(rul_pred_list: list, rul_actual_list: list) -> Optional[float]:
    """PHM Challenge Score — 지연 예측에 더 큰 패널티를 부여하는 비대칭 손실.

    공식:
        error = RUL_pred - RUL_actual
        s_i = exp(-error/13) - 1  if error < 0  (조기 예측, early)
        s_i = exp(error/10) - 1   if error >= 0  (지연 예측, late — 더 큰 패널티)
        PHM Score = sum(s_i)

    낮을수록(0에 가까울수록) 좋음.

    Args:
        rul_pred_list   : 예측 RUL 리스트 (h)
        rul_actual_list : 실제 RUL 리스트 (h)

    Returns:
        PHM Score. 빈 리스트면 None.
    """
    if not rul_pred_list or not rul_actual_list:
        return None
    scores = []
    for pred, actual in zip(rul_pred_list, rul_actual_list):
        error = pred - actual
        if error < 0:   # 조기 예측 (early)
            scores.append(np.exp(-error / 13) - 1)
        else:           # 지연 예측 (late) — 더 큰 패널티
            scores.append(np.exp(error / 10) - 1)
    return round(float(np.sum(scores)), 2)


def calc_calibration(rul_samples_list: list,
                      rul_actual_list: list,
                      alpha: float = 0.95) -> Optional[float]:
    """불확실성 보정도 (Calibration) — 실제값이 신뢰구간 안에 드는 비율.

    이상적으로는 95% CI를 사용했을 때 실제값의 95%가 CI 안에 있어야 함.

    Args:
        rul_samples_list : AGV별 MC Dropout RUL 샘플 [[s1, s2, ...], ...]
        rul_actual_list  : 실제 RUL 리스트
        alpha            : 신뢰 수준 (0.95 = 95%)

    Returns:
        보정도 (0~1). 1에 가까울수록 좋음.
    """
    if not rul_samples_list or not rul_actual_list:
        return None

    hit = 0
    total = 0
    lower_p = (1 - alpha) / 2 * 100
    upper_p = (1 + alpha) / 2 * 100

    for samples, actual in zip(rul_samples_list, rul_actual_list):
        if not samples:
            continue
        lo = np.percentile(samples, lower_p)
        hi = np.percentile(samples, upper_p)
        if lo <= actual <= hi:
            hit += 1
        total += 1

    if total == 0:
        return None
    return round(hit / total, 3)


def calc_early_warning_rate(warning_events: list, failure_events: list,
                             lookahead_h: float = 72.0) -> Optional[float]:
    """조기 경보 적중률 — 고장 전 lookahead_h 이내에 경보가 발령된 비율.

    Args:
        warning_events : 경보 발생 시각 리스트 (h)
        failure_events : 고장 발생 시각 리스트 (h)
        lookahead_h    : 경보 인정 범위 (h), 기본 72h

    Returns:
        적중률 (0~1). 고장 없으면 None.
    """
    if not failure_events:
        return None

    hit = 0
    for ft in failure_events:
        # ft 이전 lookahead_h 이내에 경보가 있었는지 확인
        for wt in warning_events:
            if ft - lookahead_h <= wt < ft:
                hit += 1
                break

    return round(hit / len(failure_events), 3)


# ──────────────────────────────────────────────────────────────────────────────
# 종합 비교 테이블 빌더
# ──────────────────────────────────────────────────────────────────────────────

def build_comparison_table(
    metrics_list: list[dict],
) -> pd.DataFrame:
    """여러 정비 방식의 종합 성과 비교 테이블 생성.

    Args:
        metrics_list: 각 방식의 지표 딕셔너리 리스트.
                      각 딕셔너리는 'mode' 키를 포함해야 함.

    Returns:
        pandas DataFrame (지표 × 방식)
    """

    def _fmt(val, unit='', default='—'):
        if val is None:
            return default
        if isinstance(val, float):
            return f"{val:,.1f}{unit}"
        return f"{val}{unit}"

    # 행 정의
    metric_labels = [
        ("총 비용 (만원)", "cost", ""),
        ("비계획 고장 횟수", "failure_count", "건"),
        ("계획 정비 횟수", "planned_count", "건"),
        ("AGV 가용률 (%)", "availability", "%"),
        ("MTBF (h)", "mtbf", ""),
        ("MTTR (h)", "mttr", ""),
        ("PF 비율", "pf_ratio", ""),
        ("비용 절감률 vs B1", "savings_rate_vs_b1", "%"),
        ("평균 대기 시간 (h/건)", "avg_wait", ""),
        ("현재 평균 RUL (h)", "mean_rul", ""),
    ]

    # 데이터 구성
    columns = ["지표"] + [m.get('mode', 'Unknown') for m in metrics_list]
    data = []

    for label, key, unit in metric_labels:
        row = [label]
        for m in metrics_list:
            val = m.get(key)
            if key == 'pf_ratio':
                row.append(val if val is not None else '—')
            elif key == 'savings_rate_vs_b1' and m.get('mode') == 'B1 No-Maint':
                row.append("기준")
            else:
                row.append(_fmt(val, unit))
        data.append(row)

    df = pd.DataFrame(data, columns=columns)
    return df
