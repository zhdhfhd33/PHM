def calculate_p_fail(rul_samples, N_AGV, T):
    """
    AGV i가 시간 t 이전에 고장날 확률인 P_fail[i][t]를 계산합니다.
    
    Args:
        rul_samples: dict {i: [MC Dropout으로 추출한 RUL 샘플 리스트 (단위: 시간)]}
        N_AGV: AGV 대수
        T: 계획 지평 (단위: 시간)
        
    Returns:
        P_fail: dict {i: {t: 확률값}}
    """
    P_fail = {}
    for i in range(N_AGV):
        P_fail[i] = {}
        samples = rul_samples[i]
        num_samples = len(samples)
        
        for t in range(T + 1):
            if num_samples > 0:
                prob = sum(1 for s in samples if s < t) / num_samples
            else:
                prob = 0.0
            P_fail[i][t] = prob
            
    return P_fail

def calculate_coefficients(P_fail, N_AGV, T_start, T, C_planned=50, C_failure=300,
                          new_bearing_rul_samples=None, d=8):
    """
    ILP 모델의 목적 함수 계수(coeff)를 계산합니다.

    정비 후 새 베어링의 RUL(LSTM 예측값)을 고장 확률에 반영합니다.

    Args:
        P_fail: calculate_p_fail 함수에서 반환된 확률 딕셔너리 (정비 전 상태)
        N_AGV: AGV 대수
        T_start: 유지보수를 시작할 수 있는 최대 시각 슬롯 (일반적으로 T - d + 1)
        T: 계획 지평 (단위: 시간)
        C_planned: 계획 정비 비용 (만원)
        C_failure: 비계획 고장 비용 (만원)
        new_bearing_rul_samples: LSTM으로 예측한 새 베어링 RUL 샘플 리스트 (선택사항)
        d: 정비 소요 시간 (시간)

    Returns:
        coeff: dict {(i, t): 비용 계수 (float)}
    """
    coeff = {}
    for i in range(N_AGV):
        for t in range(T_start):
            # 정비하지 않는 경우: 현재 상태에서 지평 T까지 고장비용
            cost_if_ignored = C_failure * P_fail[i][T]

            # 정비하는 경우의 기대 비용.
            #   - 정비 시각 t 이전에 고장날 확률 P_fail[i][t]만큼은 계획정비(C_planned)가
            #     아니라 비계획 고장(C_failure)을 치르게 된다.
            #     → C_planned*(1-P_fail[t]) + C_failure*P_fail[t]
            #       = C_planned + (C_failure - C_planned)*P_fail[i][t]
            #   이 t-의존 항이 "조기 정비일수록 정비 전 고장 리스크 회피"라는 시간 선호를 부여한다.
            cost_if_maintained = (
                C_planned + (C_failure - C_planned) * P_fail[i][t]
            )

            # 정비 후 새 베어링의 고장 확률 계산 (LSTM 예측값 사용)
            if new_bearing_rul_samples is not None:
                remaining_time = T - t - d  # 정비 후 남은 지평
                if remaining_time > 0:
                    # 남은 지평 내에 고장날 확률
                    num_fail = sum(1 for s in new_bearing_rul_samples if s < remaining_time)
                    p_fail_after_maint = num_fail / len(new_bearing_rul_samples)
                    cost_if_maintained += C_failure * p_fail_after_maint

            coeff[i, t] = cost_if_maintained - cost_if_ignored

    return coeff
