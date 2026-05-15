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

def calculate_coefficients(P_fail, N_AGV, T_start, T, C_planned=50, C_failure=300):
    """
    MILP 모델의 목적 함수 계수(coeff)를 계산합니다.
    coeff[i][t] = C_planned + (C_failure - C_planned)*P_fail[i][t] - C_failure*P_fail[i][T]
    
    계수가 음수이면 시각 t에 유지보수를 스케줄링하는 것이 비용 효율적임을 의미합니다.
    
    Args:
        P_fail: calculate_p_fail 함수에서 반환된 확률 딕셔너리
        N_AGV: AGV 대수
        T_start: 유지보수를 시작할 수 있는 최대 시각 슬롯 (일반적으로 T - d + 1)
        T: 계획 지평 (단위: 시간)
        C_planned: 계획 정비 비용
        C_failure: 비계획 고장 비용
        
    Returns:
        coeff: dict {(i, t): 비용 계수 (float)}
    """
    coeff = {}
    for i in range(N_AGV):
        for t in range(T_start):
            # 시간 t에 정비를 수행할 경우의 비용과 지평 T 내에 정비하지 않을 경우의 비용 차이
            cost_if_maintained = C_planned + (C_failure - C_planned) * P_fail[i][t]
            cost_if_ignored = C_failure * P_fail[i][T]
            
            coeff[i, t] = cost_if_maintained - cost_if_ignored
            
    return coeff
