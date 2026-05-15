import torch
import numpy as np

def enable_dropout(model):
    """
    Monte Carlo Dropout을 수행하기 위해 테스트 시에도 Dropout 레이어를 활성화합니다.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout_predict(model, x, num_samples=100):
    """
    Monte Carlo Dropout 추론을 수행하여 예측값의 분포를 얻습니다.
    
    Args:
        model: 학습된 PyTorch 모델.
        x: 입력 텐서, 형태: (batch, seq_len, input_dim).
        num_samples: 확률적 순전파(stochastic forward pass) 횟수.
        
    Returns:
        samples: 예측값을 포함하는 (batch, num_samples) 형태의 numpy 배열.
    """
    model.eval()  # 모델을 평가 모드로 설정 (예: BatchNorm 고정)
    enable_dropout(model)  # 단, Dropout은 강제로 활성화
    
    with torch.no_grad():
        predictions = []
        for _ in range(num_samples):
            out = model(x)
            predictions.append(out.cpu().numpy())
            
    # predictions는 (batch) 형태의 배열들의 리스트입니다.
    # 이를 (batch, num_samples) 형태의 배열로 변환합니다.
    samples = np.stack(predictions, axis=1)
    
    # 음수인 RUL을 0으로 클리핑합니다.
    samples = np.clip(samples, a_min=0, a_max=None)
    
    return samples

def get_rul_statistics(samples):
    """
    MC Dropout 샘플이 주어지면 평균 및 95% 신뢰 구간의 상/하한을 반환합니다.
    """
    rul_mean = np.mean(samples, axis=-1)
    rul_low = np.percentile(samples, 5, axis=-1)  # 5백분위수 (최악의 경우)
    rul_high = np.percentile(samples, 95, axis=-1) # 95백분위수 (최선의 경우)
    
    return rul_mean, rul_low, rul_high
