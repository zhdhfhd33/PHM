import numpy as np
import pandas as pd
from scipy.stats import kurtosis

def calculate_time_features(signal: np.ndarray) -> dict:
    """
    1D 신호 배열에 대한 시간 도메인 특징(Time-domain features)을 계산합니다.
    """
    # 결측치(NaN) 제거
    signal = signal[~np.isnan(signal)]
    
    if len(signal) == 0:
        return {'rms': 0, 'kurtosis': 0, 'crest_factor': 0, 'peak_to_peak': 0}
        
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal, fisher=False)  # 피어슨 정의 (정규분포 = 3.0)
    peak = np.max(np.abs(signal))
    crest_factor = peak / rms if rms > 0 else 0
    peak_to_peak = np.max(signal) - np.min(signal)
    
    return {
        'rms': rms,
        'kurtosis': kurt,
        'crest_factor': crest_factor,
        'peak_to_peak': peak_to_peak
    }

def calculate_freq_features(signal: np.ndarray, fs: float = 25600.0) -> dict:
    """
    1D 신호에 대한 주파수 도메인 특징(FFT 기반)을 계산합니다.
    실제 환경에서 BPFO(외륜 결함 주파수) 및 BPFI(내륜 결함 주파수)는
    베어링 형상 및 축 속도(RPM)에 따라 달라집니다.
    여기서는 특정 주파수 대역의 에너지를 프록시로 계산하거나 가정된 값을 사용합니다.
    """
    signal = signal[~np.isnan(signal)]
    n = len(signal)
    if n == 0:
        return {'bpfo_energy': 0, 'bpfi_energy': 0, 'high_freq_energy': 0}
        
    # FFT 연산
    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_mag = np.abs(fft_vals) / n
    
    # 베어링 결함을 위한 일반적인 주파수 대역 가정
    # (실제 환경에서는 동역학적 주파수로 대체되어야 함)
    # 예: 특정 RPM에서 BPFO는 100-150Hz, BPFI는 150-200Hz 부근
    
    bpfo_mask = (fft_freqs >= 100) & (fft_freqs <= 150)
    bpfi_mask = (fft_freqs >= 150) & (fft_freqs <= 200)
    high_freq_mask = (fft_freqs >= 1000)
    
    bpfo_energy = np.sum(fft_mag[bpfo_mask]**2)
    bpfi_energy = np.sum(fft_mag[bpfi_mask]**2)
    high_freq_energy = np.sum(fft_mag[high_freq_mask]**2)
    
    return {
        'bpfo_energy': bpfo_energy,
        'bpfi_energy': bpfi_energy,
        'high_freq_energy': high_freq_energy
    }

def extract_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    `load_pronostia_bearing_data`에서 로드된 DataFrame의 각 행에 대해 특징을 추출합니다.
    """
    features_list = []
    
    for idx, row in df_raw.iterrows():
        h_acc = row['h_acc']
        v_acc = row['v_acc']
        timestamp_idx = row['timestamp_idx']
        
        # 수평 가속도 특징
        h_time = calculate_time_features(h_acc)
        h_freq = calculate_freq_features(h_acc)
        
        # 수직 가속도 특징
        v_time = calculate_time_features(v_acc)
        v_freq = calculate_freq_features(v_acc)
        
        feat_dict = {'timestamp_idx': timestamp_idx}
        
        for k, v in h_time.items():
            feat_dict[f'h_{k}'] = v
        for k, v in h_freq.items():
            feat_dict[f'h_{k}'] = v
            
        for k, v in v_time.items():
            feat_dict[f'v_{k}'] = v
        for k, v in v_freq.items():
            feat_dict[f'v_{k}'] = v
            
        features_list.append(feat_dict)
        
    return pd.DataFrame(features_list)
