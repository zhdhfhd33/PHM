import os
import glob
import pandas as pd
import numpy as np

def load_pronostia_bearing_data(bearing_dir: str) -> pd.DataFrame:
    """
    주어진 베어링 디렉토리에서 모든 가속도 파일(acc_*.csv 또는 acc_*.txt)을 로드합니다.
    표준 PRONOSTIA 데이터 포맷을 가정합니다:
    - 파일 이름 규칙: acc_00001.csv, acc_00002.csv 등
    - 각 파일은 2560개의 샘플을 포함 (25.6kHz로 0.1초 동안 측정)
    - 컬럼 구성: 시간(hour), 분(minute), 초(second), 마이크로초(microsecond), 수평 가속도(horiz_acc), 수직 가속도(vert_acc)
    
    Args:
        bearing_dir: 하나의 베어링 테스트 데이터(csv/txt)가 포함된 디렉토리 경로.
        
    Returns:
        각 행이 하나의 파일(0.1초, 2560개 샘플)의 원시 데이터를 나타내는 pandas DataFrame.
    """
    file_pattern = os.path.join(bearing_dir, "acc_*.*")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise ValueError(f"{bearing_dir}에서 가속도 파일을 찾을 수 없습니다.")
        
    data_list = []
    
    for i, file_path in enumerate(files):
        try:
            # PRONOSTIA 파일은 종종 세미콜론이나 쉼표로 구분되며 헤더가 없습니다.
            # 컬럼: Hour, Minute, Second, Microsecond, Horiz_acc, Vert_acc
            df = pd.read_csv(file_path, header=None, sep=None, engine='python')
            
            # 수평 및 수직 가속도 데이터만 추출
            # 정확한 포맷에 따라 컬럼이 4, 5번일 수 있습니다.
            if df.shape[1] >= 6:
                h_acc = df.iloc[:, 4].values
                v_acc = df.iloc[:, 5].values
            else:
                # 포맷이 다를 경우 마지막 두 개의 컬럼을 사용
                h_acc = df.iloc[:, -2].values
                v_acc = df.iloc[:, -1].values
                
            data_list.append({
                'timestamp_idx': i,  # 상대적 시간 인덱스 (예: 10초 단위)
                'h_acc': h_acc,
                'v_acc': v_acc
            })
            
        except Exception as e:
            print(f"파일 읽기 오류 {file_path}: {e}")
            
    return pd.DataFrame(data_list)
