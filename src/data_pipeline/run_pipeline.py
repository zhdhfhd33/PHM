import os
import sys
import glob
import pandas as pd

# Add the project root to sys.path
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_PROJ_ROOT)

from src.data_pipeline.loader import load_pronostia_bearing_data
from src.data_pipeline.features import extract_features

# FEMTO/PRONOSTIA는 같은 베어링이 Test_set(중간 절단 런)과 Full_Test_Set(완주 런)에
# 동명으로 존재한다. 절단 런은 마지막 스냅샷이 고장 지점이 아니어서 RUL 라벨이
# 거짓이 되므로, 완주 런(Full_Test_Set)만 사용하고 Test_set은 제외한다.
_EXCLUDED_PARENT_DIRS = {"Test_set"}

def main():
    raw_dir = os.path.join(_PROJ_ROOT, "data", "raw")
    processed_dir = os.path.join(_PROJ_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    bearing_dirs = [d for d in glob.glob(os.path.join(raw_dir, "**", "Bearing*"), recursive=True) if os.path.isdir(d)]

    if not bearing_dirs:
        print("data/raw 디렉토리에 Bearing 데이터 폴더가 없습니다.")
        return

    all_features = []

    for b_dir in bearing_dirs:
        bearing_name = os.path.basename(b_dir)
        parent_name = os.path.basename(os.path.dirname(b_dir))
        if parent_name in _EXCLUDED_PARENT_DIRS:
            print(f"Skipping {parent_name}/{bearing_name} (절단 런 - Full_Test_Set 완주 런 사용)")
            continue
        print(f"Processing {parent_name}/{bearing_name}...")
        
        try:
            # 1. 센서 원시 데이터 로드
            df_raw = load_pronostia_bearing_data(b_dir)
            
            # 2. 특징 추출 (Feature extraction)
            df_features = extract_features(df_raw)
            
            # 3. 베어링 이름 및 레이블(RUL) 추가
            # 시뮬레이션을 위해 마지막 인덱스를 Failure 지점으로 간주하고 RUL 부여
            df_features['bearing_name'] = bearing_name
            max_idx = df_features['timestamp_idx'].max()
            df_features['RUL'] = max_idx - df_features['timestamp_idx']
            
            all_features.append(df_features)
            
        except Exception as e:
            print(f"Error processing {bearing_name}: {e}")
            
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        out_path = os.path.join(processed_dir, "features_all.csv")
        final_df.to_csv(out_path, index=False)
        print(f"특징 추출 완료! shape: {final_df.shape}, 저장 경로: {out_path}")
    else:
        print("특징 추출된 데이터가 없습니다.")

if __name__ == "__main__":
    main()
