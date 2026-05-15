"""
AGV 시나리오 CSV 생성 스크립트

features_all.csv (17개 실제 베어링)를 기반으로 augmentation을 통해
agv_00 ~ agv_99 시나리오 파일을 생성한다.

- agv_00 ~ agv_09: 기존 파일이 있으면 유지, 없으면 생성
- agv_10 ~ agv_99: time-scaling + feature noise 로 새로 생성

Usage:
    python scripts/generate_scenarios.py [--n_agv 100]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJ = Path(__file__).parent.parent
_SRC  = _PROJ / "data" / "simulation"
_FEAT = _PROJ / "data" / "processed" / "features_all.csv"

# 기존 10개 AGV에 사용한 베어링 (순서 고정)
_ORIGINAL_BEARINGS = [
    "Bearing1_1", "Bearing1_2", "Bearing1_6", "Bearing3_2", "Bearing2_5",
    "Bearing3_1", "Bearing2_2", "Bearing2_7", "Bearing1_3", "Bearing2_3",
]

_FEATURE_COLS = [
    "h_rms", "h_kurtosis", "h_crest_factor", "h_peak_to_peak",
    "h_bpfo_energy", "h_bpfi_energy", "h_high_freq_energy",
    "v_rms", "v_kurtosis", "v_crest_factor", "v_peak_to_peak",
    "v_bpfo_energy", "v_bpfi_energy", "v_high_freq_energy",
]


def load_features() -> dict[str, pd.DataFrame]:
    """features_all.csv에서 베어링별 DataFrame 반환."""
    df = pd.read_csv(_FEAT)
    return {name: grp.reset_index(drop=True) for name, grp in df.groupby("bearing_name")}


def augment_bearing(df: pd.DataFrame, agv_id: int, rng: np.random.Generator) -> pd.DataFrame:
    """베어링 데이터를 time-scaling + feature noise로 augment.

    Args:
        df    : 원본 베어링 DataFrame (timestamp_idx, features, RUL)
        agv_id: 생성할 AGV ID
        rng   : numpy random generator
    """
    # time-scaling: 베어링 수명 ±30% 조정
    scale = rng.uniform(0.7, 1.3)
    n_original = len(df)
    n_new = max(50, int(n_original * scale))

    # 시간 축 리샘플링 (선형 보간)
    old_idx = np.linspace(0, n_original - 1, n_original)
    new_idx = np.linspace(0, n_original - 1, n_new)

    new_df = pd.DataFrame()
    new_df["timestamp_idx"] = np.arange(n_new)

    for col in _FEATURE_COLS:
        interp = np.interp(new_idx, old_idx, df[col].values)
        # feature noise: ±5% 가우시안
        noise_std = np.abs(interp).mean() * 0.05
        noisy = interp + rng.normal(0, noise_std, n_new)
        # 음수 방지 (RMS, energy 계열)
        new_df[col] = np.clip(noisy, 0, None)

    # RUL: n_new-1 에서 0으로 선형 감소
    new_df["RUL"] = np.maximum(0, np.round(np.linspace(n_new - 1, 0, n_new))).astype(int)
    new_df["bearing_name"] = df["bearing_name"].iloc[0] + f"_aug{agv_id}"
    new_df["agv_id"] = agv_id

    return new_df


def generate(n_agv: int = 100, seed: int = 2026):
    """n_agv개 시나리오 파일 생성."""
    _SRC.mkdir(parents=True, exist_ok=True)
    all_bearings = load_features()
    bearing_names = list(all_bearings.keys())

    rng = np.random.default_rng(seed)

    generated, skipped = 0, 0
    for agv_id in range(n_agv):
        out_path = _SRC / f"agv_{agv_id:02d}_scenario.csv"

        # agv_00~09: 기존 파일 있으면 유지
        if out_path.exists() and agv_id < 10:
            skipped += 1
            continue

        # 베어링 선택
        if agv_id < len(_ORIGINAL_BEARINGS):
            bearing_name = _ORIGINAL_BEARINGS[agv_id]
        else:
            bearing_name = rng.choice(bearing_names)

        if bearing_name not in all_bearings:
            bearing_name = rng.choice(bearing_names)

        base_df = all_bearings[bearing_name]

        if agv_id < 10 and not out_path.exists():
            # 원본 그대로 (augmentation 없이)
            out_df = base_df.copy()
            out_df["agv_id"] = agv_id
        else:
            out_df = augment_bearing(base_df, agv_id, rng)

        out_df.to_csv(out_path, index=False)
        generated += 1

    print(f"Done: generated={generated}, skipped(existing)={skipped}, total={n_agv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agv", type=int, default=100)
    parser.add_argument("--seed",  type=int, default=2026)
    args = parser.parse_args()
    generate(n_agv=args.n_agv, seed=args.seed)
