import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

_HERE = Path(__file__).parent
_PROJ_ROOT = _HERE.parent.parent
_DATA_PATH = _PROJ_ROOT / "data" / "processed" / "features_all.csv"
_MODEL_PATH = _HERE / "lstm_rul.pt"
_SCALER_PATH = _HERE / "scaler.npz"

FEATURE_COLS = [
    'h_rms', 'h_kurtosis', 'h_crest_factor', 'h_peak_to_peak',
    'h_bpfo_energy', 'h_bpfi_energy', 'h_high_freq_energy',
    'v_rms', 'v_kurtosis', 'v_crest_factor', 'v_peak_to_peak',
    'v_bpfo_energy', 'v_bpfi_energy', 'v_high_freq_energy',
]


class RULLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout_rate: float = 0.3):
        super(RULLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # batch_first=True → 입력 형태: (batch, seq_len, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)

        # 추론 시(MC Dropout) 사용할 수 있도록 Dropout 명시적으로 추가
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)  # 단일 스칼라값(RUL) 예측

    def forward(self, x):
        # x 형태: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)

        # 마지막 타임 스텝의 출력 사용
        last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)

        out = self.dropout(last_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out.squeeze(1)  # (batch,)


class BearingDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def create_sequences(df: pd.DataFrame, sequence_length: int = 30):
    """베어링 그룹별로 슬라이딩 윈도우 시퀀스를 생성한다."""
    available_cols = [c for c in FEATURE_COLS if c in df.columns]

    sequences, labels = [], []

    for _, group in df.groupby('bearing_name'):
        group = group.sort_values('timestamp_idx')
        data = group[available_cols].values
        rul = group['RUL'].values

        for i in range(len(data) - sequence_length):
            sequences.append(data[i: i + sequence_length])
            labels.append(rul[i + sequence_length])

    return np.array(sequences), np.array(labels), len(available_cols)


def train_model(epochs: int = 150, sequence_length: int = 30,
                test_ratio: float = 0.2, batch_size: int = 64,
                lr: float = 1e-3):
    """LSTM RUL 모델 학습.

    - Train / Test 80/20 분리 (bearing 단위 분리로 데이터 누수 방지)
    - 최종 Test RMSE, MAE, MAPE 출력
    - 모델·스케일러를 src/models/ 에 저장
    """
    if not _DATA_PATH.exists():
        print(f"[train] Data not found: {_DATA_PATH}\n"
              "Please run 'python -m src.data_pipeline.run_pipeline' first.")
        return

    df = pd.read_csv(_DATA_PATH)

    # ----- Train / Test 분리 (베어링 단위) -----
    bearings = sorted(df['bearing_name'].unique())
    np.random.seed(42)
    np.random.shuffle(bearings)
    n_test = max(1, int(len(bearings) * test_ratio))
    test_bearings = set(bearings[:n_test])
    train_bearings = set(bearings[n_test:])

    train_df = df[df['bearing_name'].isin(train_bearings)]
    test_df = df[df['bearing_name'].isin(test_bearings)]

    print(f"[train] Bearings - train: {len(train_bearings)}, test: {len(test_bearings)}")

    X_train, y_train, input_dim = create_sequences(train_df, sequence_length)
    X_test, y_test, _ = create_sequences(test_df, sequence_length)

    print(f"[train] X_train: {X_train.shape}, X_test: {X_test.shape}, input_dim: {input_dim}")

    if len(X_train) == 0:
        print("[train] No training sequences. Exiting.")
        return

    # ----- 정규화 (Train 통계치 기준) -----
    X_mean = np.mean(X_train, axis=(0, 1), keepdims=True)  # (1, 1, feat)
    X_std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8

    X_train_s = (X_train - X_mean) / X_std
    X_test_s = (X_test - X_mean) / X_std if len(X_test) > 0 else X_test

    # ----- RUL 정규화 (0~1, Train max 기준) -----
    # MSE 손실이 큰 RUL 스케일에 압도되지 않도록 정규화 후 학습
    y_max = float(np.max(y_train)) + 1e-6
    y_train_s = y_train / y_max
    y_test_s = y_test / y_max if len(y_test) > 0 else y_test

    # ----- 데이터 로더 -----
    train_loader = DataLoader(
        BearingDataset(X_train_s, y_train_s),
        batch_size=batch_size, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    model = RULLSTM(input_dim=input_dim, hidden_dim=128,
                    num_layers=2, dropout_rate=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # ----- 학습 루프 -----
    best_loss = float('inf')
    print(f"[train] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bX.size(0)

        epoch_loss /= len(train_loader.dataset)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), _MODEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  TrainLoss: {epoch_loss:.4f}")

    print(f"[train] Best train loss: {best_loss:.6f}. Model saved.")

    # ----- 스케일러 저장 (X 정규화 + RUL 역정규화용 y_max 포함) -----
    np.savez(_SCALER_PATH, mean=X_mean, std=X_std, y_max=np.array([y_max]))
    print(f"[train] Scaler saved (y_max={y_max:.1f}h).")

    # ----- 테스트 평가 -----
    if len(X_test_s) == 0:
        print("[train] No test sequences - skipping evaluation.")
        return

    # Best 모델 로드
    best_model = RULLSTM(input_dim=input_dim, hidden_dim=128,
                          num_layers=2, dropout_rate=0.3)
    best_model.load_state_dict(torch.load(_MODEL_PATH, map_location='cpu',
                                           weights_only=True))
    best_model.eval()

    with torch.no_grad():
        x_t = torch.tensor(X_test_s, dtype=torch.float32)
        # 예측 후 역정규화 (시간 단위로 복원)
        preds = best_model(x_t).numpy() * y_max
        y_test_orig = y_test_s * y_max  # 역정규화된 정답

    rmse = float(np.sqrt(np.mean((preds - y_test_orig) ** 2)))
    mae = float(np.mean(np.abs(preds - y_test_orig)))
    mape_mask = y_test_orig > 0
    mape = float(np.mean(np.abs((preds[mape_mask] - y_test_orig[mape_mask])
                                 / y_test_orig[mape_mask])) * 100) if mape_mask.any() else float('nan')
    rel_rmse = rmse / (np.mean(y_test_orig) + 1e-6) * 100  # % of mean RUL

    print(f"\n[eval] ===== Test Results =====")
    print(f"  RMSE   : {rmse:.2f} h  ({rel_rmse:.1f}% of mean RUL)")
    print(f"  MAE    : {mae:.2f} h")
    print(f"  MAPE   : {mape:.1f} %")
    print(f"  Target : RMSE <= 10% of mean RUL")
    print(f"  Pass   : {'PASS' if rel_rmse <= 10 else 'FAIL - consider more epochs or tuning'}")
    print("=" * 35)

    # 결과 로그 저장
    log_dir = _PROJ_ROOT / "agent_log"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "lstm_eval_results.md"
    pass_str = 'PASS' if rel_rmse <= 10 else 'FAIL'
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# LSTM RUL Model Evaluation Results\n\n")
        f.write(f"- Epochs: {epochs}\n")
        f.write(f"- Train bearings: {len(train_bearings)}\n")
        f.write(f"- Test bearings: {len(test_bearings)}\n\n")
        f.write("## Metrics\n\n")
        f.write(f"| Metric | Value | Target |\n|---|---|---|\n")
        f.write(f"| RMSE | {rmse:.2f} h ({rel_rmse:.1f}% of mean RUL) | <= 10% of mean RUL |\n")
        f.write(f"| MAE | {mae:.2f} h | - |\n")
        f.write(f"| MAPE | {mape:.1f}% | - |\n")
        f.write(f"\n**Result**: {pass_str}\n")
    print(f"[eval] Results saved: {log_path}")


if __name__ == "__main__":
    train_model(epochs=150)
