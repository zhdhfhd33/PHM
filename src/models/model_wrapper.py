import os
import torch
import numpy as np

from .lstm_rul import RULLSTM, FEATURE_COLS
from .inference import mc_dropout_predict, get_rul_statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_HERE, 'lstm_rul.pt')
_SCALER_PATH = os.path.join(_HERE, 'scaler.npz')

_NUM_MC_SAMPLES = 100  # 계획서 기준 N=100


class ModelNotTrainedError(FileNotFoundError):
    """모델 가중치 파일이 없을 때 발생하는 예외."""
    pass


class ModelWrapper:
    """LSTM RUL 모델 래퍼.

    모델 가중치가 있으면 로드하고 두 가지 인터페이스를 제공한다:
      - predict(features_df) → (rul_mean, rul_low, rul_high, rul_samples)
      - get_uncertainty_cv(features_df) → float (CV = std/mean)
    """

    def __init__(self, seq_len: int = 10, num_mc_samples: int = _NUM_MC_SAMPLES):
        self.seq_len = seq_len
        self.num_mc_samples = num_mc_samples
        self.model = None
        self.feat_mean = None  # shape (1, 1, n_feat)
        self.feat_std = None   # shape (1, 1, n_feat)
        self.y_max = 1.0       # RUL 역정규화용 (scaler에 저장)
        self._load()

    # ------------------------------------------------------------------
    # 내부 로드
    # ------------------------------------------------------------------

    def _load(self):
        missing = [p for p in (_MODEL_PATH, _SCALER_PATH) if not os.path.exists(p)]
        if missing:
            raise ModelNotTrainedError(
                "\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "[ModelWrapper] 오류: 학습된 가중치 파일이 없습니다.\n"
                f"  누락 파일: {', '.join(os.path.basename(p) for p in missing)}\n"
                "\n"
                "  먼저 모델을 학습하세요:\n"
                "    1) 데이터 파이프라인 실행:\n"
                "       python -m src.data_pipeline.run_pipeline\n"
                "    2) 모델 학습:\n"
                "       python -m src.models.lstm_rul\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
        try:
            scaler = np.load(_SCALER_PATH)
            self.feat_mean = scaler['mean']
            self.feat_std = scaler['std']
            input_dim = int(self.feat_mean.shape[-1])

            mdl = RULLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout_rate=0.2)
            mdl.load_state_dict(
                torch.load(_MODEL_PATH, map_location='cpu', weights_only=True)
            )
            mdl.eval()
            self.model = mdl
            # RUL 역정규화용 y_max (scaler에 저장된 경우 사용, 없으면 1.0)
            self.y_max = float(scaler['y_max'][0]) if 'y_max' in scaler else 1.0
            print(f"[ModelWrapper] LSTM RUL model loaded (input_dim={input_dim}, y_max={self.y_max:.1f}h)")
        except ModelNotTrainedError:
            raise
        except Exception as exc:
            raise RuntimeError(f"[ModelWrapper] 모델 로드 실패: {exc}") from exc

    @property
    def loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _prepare_input(self, features_df):
        """features_df → 정규화된 torch 텐서 (1, seq_len, n_feat)."""
        avail = [c for c in FEATURE_COLS if c in features_df.columns]
        if len(avail) < len(FEATURE_COLS) or len(features_df) < self.seq_len:
            return None
        data = features_df[avail].values[-self.seq_len:]          # (seq_len, feat)
        data_s = (data - self.feat_mean[0]) / self.feat_std[0]    # 정규화
        return torch.tensor(data_s[np.newaxis], dtype=torch.float32)  # (1, seq, feat)

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def predict(self, features_df):
        """MC Dropout으로 RUL 분포를 예측한다.

        Args:
            features_df: 최근 센서 특징 DataFrame (최소 seq_len 행 필요)

        Returns:
            rul_mean (float): 평균 RUL (시간)
            rul_low  (float): 5th percentile — 최악 시나리오
            rul_high (float): 95th percentile — 최선 시나리오
            rul_samples (list[float]): MC Dropout 샘플 N=100 — MILP P_fail 계산용

        데이터 부족 시 None 반환.
        """
        x = self._prepare_input(features_df)
        if x is None:
            return None

        try:
            # mc_dropout_predict -> (1, num_mc_samples)
            raw = mc_dropout_predict(self.model, x,
                                     num_samples=self.num_mc_samples)[0]  # (num_mc,)
            # 역정규화: 시간 단위(h)로 복원
            raw_h = raw * self.y_max
            rul_mean, rul_low, rul_high = get_rul_statistics(
                raw_h[np.newaxis]  # (1, num_mc)
            )
            return (
                float(rul_mean[0]),
                float(rul_low[0]),
                float(rul_high[0]),
                raw_h.tolist(),
            )
        except Exception as exc:
            print(f"[ModelWrapper] predict 오류: {exc}")
            return None

    def get_uncertainty_cv(self, features_df) -> float:
        """MC Dropout 예측의 변동계수(CV = std/mean)를 반환한다.

        SimulationEngine 이 이 CV 를 시뮬레이션 RUL 에 곱해
        현실적인 불확실성 밴드를 만든다.
        반환 범위: [0.05, 0.40]
        """
        x = self._prepare_input(features_df)
        if x is None:
            return 0.10

        try:
            raw = mc_dropout_predict(self.model, x,
                                     num_samples=self.num_mc_samples)[0]  # (num_mc,)
            raw_mean = float(np.mean(np.abs(raw))) + 1e-6
            raw_std = float(np.std(raw))
            return float(np.clip(raw_std / raw_mean, 0.05, 0.40))
        except Exception as exc:
            print(f"[ModelWrapper] get_uncertainty_cv 오류: {exc}")
            return 0.10
