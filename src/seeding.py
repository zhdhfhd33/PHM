"""재현성(reproducibility)을 위한 전역 시드 고정 유틸.

mock 모드를 제거한 뒤 시뮬레이션의 무작위성은 사실상 LSTM의 MC Dropout(torch)에서만
나온다. 따라서 numpy 시드만 고정하던 기존 방식으로는 실행마다 결과가 미세하게 흔들렸다.
이 모듈은 random / numpy / torch 시드를 한 번에 고정해 동일 입력 → 동일 결과를 보장한다.
"""
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42):
    """random·numpy·torch 전역 시드를 고정한다.

    Args:
        seed: 고정할 시드 값.

    Note:
        시뮬레이션/백테스트를 시작하기 직전에 호출해야 한다. 호출 이후의 torch
        무작위 연산(MC Dropout 마스크)이 결정론적 순서로 재현된다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
