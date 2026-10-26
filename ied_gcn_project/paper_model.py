"""Model 1 - the paper's IED-GCN, reproduced as published (Li et al., IEEE TGRS 2025).

Nothing here goes beyond the paper: the 16 factors are fed in as they are, one network is trained,
and the five susceptibility classes come from natural breaks (Jenks) on the predicted probability.
"""
from __future__ import annotations

import numpy as np

from common import HyperParams, jenks_breaks, train_and_predict

NAME = "paper"
TITLE = "Paper model (IED-GCN as published)"


def model_features(features: np.ndarray) -> np.ndarray:
    return features


def fit_predict(g, train_idx: np.ndarray, hp: HyperParams, seed: int) -> tuple[np.ndarray, dict]:
    return train_and_predict(g, train_idx, hp, seed, tag="paper"), {}


def classify(prob: np.ndarray, valid_sp: np.ndarray) -> np.ndarray:
    """Five classes (0 = very low ... 4 = very high) by natural breaks on the probability,
    with the breaks fitted on superpixels inside the study area only."""
    return np.digitize(prob, jenks_breaks(prob[valid_sp]))
