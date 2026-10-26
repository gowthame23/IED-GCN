"""Model 2 - our model: the same IED-GCN with two changes on top of the paper.

1. Categorical factors (aspect, land use, lithology) are one-hot encoded instead of being fed as
   numbers, so e.g. "lithology 3" is not treated as three times "lithology 1".
2. Map classing: natural breaks are taken on the logit of the probability. On raw probabilities
   almost every superpixel sits near 0 or 1, which leaves the Lo/Mo/Hi classes nearly empty.

(An ensemble of 3 networks was also tried and gave no measurable gain, so it was dropped.)

Superpixels, graphs, split, hyper-parameters and training loop are identical to the paper model.
"""
from __future__ import annotations

import numpy as np

from common import HyperParams, jenks_breaks, logit, one_hot_categoricals, train_and_predict

NAME = "ours"
TITLE = "Our model (paper + one-hot factors + logit classing)"


def model_features(features: np.ndarray) -> np.ndarray:
    return one_hot_categoricals(features)


def fit_predict(g, train_idx: np.ndarray, hp: HyperParams, seed: int) -> tuple[np.ndarray, dict]:
    return train_and_predict(g, train_idx, hp, seed, tag="ours"), {}


def classify(prob: np.ndarray, valid_sp: np.ndarray) -> np.ndarray:
    score = logit(prob)
    return np.digitize(score, jenks_breaks(score[valid_sp]))
