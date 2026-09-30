"""
Scoring functions for MIAMI Imputer.

Two scorers:
- MaskScorer: sum classifier probabilities on imputed features (lower = better)
- RegressorOnlyScorer: pass-through of regressor predictions (lower = better)
"""

import numpy as np

from .base import BaseScorer


class MaskScorer(BaseScorer):
    """
    Scorer for MaskPredictor.

    Sums the predicted P(imputed) probabilities only on the imputed
    features of each sample. The candidate with the lowest sum is the
    one least likely to be fake.

    Parameters
    ----------
    method : str, default='sum'
        Aggregation method: 'sum' or 'max'.
    """

    def __init__(self, method: str = "sum"):
        if method not in ("sum", "max"):
            raise ValueError("method must be 'sum' or 'max'")
        self.method = method

    def score(
        self,
        predictions: np.ndarray,
        mask_batch: np.ndarray = None,
        metadata: list = None,
        **kwargs,
    ) -> np.ndarray:
        if mask_batch is None:
            raise ValueError("mask_batch required for MaskScorer")
        if metadata is None:
            raise ValueError("metadata required for MaskScorer")

        scores = np.zeros(predictions.shape[0])
        cursor = 0

        for meta_idx, num_candidates in metadata:
            sample_mask = mask_batch[meta_idx].astype(bool)

            block = predictions[cursor : cursor + num_candidates, sample_mask]
            if block.shape[1]:
                aggregate = np.sum if self.method == "sum" else np.max
                scores[cursor : cursor + num_candidates] = aggregate(block, axis=1)

            cursor += num_candidates

        return scores

    @property
    def lower_is_better(self) -> bool:
        return True


class RegressorOnlyScorer(BaseScorer):
    """
    Scorer for IterativeXGBoostPredictor.

    Returns the regressor's predicted miss_pct as the score directly.
    Lower predicted missing percentage = more real-looking = better.
    """

    def score(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        return predictions.flatten()

    @property
    def lower_is_better(self) -> bool:
        return True


def get_scorer(mode: str = "mask", **kwargs) -> BaseScorer:
    """
    Factory to create scorer based on mode.

    Parameters
    ----------
    mode : str
        ``'mask'`` or ``'regressor'``.
    **kwargs : dict
        Parameters forwarded to scorer constructor.

    Returns
    -------
    scorer : BaseScorer
    """
    if mode == "mask":
        return MaskScorer(**kwargs)
    elif mode == "regressor":
        return RegressorOnlyScorer(**kwargs)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Valid modes: 'mask', 'regressor'")
