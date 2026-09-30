"""
Abstract base classes for MIAMI components.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseDiscriminator(ABC):
    """Base class for discriminating/scoring candidates."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseDiscriminator":
        """Fit the discriminator."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict scores for candidates."""
        pass

    def get_metrics(self, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
        """Get evaluation metrics on validation set."""
        return {}


class BaseScorer(ABC):
    """Base class for interpreting discriminator output."""

    @abstractmethod
    def score(self, predictions: np.ndarray, **kwargs) -> np.ndarray:
        """
        Transform discriminator output to scores.

        Returns
        -------
        scores : ndarray
            Scores to use for selection (lower is better by default).
        """
        pass

    @property
    @abstractmethod
    def lower_is_better(self) -> bool:
        """Whether lower scores are better."""
        pass


class BaseGenerator(ABC):
    """Base class for generating correction candidates."""

    @abstractmethod
    def generate(
        self, X: np.ndarray, mask: np.ndarray, codebooks: dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Generate candidate variations.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Current data.
        mask : ndarray of shape (n_samples, n_features)
            Boolean array where True indicates imputed values.
        codebooks : dict
            Dictionary mapping feature index to array of valid values.

        Returns
        -------
        candidates : ndarray of shape (n_candidates, n_features)
            Generated candidates.
        """
        pass

    @property
    @abstractmethod
    def n_candidates_per_sample(self) -> int:
        """Return the number of candidates per sample."""
        pass
