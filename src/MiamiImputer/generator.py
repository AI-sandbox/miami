"""
Candidate generators for MIAMI Imputer.

Two strategies combined by MixedGenerator:
- Codebook: randomly sample values from each feature's codebook of valid values
- ExactCopy: copy eligible feature values from observed donor coordinates
"""

import numpy as np
from sklearn.utils import check_random_state

from .base import BaseGenerator


class CodebookGenerator(BaseGenerator):
    """
    Generate candidates by randomly sampling from feature codebooks.

    For each candidate, a random subset of the sample's imputed features
    is replaced with values drawn from the feature's codebook.

    Parameters
    ----------
    n_candidates : int, default=10000
        Number of candidates per sample.
    n_features_to_mutate : int or 'all', optional
        Number of features to consider for mutation per candidate.
        'all' = all imputed features are eligible (each has ~50% chance).
    """

    def __init__(
        self,
        n_candidates: int = 10000,
        n_features_to_mutate: int | str | None = None,
        random_state=None,
    ):
        self.n_candidates = n_candidates
        self.n_features_to_mutate = n_features_to_mutate
        self.rng = check_random_state(random_state)

    @property
    def n_candidates_per_sample(self) -> int:
        return self.n_candidates

    def generate(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        codebooks: dict[int, np.ndarray],
        columns_to_skip: set | None = None,
    ) -> np.ndarray:
        n_samples = X.shape[0]
        n_candidates = self.n_candidates
        candidates = np.repeat(X, n_candidates, axis=0)

        for sample_idx in range(n_samples):
            imp_feats = np.where(mask[sample_idx])[0]
            if len(imp_feats) == 0:
                continue

            base_idx = sample_idx * n_candidates

            if self.n_features_to_mutate == "all":
                pool_feats = np.array(
                    [f for f in imp_feats if columns_to_skip is None or f not in columns_to_skip]
                )
                if len(pool_feats) == 0:
                    continue
                k_pool = len(pool_feats)
            else:
                eligible = [
                    f for f in imp_feats if columns_to_skip is None or f not in columns_to_skip
                ]
                k_pool = min(
                    3 if self.n_features_to_mutate is None else self.n_features_to_mutate,
                    len(eligible),
                )
                pool_feats = self.rng.choice(eligible, k_pool, replace=False)

            mutate_mask = self.rng.randint(0, 2, size=(n_candidates, k_pool)).astype(bool)

            for i, feat_idx in enumerate(pool_feats):
                cb = codebooks.get(feat_idx, np.array([]))
                if len(cb) == 0:
                    continue
                mask_i = mutate_mask[:, i]
                if mask_i.any():
                    cb_indices = self.rng.randint(0, len(cb), size=mask_i.sum())
                    candidates[base_idx + np.where(mask_i)[0], feat_idx] = cb[cb_indices]

        return candidates


class ExactCopyGenerator(BaseGenerator):
    """
    Generate candidates by copying real values from donor samples.

    For each sample, finds donor samples that have real (non-missing)
    values in all the same feature positions, and copies their values.
    Copied coordinates come from a common observed donor; the completed row
    need not itself be an observed joint sample.

    Parameters
    ----------
    n_candidates : int, default=10000
        Number of candidates per sample.
    n_features_to_mutate : int or 'all', optional
        Number of features to copy per candidate.
        'all' = all imputed features are copied from the donor.
    """

    def __init__(
        self,
        n_candidates: int = 10000,
        n_features_to_mutate: int | str | None = None,
        random_state=None,
    ):
        self.n_candidates = n_candidates
        self.n_features_to_mutate = n_features_to_mutate
        self.rng = check_random_state(random_state)
        self.X_full = None
        self.feature_available = None

    @property
    def n_candidates_per_sample(self) -> int:
        return self.n_candidates

    def generate(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        codebooks: dict[int, np.ndarray] | None = None,
        columns_to_skip: set | None = None,
    ) -> np.ndarray:
        n_samples, n_features = X.shape
        n_candidates = self.n_candidates
        candidates = np.empty((n_samples * n_candidates, n_features), dtype=X.dtype)
        candidates[:] = np.nan

        X_source = self.X_full if self.X_full is not None else X
        has_real = ~np.isnan(X_source)

        for i in range(n_samples):
            base_idx = i * n_candidates
            imp_feats = np.array(
                [
                    f
                    for f in np.where(mask[i])[0]
                    if columns_to_skip is None or f not in columns_to_skip
                ],
                dtype=int,
            )

            if len(imp_feats) == 0:
                candidates[base_idx : base_idx + n_candidates] = X[i]
                continue

            candidates[base_idx : base_idx + n_candidates] = X[i]

            if self.feature_available is not None:
                candidates_idx = set(self.feature_available[imp_feats[0]])
                for feat in imp_feats[1:]:
                    candidates_idx &= set(self.feature_available[feat])
                candidates_idx = sorted(candidates_idx)
            else:
                candidates_mask = has_real[:, imp_feats].all(axis=1)
                candidates_idx = np.where(candidates_mask)[0].tolist()

            if len(candidates_idx) == 0:
                continue

            n_to_sample = min(n_candidates, len(candidates_idx))
            selected_idx = self.rng.choice(candidates_idx, n_to_sample, replace=False)

            if self.n_features_to_mutate == "all":
                for j, src_idx in enumerate(selected_idx):
                    candidates[base_idx + j, imp_feats] = X_source[src_idx, imp_feats]
            elif self.n_features_to_mutate is None:
                for j, src_idx in enumerate(selected_idx):
                    n_to_copy = self.rng.randint(1, min(4, len(imp_feats) + 1))
                    feats_to_copy = self.rng.choice(imp_feats, n_to_copy, replace=False)
                    candidates[base_idx + j, feats_to_copy] = X_source[src_idx, feats_to_copy]
            else:
                for j, src_idx in enumerate(selected_idx):
                    n_to_copy = min(self.n_features_to_mutate, len(imp_feats))
                    feats_to_copy = self.rng.choice(imp_feats, n_to_copy, replace=False)
                    candidates[base_idx + j, feats_to_copy] = X_source[src_idx, feats_to_copy]

        return candidates


class MixedGenerator(BaseGenerator):
    """
    Mixed candidate generator combining codebook and exact copy strategies.

    Produces n_codebook + n_exact_copy candidates per sample, interleaved
    for structured batch processing.

    Parameters
    ----------
    n_codebook : int, default=10000
        Number of codebook candidates per sample.
    n_exact_copy : int, default=10000
        Number of exact copy candidates per sample.
    n_features_to_mutate : int or 'all', optional
        Number of features to replace per candidate.
    """

    def __init__(
        self,
        n_codebook: int = 10000,
        n_exact_copy: int = 10000,
        n_features_to_mutate: int | str | None = None,
        random_state=None,
    ):
        self.n_codebook = n_codebook
        self.n_exact_copy = n_exact_copy
        self.n_features_to_mutate = n_features_to_mutate
        self.rng = check_random_state(random_state)
        self._cb = CodebookGenerator(
            n_candidates=n_codebook,
            n_features_to_mutate=n_features_to_mutate,
            random_state=self.rng,
        )
        self._ec = ExactCopyGenerator(
            n_candidates=n_exact_copy,
            n_features_to_mutate=n_features_to_mutate,
            random_state=self.rng,
        )
        self.codebooks = None
        self.X_full = None
        self.feature_available = None

    @property
    def n_candidates_per_sample(self) -> int:
        return self.n_codebook + self.n_exact_copy

    def generate(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        codebooks: dict[int, np.ndarray],
        columns_to_skip: set | None = None,
    ) -> np.ndarray:
        if self.codebooks is not None:
            codebooks = self.codebooks
        if self.X_full is not None:
            self._ec.X_full = self.X_full
        if self.feature_available is not None:
            self._ec.feature_available = self.feature_available

        cb_candidates = self._cb.generate(X, mask, codebooks, columns_to_skip)
        ec_candidates = self._ec.generate(X, mask, codebooks, columns_to_skip)

        n_samples = X.shape[0]
        total_per_sample = self.n_candidates_per_sample
        result = np.empty((n_samples * total_per_sample, X.shape[1]), dtype=X.dtype)

        for i in range(n_samples):
            base = i * total_per_sample
            result[base : base + self.n_codebook] = cb_candidates[
                i * self.n_codebook : (i + 1) * self.n_codebook
            ]
            result[base + self.n_codebook : base + total_per_sample] = ec_candidates[
                i * self.n_exact_copy : (i + 1) * self.n_exact_copy
            ]

        return result
