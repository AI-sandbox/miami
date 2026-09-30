"""
Codebook: unique original (non-missing) values per feature.
"""

import numpy as np


def build_codebooks(X, mask):
    """
    Build codebook from the original (non-missing) values.

    For each feature j, collects all values X[i, j] where
    mask[i, j] == False (i.e., the value was originally observed,
    not missing/imputed).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data (can contain NaNs for missing values).
    mask : array-like of shape (n_samples, n_features)
        Boolean mask. True = value was missing, False = originally observed.

    Returns
    -------
    codebooks : dict
        Dictionary mapping feature index to sorted unique observed values.
    """
    n_features = X.shape[1]
    codebooks = {}
    for j in range(n_features):
        observed = X[~mask[:, j], j]
        codebooks[j] = np.unique(observed)
    return codebooks
