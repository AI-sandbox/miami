"""Smoke tests — both modes run without error on a tiny dataset."""

import numpy as np

try:
    import pytest  # noqa: F401
except ImportError:
    pytest = None  # type: ignore[assignment]


def _make_toy(n=80, p=6, miss_rate=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    mask = rng.random((n, p)) < miss_rate
    # ensure every column has at least one missing and one observed
    for j in range(p):
        mask[0, j] = True
        mask[1, j] = False
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing


def test_import():
    from MiamiImputer.config import DEFAULTS, FLAT_DEFAULTS

    assert "mask" in DEFAULTS
    assert "regressor" in DEFAULTS
    assert "mask" in FLAT_DEFAULTS
    assert FLAT_DEFAULTS["mask"]["knee_savgol_window_length"] == 25
    assert FLAT_DEFAULTS["regressor"]["r2_threshold"] == 0.3


def test_mask_smoke():
    from MiamiImputer import MIAMIImputer

    X = _make_toy()
    imp = MIAMIImputer(
        mode="mask", verbose=False, max_iter=2, n_estimators=5, n_codebook=8, n_exact_copy=8
    )
    X_out = imp.fit_transform(X)
    assert X_out.shape == X.shape
    assert not np.isnan(X_out).any()
    assert len(imp.get_iteration_metrics()) > 0


def test_regressor_smoke():
    from MiamiImputer import MIAMIImputer

    X = _make_toy()
    imp = MIAMIImputer(
        mode="regressor", verbose=False, max_iter=2, n_estimators=5, n_codebook=8, n_exact_copy=8
    )
    X_out = imp.fit_transform(X)
    assert X_out.shape == X.shape
    assert not np.isnan(X_out).any()


def test_grouped_overrides():
    from MiamiImputer import MIAMIImputer

    imp = MIAMIImputer(
        mode="mask",
        verbose=False,
        discriminator={"max_depth": 3, "n_estimators": 10},
        knee={"knee_S": 5},
    )
    assert imp.discriminator["max_depth"] == 3
    assert imp.discriminator["n_estimators"] == 10
    assert imp.knee["knee_S"] == 5

    imp2 = MIAMIImputer(mode="regressor", verbose=False, general={"r2_threshold": 0.4})
    imp2.set_params(max_iter=0).fit(_make_toy())
    assert imp2.effective_params_["r2_threshold"] == 0.4
    assert imp2.effective_params_["column_auc_threshold"] == 0.4


def test_self_contained_no_sys_path():
    import sys

    # Importing the package must NOT inject phase1_dynamics into sys.path
    import MiamiImputer  # noqa: F401

    assert not any("phase1_dynamics" in p for p in sys.path)
    assert not any("miami_copy" in p for p in sys.path)
