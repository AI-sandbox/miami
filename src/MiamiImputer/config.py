"""Mode defaults for iterative repair and grouped configuration.

Scorer settings follow the research configuration.
Grouped values override explicit flat arguments during fit.
"""

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_mode(mode_defaults: dict) -> dict:
    flat: dict = {}
    for block in mode_defaults.values():
        if isinstance(block, dict):
            flat.update(block)
    return flat


# ---------------------------------------------------------------------------
# Public defaults — grouped for readability
# ---------------------------------------------------------------------------

DEFAULTS = {
    "mask": {
        # General behaviour of the correction loop
        "general": {
            "max_iter": 50,
            "random_state": 42,
            "verbose": False,
            "test_size": 0.2,
            "column_auc_threshold": 0.6,  # skip columns with AUC < 0.6
            "improvement_threshold": 0.1,  # min relative score gain to replace
            "compute_iteration_metrics": True,
        },
        # Candidate generation (codebooks)
        "generator": {
            "n_neighbors": 1,  # KNN for starter imputation
            "n_codebook": 10000,
            "n_exact_copy": 10000,
            "n_features_to_mutate": "all",
        },
        # Discriminator — MaskPredictor (XGBoost multi-label classifier)
        "discriminator": {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.01,
            "subsample": 1.0,
            "colsample_bylevel": 1.0,
        },
        # Knee-based early stopping (global per-column AUC)
        "knee": {
            "use_knee": True,
            "knee_savgol_window_length": 25,
            "knee_savgol_polyorder": 4,
            "knee_S": 3,
            "knee_min_points": 5,
        },
    },
    "regressor": {
        "general": {
            "max_iter": 50,
            "random_state": 42,
            "verbose": False,
            "test_size": 0.2,
            "r2_threshold": 0.3,  # R² gate (was column_auc_threshold)
            "improvement_threshold": 0.1,
            "compute_iteration_metrics": True,
        },
        "generator": {
            "n_neighbors": 1,
            "n_codebook": 10000,
            "n_exact_copy": 10000,
            "n_features_to_mutate": "all",
        },
        # Discriminator — IterativeXGBoostPredictor (regressor)
        "discriminator": {
            "n_estimators": 500,  # maximum rounds per fresh fit
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_lambda": 2.0,
            "reg_alpha": 0.5,
            "gamma": 0.2,
            "min_child_weight": 3,
            "window_size": 1,  # sliding window history
            "objective": "reg:pseudohubererror",
            "early_stopping_rounds": 30,
        },
        "knee": {
            "use_knee": True,
            "knee_savgol_window_length": 25,
            "knee_savgol_polyorder": 4,
            "knee_S": 3,
            "knee_min_points": 5,
        },
    },
}

# ---------------------------------------------------------------------------
# Flattened view used when resolving the public configuration
# ---------------------------------------------------------------------------
FLAT_DEFAULTS: dict[str, dict] = {mode: _flatten_mode(blocks) for mode, blocks in DEFAULTS.items()}

# Backwards-compat alias: some tuned configs used ``column_auc_threshold``
# for the regressor R² gate.  Both names resolve to the same value.
FLAT_DEFAULTS["regressor"]["column_auc_threshold"] = FLAT_DEFAULTS["regressor"]["r2_threshold"]

# ---------------------------------------------------------------------------
# Legacy flat constants — kept for discriminator.py imports.
# New code should use DEFAULTS / FLAT_DEFAULTS directly.
# ---------------------------------------------------------------------------
_MASK_N_ESTIMATORS = FLAT_DEFAULTS["mask"]["n_estimators"]
_MASK_MAX_DEPTH = FLAT_DEFAULTS["mask"]["max_depth"]
_MASK_LEARNING_RATE = FLAT_DEFAULTS["mask"]["learning_rate"]
_MASK_SUBSAMPLE = FLAT_DEFAULTS["mask"]["subsample"]
_REGRESSOR_N_ESTIMATORS = FLAT_DEFAULTS["regressor"]["n_estimators"]
_REGRESSOR_MAX_DEPTH = FLAT_DEFAULTS["regressor"]["max_depth"]
_REGRESSOR_LEARNING_RATE = FLAT_DEFAULTS["regressor"]["learning_rate"]
_REGRESSOR_REG_LAMBDA = FLAT_DEFAULTS["regressor"]["reg_lambda"]
_REGRESSOR_REG_ALPHA = FLAT_DEFAULTS["regressor"]["reg_alpha"]
_REGRESSOR_GAMMA = FLAT_DEFAULTS["regressor"]["gamma"]
_REGRESSOR_MIN_CHILD_WEIGHT = FLAT_DEFAULTS["regressor"]["min_child_weight"]
_REGRESSOR_WINDOW_SIZE = FLAT_DEFAULTS["regressor"]["window_size"]
_REGRESSOR_OBJECTIVE = FLAT_DEFAULTS["regressor"]["objective"]
_REGRESSOR_SUBSAMPLE = FLAT_DEFAULTS["regressor"]["subsample"]
_REGRESSOR_COLSAMPLE_BYTREE = FLAT_DEFAULTS["regressor"]["colsample_bytree"]
_REGRESSOR_EARLY_STOPPING_ROUNDS = FLAT_DEFAULTS["regressor"]["early_stopping_rounds"]
