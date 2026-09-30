"""
Discriminator/Predictor models for MIAMI Imputer.

Two predictors:
- MaskPredictor: XGBoost multi-label classifier predicting P(imputed) per feature
- IterativeXGBoostPredictor: XGBoost regressor fitted from scratch on a sliding data window
"""

import json

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from .base import BaseDiscriminator
from .config import (
    _MASK_LEARNING_RATE,
    _MASK_MAX_DEPTH,
    _MASK_N_ESTIMATORS,
    _MASK_SUBSAMPLE,
    _REGRESSOR_COLSAMPLE_BYTREE,
    _REGRESSOR_EARLY_STOPPING_ROUNDS,
    _REGRESSOR_GAMMA,
    _REGRESSOR_LEARNING_RATE,
    _REGRESSOR_MAX_DEPTH,
    _REGRESSOR_MIN_CHILD_WEIGHT,
    _REGRESSOR_N_ESTIMATORS,
    _REGRESSOR_OBJECTIVE,
    _REGRESSOR_REG_ALPHA,
    _REGRESSOR_REG_LAMBDA,
    _REGRESSOR_SUBSAMPLE,
    _REGRESSOR_WINDOW_SIZE,
    RANDOM_STATE,
)

MASK_N_ESTIMATORS, MASK_MAX_DEPTH = _MASK_N_ESTIMATORS, _MASK_MAX_DEPTH
MASK_LEARNING_RATE, MASK_SUBSAMPLE = _MASK_LEARNING_RATE, _MASK_SUBSAMPLE
REGRESSOR_N_ESTIMATORS = _REGRESSOR_N_ESTIMATORS
REGRESSOR_MAX_DEPTH, REGRESSOR_LEARNING_RATE = _REGRESSOR_MAX_DEPTH, _REGRESSOR_LEARNING_RATE
REGRESSOR_REG_LAMBDA, REGRESSOR_REG_ALPHA = _REGRESSOR_REG_LAMBDA, _REGRESSOR_REG_ALPHA
REGRESSOR_GAMMA, REGRESSOR_MIN_CHILD_WEIGHT = _REGRESSOR_GAMMA, _REGRESSOR_MIN_CHILD_WEIGHT
REGRESSOR_WINDOW_SIZE, REGRESSOR_OBJECTIVE = _REGRESSOR_WINDOW_SIZE, _REGRESSOR_OBJECTIVE
REGRESSOR_SUBSAMPLE, REGRESSOR_COLSAMPLE_BYTREE = _REGRESSOR_SUBSAMPLE, _REGRESSOR_COLSAMPLE_BYTREE
REGRESSOR_EARLY_STOPPING_ROUNDS = _REGRESSOR_EARLY_STOPPING_ROUNDS


def _check_requested_device(model, requested_device):
    """Reject XGBoost's CPU fallback when the caller explicitly requests CUDA."""
    actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
    if requested_device.startswith("cuda") and not actual.startswith("cuda"):
        raise RuntimeError(
            f"Requested device={requested_device!r}, but XGBoost used {actual!r}. "
            "Install a CUDA-enabled XGBoost build and check the NVIDIA driver/GPU, "
            "or explicitly choose device='cpu'."
        )
    return actual


class MaskPredictor(BaseDiscriminator):
    """
    Multi-label classifier predicting P(imputed) per feature via XGBoost.

    Trains on the full binary mask matrix: for each feature column,
    predicts the probability that a value was imputed (mask == 1).

    Parameters
    ----------
    n_estimators : int, default=500
        Number of boosting rounds.
    max_depth : int, default=7
        Maximum tree depth.
    learning_rate : float, default=0.01
        Boosting learning rate.
    subsample : float, default=1.0
        Subsample ratio of training instances.
    colsample_bylevel : float, default=1.0
        Subsample ratio of columns for each split.
    random_state : int, default=42
        Random seed.
    early_stopping_rounds : int, default=20
        Early stopping rounds on validation set.
    cols_with_missing : list, optional
        Column indices with missing values. If None, auto-detected.
    """

    is_classifier = True

    def __init__(
        self,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        learning_rate: float | None = None,
        subsample: float | None = None,
        colsample_bylevel: float = 1.0,
        random_state: int = RANDOM_STATE,
        early_stopping_rounds: int = 20,
        cols_with_missing: list | None = None,
        device: str = "cpu",
        n_jobs: int = 1,
        validation_fraction: float = 0.15,
    ):
        self.n_estimators = n_estimators if n_estimators is not None else MASK_N_ESTIMATORS
        self.max_depth = max_depth if max_depth is not None else MASK_MAX_DEPTH
        self.learning_rate = learning_rate if learning_rate is not None else MASK_LEARNING_RATE
        self.subsample = subsample if subsample is not None else MASK_SUBSAMPLE
        self.colsample_bylevel = colsample_bylevel
        self.random_state = random_state
        self.device = device
        self.n_jobs = n_jobs
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.cols_with_missing = cols_with_missing
        self._model = None
        self._n_features = None

    def fit(self, X: np.ndarray, mask: np.ndarray) -> "MaskPredictor":
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        self._n_features = X.shape[1]

        if self.cols_with_missing is None:
            self.cols_with_missing = [i for i in range(mask.shape[1]) if mask[:, i].any()]

        if len(self.cols_with_missing) == 0:
            raise ValueError("No columns with missing values found.")

        # XGBoost cannot fit an all-one target. Retain constant outputs as
        # constants and fit only mask coordinates containing both classes.
        self._output_columns = list(self.cols_with_missing)
        self._constant_probs = np.asarray(mask, dtype=float).mean(axis=0)
        self._trained_columns = [c for c in self._output_columns if 0 < self._constant_probs[c] < 1]
        if not self._trained_columns:
            self._model = None
            return self
        y = np.asarray(mask[:, self._trained_columns], dtype=np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=min(self.validation_fraction, 1 - 1 / len(X)),
            random_state=self.random_state,
        )

        if np.unique(y_tr).size < 2:
            X_tr, y_tr = X, y
        if y_tr.shape[1] == 1:
            y_tr, y_val = y_tr.ravel(), y_val.ravel()

        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bylevel=self.colsample_bylevel,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            multi_strategy="one_output_per_tree",
            device=self.device,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds,
            verbosity=0,
        )
        self._model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        _check_requested_device(self._model, self.device)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_constant_probs"):
            raise ValueError("Predictor not fitted. Call fit() first.")
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
        full = np.tile(self._constant_probs, (len(X), 1))
        if self._model is not None:
            probs = self._model.predict_proba(X)
            if len(self._trained_columns) == 1:
                probs = probs[:, 1:2]
            full[:, self._trained_columns] = probs
        return full

    def get_metrics(self, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        aucs = self.get_metrics_per_column(X_val, y_val)["column_aucs"]
        valid = [a for a in aucs if a >= 0]
        return {"mean_auc": float(np.mean(valid)) if valid else 0.0}

    def get_metrics_per_column(self, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        probs = self.predict(X_val)[:, self.cols_with_missing]
        return {
            "column_aucs": [
                float(roc_auc_score(y_val[:, j], probs[:, j]))
                if np.unique(y_val[:, j]).size == 2
                else -1
                for j in range(y_val.shape[1])
            ]
        }


class IterativeXGBoostPredictor(BaseDiscriminator):
    """
    XGBoost regressor fitted from scratch on a sliding data window.

    Trains on a sliding window of the last N iterations of data.
    Every iteration trains a new model; trees are never retained between fits.
    The default data window contains only the current iteration.

    Parameters
    ----------
    n_estimators : int, default=500
        Maximum number of boosting rounds for each fresh fit.
    max_depth : int, default=6
        Maximum tree depth.
    learning_rate : float, default=0.1
        Boosting learning rate.
    reg_lambda : float, default=2.0
        L2 regularization on weights.
    reg_alpha : float, default=0.5
        L1 regularization on weights.
    gamma : float, default=0.2
        Minimum loss reduction for partition.
    min_child_weight : int, default=3
        Minimum sum of instance weight in child.
    colsample_bytree : float, default=0.7
        Subsample ratio of columns per tree.
    subsample : float, default=0.8
        Subsample ratio of training instances.
    early_stopping_rounds : int, default=30
        Early stopping rounds.
    random_state : int, default=42
        Random seed.
    window_size : int, default=1
        Number of past iterations to include in training window.
    cols_with_missing : list, optional
        Column indices with missing values.
    objective : str, default='reg:pseudohubererror'
        XGBoost objective function.
    """

    is_classifier = False

    def __init__(
        self,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        learning_rate: float | None = None,
        reg_lambda: float | None = None,
        reg_alpha: float | None = None,
        gamma: float | None = None,
        min_child_weight: int | None = None,
        colsample_bytree: float | None = None,
        subsample: float | None = None,
        early_stopping_rounds: int | None = None,
        random_state: int = RANDOM_STATE,
        window_size: int | None = None,
        cols_with_missing: list | None = None,
        objective: str | None = None,
        device: str = "cpu",
        n_jobs: int = 1,
        validation_fraction: float = 0.15,
    ):

        self.n_estimators = n_estimators if n_estimators is not None else REGRESSOR_N_ESTIMATORS
        self.max_depth = max_depth if max_depth is not None else REGRESSOR_MAX_DEPTH
        self.learning_rate = learning_rate if learning_rate is not None else REGRESSOR_LEARNING_RATE
        self.reg_lambda = reg_lambda if reg_lambda is not None else REGRESSOR_REG_LAMBDA
        self.reg_alpha = reg_alpha if reg_alpha is not None else REGRESSOR_REG_ALPHA
        self.gamma = gamma if gamma is not None else REGRESSOR_GAMMA
        self.min_child_weight = (
            min_child_weight if min_child_weight is not None else REGRESSOR_MIN_CHILD_WEIGHT
        )
        self.colsample_bytree = (
            colsample_bytree if colsample_bytree is not None else REGRESSOR_COLSAMPLE_BYTREE
        )
        self.subsample = subsample if subsample is not None else REGRESSOR_SUBSAMPLE
        self.early_stopping_rounds = (
            early_stopping_rounds
            if early_stopping_rounds is not None
            else REGRESSOR_EARLY_STOPPING_ROUNDS
        )
        self.random_state = random_state
        self.device = device
        self.n_jobs = n_jobs
        self.validation_fraction = validation_fraction
        self.window_size = window_size if window_size is not None else REGRESSOR_WINDOW_SIZE
        self.cols_with_missing = cols_with_missing
        self.objective = objective if objective is not None else REGRESSOR_OBJECTIVE

        self._model = None
        self._X_val = None
        self._y_val = None
        self._X_train_history = []
        self._y_train_history = []
        self._train_r2 = 0.0
        self._val_r2 = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IterativeXGBoostPredictor":
        import gc

        X = np.nan_to_num(X, nan=0.0).astype(np.float32)

        self._X_train_history.append(X.copy())
        self._y_train_history.append(y.copy())

        if len(self._X_train_history) > self.window_size:
            self._X_train_history.pop(0)
            self._y_train_history.pop(0)

        X_combined = np.vstack(self._X_train_history)
        y_combined = np.concatenate(self._y_train_history)

        self._X_tr, self._X_val, self._y_tr, self._y_val = train_test_split(
            X_combined,
            y_combined,
            test_size=min(self.validation_fraction, 1 - 1 / len(X_combined)),
            random_state=self.random_state,
        )

        # Rebuild the scorer at every iteration. The data window controls which
        # observations are used; it never carries trees over from previous fits.
        if self._model is not None:
            self._model = None
            gc.collect()

        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            objective=self.objective,
            tree_method="hist",
            device=self.device,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=0,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self._model.fit(
            self._X_tr,
            self._y_tr,
            eval_set=[(self._X_val, self._y_val)],
            verbose=False,
        )

        _check_requested_device(self._model, self.device)

        train_pred = self._model.predict(self._X_tr)
        self._train_r2 = self._compute_r2(self._y_tr, train_pred)
        val_pred = self._model.predict(self._X_val)
        self._val_r2 = self._compute_r2(self._y_val, val_pred)

        return self

    def _compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("Predictor not fitted. Call fit() first.")
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
        return self._model.predict(X)

    def get_metrics(self, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        if self._model is None:
            return {}
        X_val = np.nan_to_num(np.asarray(X_val, dtype=np.float32), nan=0.0)
        pred = self._model.predict(X_val)
        r2 = self._compute_r2(y_val, pred)
        return {"r2": r2, "train_r2": self._train_r2, "val_r2": self._val_r2}


def get_predictor(mode: str = "mask", **kwargs) -> BaseDiscriminator:
    """
    Factory to create predictor based on mode.

    Parameters
    ----------
    mode : str
        ``'mask'`` or ``'regressor'``.
    **kwargs : dict
        Parameters forwarded to predictor constructor.

    Returns
    -------
    predictor : BaseDiscriminator
    """
    if mode == "mask":
        return MaskPredictor(**kwargs)
    elif mode == "regressor":
        return IterativeXGBoostPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Valid modes: 'mask', 'regressor'")
