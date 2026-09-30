"""Public scikit-learn interface for Miami distributional imputation."""

import hashlib
import re
from copy import deepcopy
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from ._engine import _MIAMIEngine
from ._table import _TableCodec
from .config import DEFAULTS, FLAT_DEFAULTS
from .generator import MixedGenerator


class MIAMIImputer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Repair numerical and categorical missing values using a learned missingness scorer.

    ``fit_transform`` runs the iterative repair on the supplied training table.
    ``transform`` initializes new rows with the fitted kNN imputer and performs
    one candidate-selection pass using the final frozen scorer and reliability
    gates. New rows never retrain the scorer or become donors. Transforming the
    original training table returns its learned iterative completion.

    Parameters
    ----------
    mode : {'mask', 'regressor'}, default='mask'
        Missingness-mask classifier or missing-fraction regressor.
    max_iter : int, default=50
        Maximum outer training iterations; zero gives kNN initialization only.
    n_neighbors : int, default=1
        Neighbors for the initial KNNImputer.
    n_codebook, n_exact_copy : int, default=10000
        Candidate budgets per row for independent values and observed donors.
    n_features_to_mutate : 'all' or positive int, default='all'
        Eligible missing coordinates for a codebook proposal. Each is mutated
        with probability one half. Donor proposals copy the eligible coordinates.
    test_size : float, default=0.2
        Training-row fraction reserved for correction at each iteration.
    improvement_threshold : float, default=0.1
        Required strict relative decrease in candidate score.
    column_auc_threshold : float or None, default=None
        Mask AUC gate (default 0.6). Legacy alias for the regressor R2 gate.
    r2_threshold : float, default=0.3
        Regressor correction gate.
    n_estimators : int, default=500
        Maximum boosting rounds per fresh scorer fit.
    max_depth, learning_rate : int or float or None, default=None
        Mode defaults: depth 7 / rate .01 for mask, depth 6 / rate .1 for regressor.
    use_knee : bool, default=True
        Stop converged columns (mask) or the outer loop (regressor) at a knee.
    random_state : int, RandomState or None, default=42
        Controls splits, scorer fitting, and proposal draws without changing
        NumPy's global RNG when an integer seed is supplied.
    device : str, default='cpu'
        XGBoost device, e.g. 'cpu' or 'cuda'. GPU use is explicitly requested.
    n_jobs : int, default=1
        Number of XGBoost threads.
    batch_size : int, default=4
        Maximum training rows whose candidates are materialized together.
    validation_fraction : float, default=0.15
        Internal scorer validation fraction, within the scoring split.
    early_stopping_rounds : int or None, default=None
        XGBoost patience; mode defaults are 20 (mask) and 30 (regressor).
    verbose : bool, default=False
        Print iteration progress.
    compute_iteration_metrics : bool, default=True
        Collect descriptive training diagnostics in ``iteration_metrics_``.
    categorical_features : list of str or int or None, default=None
        Additional categorical columns, identified by DataFrame names or column
        indices. Strings, booleans and pandas categorical columns are detected
        automatically; use this option for categories stored as numeric codes.
    general, generator, discriminator, knee : dict or None, default=None
        Optional grouped overrides of keys in ``DEFAULTS[mode]``. Groups take
        precedence over flat arguments. Unknown options are rejected at fit.
        Values are copied before use and are never changed in place.

    Attributes
    ----------
    n_features_in_ : int
        Number of fitted features; feature_names_in_ is set for named tables.
    imputed_X_ : ndarray or DataFrame
        Iteratively completed training data, in original feature units.
    n_iter_ : int
        Number of completed outer iterations.
    iteration_metrics_ : list of dict
        Diagnostics for completed iterations.
    effective_params_ : dict
        Validated mode defaults and user overrides used for the fitted engine.
    empty_features_ : ndarray of bool
        Columns without training observations; missing values receive zero.

    Notes
    -----
    Inputs are NumPy arrays, pandas DataFrames or rectangular lists. Missing
    entries may be np.nan, None or pd.NA. Categories are encoded internally and
    restored on output. DataFrames retain their index and column labels; other
    inputs return arrays. Empty numeric columns receive zero; categorical columns
    require at least one observed value. Observed entries are preserved.
    """

    def __init__(
        self,
        mode="mask",
        *,
        max_iter=50,
        n_neighbors=1,
        n_codebook=10000,
        n_exact_copy=10000,
        n_features_to_mutate="all",
        test_size=0.2,
        improvement_threshold=0.1,
        column_auc_threshold=None,
        r2_threshold=0.3,
        n_estimators=500,
        max_depth=None,
        learning_rate=None,
        use_knee=True,
        random_state=42,
        device="cpu",
        n_jobs=1,
        batch_size=4,
        validation_fraction=0.15,
        early_stopping_rounds=None,
        verbose=False,
        compute_iteration_metrics=True,
        categorical_features=None,
        general=None,
        generator=None,
        discriminator=None,
        knee=None,
    ):
        self.mode = mode
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.n_codebook = n_codebook
        self.n_exact_copy = n_exact_copy
        self.n_features_to_mutate = n_features_to_mutate
        self.test_size = test_size
        self.improvement_threshold = improvement_threshold
        self.column_auc_threshold = column_auc_threshold
        self.r2_threshold = r2_threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_knee = use_knee
        self.random_state = random_state
        self.device = device
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.compute_iteration_metrics = compute_iteration_metrics
        self.categorical_features = categorical_features
        self.general = general
        self.generator = generator
        self.discriminator = discriminator
        self.knee = knee

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        return tags

    def _resolve_params(self):
        if self.mode not in DEFAULTS:
            raise ValueError("mode must be 'mask' or 'regressor'")
        params = deepcopy(FLAT_DEFAULTS[self.mode])
        groups = {"general", "generator", "discriminator", "knee"}
        for name, value in self.get_params(deep=False).items():
            if (
                name not in groups | {"mode", "random_state", "categorical_features"}
                and value is not None
            ):
                params[name] = value
        for group in sorted(groups):
            values = getattr(self, group)
            if values is None:
                continue
            if not isinstance(values, dict):
                raise TypeError(f"{group} must be a dict or None")
            unknown = values.keys() - DEFAULTS[self.mode][group].keys()
            if unknown:
                raise ValueError(f"Unknown {group} parameters: {sorted(unknown)}")
            params.update(deepcopy(values))
        if self.mode == "regressor":
            # Preserve the old AUC-name alias, but the fitted key is explicit.
            alias = (self.general or {}).get("column_auc_threshold", self.column_auc_threshold)
            if alias is not None:
                params["r2_threshold"] = alias
            params["column_auc_threshold"] = params["r2_threshold"]
        for name in ["max_iter", "n_codebook", "n_exact_copy"]:
            if (
                not isinstance(params[name], Integral)
                or isinstance(params[name], bool)
                or params[name] < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in [
            "n_neighbors",
            "n_estimators",
            "max_depth",
            "batch_size",
            "knee_min_points",
            "knee_savgol_window_length",
            "knee_savgol_polyorder",
        ]:
            if (
                not isinstance(params[name], Integral)
                or isinstance(params[name], bool)
                or params[name] < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in ["test_size", "validation_fraction"]:
            if not isinstance(params[name], Real) or not 0 < params[name] < 1:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        for name in [
            "improvement_threshold",
            "learning_rate",
            "column_auc_threshold",
            "r2_threshold",
        ]:
            value = params.get(name)
            if value is not None and (
                not isinstance(value, Real) or not np.isfinite(value) or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative number")
        mutation = params["n_features_to_mutate"]
        if mutation != "all" and (
            not isinstance(mutation, Integral) or isinstance(mutation, bool) or mutation < 1
        ):
            raise ValueError("n_features_to_mutate must be 'all' or a positive integer")
        if params["n_jobs"] == 0 or not isinstance(params["n_jobs"], Integral):
            raise ValueError("n_jobs must be a nonzero integer")
        if not isinstance(params["device"], str) or not (
            re.fullmatch(r"cpu|cuda(?::[0-9]+)?", params["device"])
        ):
            raise ValueError("device must be 'cpu', 'cuda' or 'cuda:<ordinal>'")
        if params.get("early_stopping_rounds") is not None and params["early_stopping_rounds"] < 1:
            raise ValueError("early_stopping_rounds must be positive")
        seed = (self.general or {}).get("random_state", self.random_state)
        rng = check_random_state(seed)
        params["random_state"] = (
            int(seed) if isinstance(seed, Integral) else int(rng.randint(0, 2**31 - 1))
        )
        return params

    def fit(self, X, y=None):
        """Learn donors, initialization and scorers from X; y is ignored."""
        params = self._resolve_params()
        # A second fit must not retain fitted state from an earlier dataset.
        for key in list(vars(self)):
            if key.endswith("_"):
                delattr(self, key)
        original = X
        codec = _TableCodec(self.categorical_features)
        X = codec.fit_transform(X)
        X = validate_data(self, X, dtype=np.float64, ensure_all_finite="allow-nan", copy=True)
        engine = _MIAMIEngine(
            mode=self.mode, categorical_columns=set(codec.categories), **params
        ).fit(X)
        completed = engine.scaler.inverse_transform(engine.X_cur)
        completed[~np.isnan(X)] = X[~np.isnan(X)]
        self._engine_ = engine
        self._fit_X_ = X.copy()
        self._codec_ = codec
        self._fit_completed_ = completed
        self.imputed_X_ = codec.inverse_transform(completed, original)
        self.empty_features_ = engine.empty_features.copy()
        self.n_iter_ = engine.n_iter
        self.iteration_metrics_ = deepcopy(engine.iteration_metrics)
        self.effective_params_ = params
        self._seed_ = params["random_state"]
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Run the paper's iterative repair on the training data itself."""
        if fit_params:
            raise TypeError(f"Unsupported fit parameters: {sorted(fit_params)}")
        return self.fit(X, y).imputed_X_.copy()

    def transform(self, X):
        """Impute new rows using frozen training state, without fitting."""
        check_is_fitted(self, "_engine_")
        original = X
        if hasattr(X, "columns"):
            validate_data(self, X, reset=False, skip_check_array=True)
        X = self._codec_.transform(X)
        X = validate_data(
            self, X, reset=False, dtype=np.float64, ensure_all_finite="allow-nan", copy=True
        )
        if np.array_equal(X, self._fit_X_, equal_nan=True):
            return self._codec_.inverse_transform(self._fit_completed_, original)
        if not np.isnan(X).any():
            return self._codec_.inverse_transform(X, original)
        engine = self._engine_
        normalized = engine.scaler.transform(X)
        completed = engine.initial_imputer.transform(normalized)
        engine.snap_categories(completed, np.isnan(X))
        mask = np.isnan(X)
        if engine.n_iter > 0:
            # Each new row gets an order-independent proposal stream. Neither
            # the fitted generator nor global RNG state is advanced.
            for row in np.flatnonzero(mask.any(axis=1)):
                key = np.nan_to_num(X[row], nan=0.0).astype("<f8").tobytes() + mask[row].tobytes()
                seed = (
                    int.from_bytes(hashlib.sha256(key).digest()[:4], "little") + self._seed_
                ) % (2**32)
                candidates = MixedGenerator(
                    self.effective_params_["n_codebook"],
                    self.effective_params_["n_exact_copy"],
                    self.effective_params_["n_features_to_mutate"],
                    random_state=seed,
                )
                candidates.X_full = engine.X_imputed
                candidates.feature_available = engine.feature_available
                completed[row : row + 1] = engine.correct_subset(
                    completed[row : row + 1],
                    mask[row : row + 1],
                    columns_to_skip=engine.last_columns_to_skip,
                    generator=candidates,
                )
        completed = engine.scaler.inverse_transform(completed)
        completed[~mask] = X[~mask]
        return self._codec_.inverse_transform(completed, original)

    def get_iteration_metrics(self):
        """Return a copy of training diagnostics."""
        check_is_fitted(self, "iteration_metrics_")
        return deepcopy(self.iteration_metrics_)
