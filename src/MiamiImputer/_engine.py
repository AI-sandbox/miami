"""Private transductive iteration and candidate-selection implementation."""

import time

import numpy as np
from kneed import KneeLocator
from scipy.signal import savgol_filter
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .codebook import build_codebooks
from .config import FLAT_DEFAULTS
from .discriminator import get_predictor
from .generator import MixedGenerator
from .scorer import get_scorer


class _MIAMIEngine:
    """Private iterative repair engine; the public estimator owns validation."""

    def __init__(self, mode: str = "mask", **kwargs):
        if mode not in FLAT_DEFAULTS:
            raise ValueError(f"mode must be {list(FLAT_DEFAULTS.keys())}, got '{mode}'")
        self.mode = mode

        # Public estimator supplies resolved flat options.
        for k, v in FLAT_DEFAULTS[mode].items():
            setattr(self, k, v)

        flat_overrides = kwargs
        for k, v in flat_overrides.items():
            setattr(self, k, v)

        # Keep the flat overrides for downstream (discriminator kwargs)
        self.kwargs = flat_overrides

        self.scaler = None
        self.predictor = None
        self.scorer = None
        self.candidates = None
        self.X_cur = None
        self.iteration_metrics = []

        self.rng = np.random.RandomState(self.random_state)
        self.n_iter = 0
        self.last_columns_to_skip = set()

    def _detect_column_knee(self, col_idx):
        """
        Detect knee point in a column's AUC history using savgol_filter
        smoothing and KneeLocator.

        Returns
        -------
        bool
            True if a knee (convergence point) has been detected.
        """
        history = self.column_aucs_history.get(col_idx, [])
        if len(history) < self.knee_min_points:
            return False

        y = np.array(history)
        x = np.arange(len(y))

        min_len = min(len(y), self.knee_savgol_window_length)
        if min_len < 3:
            min_len = 3
        if min_len % 2 == 0:
            min_len = min_len - 1

        polyorder = min(self.knee_savgol_polyorder, min_len - 1)
        if polyorder < 1:
            polyorder = 1

        try:
            if len(y) >= min_len and min_len >= 3:
                y_smooth = savgol_filter(y, window_length=min_len, polyorder=polyorder)
            else:
                y_smooth = y

            kl = KneeLocator(x, y_smooth, curve="convex", direction="decreasing", S=self.knee_S)
            return kl.knee is not None
        except (ValueError, FloatingPointError):
            return False

    def fit(self, X: np.ndarray) -> "_MIAMIEngine":
        """
        Fit the MIAMI imputer.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data WITH NaNs (in original scale).

        Returns
        -------
        self : MIAMIImputer
        """
        # Always fit scaling on training observations; retain empty columns.
        self.empty_features = np.isnan(X).all(axis=0)
        scaling_data = X.copy()
        scaling_data[:, self.empty_features] = 0.0
        self.scaler = MinMaxScaler().fit(scaling_data)
        X_normalized = self.scaler.transform(X)

        self.X = X.copy()
        self.mask = np.isnan(X)

        t_cb = time.time()
        self.codebooks = build_codebooks(X_normalized, self.mask)
        if self.verbose:
            print(f"  Codebooks built in {time.time() - t_cb:.3f}s")

        t_impute = time.time()
        self.initial_imputer = KNNImputer(n_neighbors=self.n_neighbors, keep_empty_features=True)
        self.X_imputed = self.initial_imputer.fit_transform(X_normalized)
        self.snap_categories(self.X_imputed, self.mask)
        if self.verbose:
            print(f"  KNN imputation in {time.time() - t_impute:.3f}s")

        self.n_samples, self.n_features = self.X_imputed.shape

        cols_with_missing = [col for col in range(self.n_features) if self.mask[:, col].any()]
        self.cols_with_missing = cols_with_missing

        self.miss_pct = self.mask.mean(axis=1)

        self.feature_available = {}
        for feat in range(self.n_features):
            self.feature_available[feat] = np.where(~self.mask[:, feat])[0]

        if self.mode == "mask":
            cols_without_missing = [
                col for col in range(self.n_features) if not self.mask[:, col].any()
            ]
            self.column_done = {
                col: (col in cols_without_missing) for col in range(self.n_features)
            }
            self.column_aucs_history = {col: [] for col in range(self.n_features)}
            if self.verbose:
                print(
                    f"  Column tracking: {len(cols_without_missing)} cols without missing "
                    f"(marked done), {len(cols_with_missing)} cols with missing"
                )

        # Only scorer parameters reach the private predictor constructors.
        common = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "random_state",
            "early_stopping_rounds",
            "device",
            "n_jobs",
            "validation_fraction",
        }
        specific = (
            {"colsample_bylevel"}
            if self.mode == "mask"
            else {
                "colsample_bytree",
                "reg_lambda",
                "reg_alpha",
                "gamma",
                "min_child_weight",
                "window_size",
                "objective",
            }
        )
        xgb_kwargs = {key: value for key, value in self.kwargs.items() if key in common | specific}
        self.predictor = get_predictor(
            mode=self.mode, cols_with_missing=self.cols_with_missing, **xgb_kwargs
        )
        if self.verbose:
            print(f"  Predictor: {self.predictor.__class__.__name__}")

        self.scorer = get_scorer(mode=self.mode)

        self.candidates = MixedGenerator(
            n_codebook=self.n_codebook,
            n_exact_copy=self.n_exact_copy,
            n_features_to_mutate=self.n_features_to_mutate,
            random_state=self.rng,
        )
        self.candidates.codebooks = self.codebooks
        self.candidates.X_full = self.X_imputed
        self.candidates.feature_available = self.feature_available

        self.X_cur = self.X_imputed.copy()

        if self.verbose:
            self._print_info()

        if (
            self.mask.any()
            and self.n_samples >= 4
            and int(self.n_samples * (1 - self.test_size)) >= 2
            and self.max_iter > 0
        ):
            self.run_iterations()

        return self

    def snap_categories(self, X, mask):
        """Keep kNN-filled category codes on observed support before scoring."""
        for col in getattr(self, "categorical_columns", ()):
            support = self.codebooks[col]
            values = X[mask[:, col], col]
            right = np.clip(np.searchsorted(support, values), 0, len(support) - 1)
            left = np.maximum(right - 1, 0)
            nearest = np.where(
                np.abs(values - support[left]) <= np.abs(values - support[right]), left, right
            )
            X[mask[:, col], col] = support[nearest]

    def run_iterations(self):
        """Main correction loop."""
        is_classifier = getattr(self.predictor, "is_classifier", False)

        if self.verbose:
            print(f"\n=== Starting {self.max_iter} iterations (mode={self.mode}) ===")

        self._global_r2_knee_detected = False

        for iteration in range(self.max_iter):
            if self._global_r2_knee_detected:
                if self.verbose:
                    print(f"\n=== R² knee detected, stopping at iteration {iteration} ===")
                break

            if self.mode == "mask":
                all_columns_done = all(self.column_done[col] for col in self.cols_with_missing)
                if all_columns_done:
                    if self.verbose:
                        print(
                            f"\n=== All {len(self.cols_with_missing)} columns with "
                            f"missing done at iteration {iteration} - stopping ==="
                        )
                    break

            t_iter_start = time.time()
            if self.verbose:
                print(f"\n=== Iteration {iteration} ===")

            self.split_data_asymmetric(iteration)

            X_tr_current = self.X_cur[self.idx_train]
            X_corr = self.X_cur[self.idx_correct]
            mask_corr = self.mask[self.idx_correct]
            miss_pct_tr = self.miss_pct[self.idx_train]
            miss_pct_corr = self.miss_pct[self.idx_correct]
            mask_tr = self.mask[self.idx_train]

            X_tr_combined = X_tr_current
            mask_tr_combined = mask_tr
            miss_pct_tr_combined = miss_pct_tr

            if is_classifier:
                self.predictor.fit(X_tr_combined, mask_tr_combined)
            else:
                self.predictor.fit(X_tr_combined, miss_pct_tr_combined)

            t_fit_end = time.time()

            if self.mode == "mask":
                columns_to_skip = {col for col, done in self.column_done.items() if done}

                mask_val_filtered = mask_corr[:, self.cols_with_missing]
                X_val_before = X_corr
                column_metrics = self.predictor.get_metrics_per_column(
                    X_val_before, mask_val_filtered
                )
                raw_aucs_cb = column_metrics.get("column_aucs", [])
                if raw_aucs_cb:
                    if len(self.cols_with_missing) < self.n_features:
                        raw_aucs = [-1] * self.n_features
                        for i, col_idx in enumerate(self.cols_with_missing):
                            if i < len(raw_aucs_cb):
                                raw_aucs[col_idx] = raw_aucs_cb[i]
                    else:
                        raw_aucs = raw_aucs_cb
                else:
                    raw_aucs = [-1] * self.n_features

                column_aucs = {f"col_{i}": auc for i, auc in enumerate(raw_aucs)}
                n_columns_corrected = 0

                for col_idx, auc in enumerate(raw_aucs):
                    self.column_aucs_history[col_idx].append(auc)

                for col_idx, col_auc in enumerate(raw_aucs):
                    if col_idx in columns_to_skip:
                        continue
                    if col_auc == -1:
                        columns_to_skip.add(col_idx)
                        continue
                    if col_auc < self.column_auc_threshold:
                        columns_to_skip.add(col_idx)
                        if self.verbose:
                            print(
                                f"    Column {col_idx} skipping "
                                f"(AUC {col_auc:.3f} < {self.column_auc_threshold})"
                            )

                    if self.use_knee and self._detect_column_knee(col_idx):
                        self.column_done[col_idx] = True
                        columns_to_skip.add(col_idx)
                        if self.verbose:
                            print(f"    Column {col_idx} marked as done (knee detected)")

                self.last_columns_to_skip = columns_to_skip.copy()
                if len(columns_to_skip) < self.n_features:
                    X_corr_new = self.correct_subset(
                        X_corr,
                        mask_corr,
                        miss_pct_sub=miss_pct_corr,
                        columns_to_skip=columns_to_skip,
                    )
                    self.X_cur[self.idx_correct] = X_corr_new
                    n_samples_modified = int(np.any(X_corr_new != X_corr, axis=1).sum())
                else:
                    n_samples_modified = 0

                n_columns_corrected = sum(
                    col not in columns_to_skip and mask_corr[:, col].any()
                    for col in self.cols_with_missing
                )

                iter_metrics = {
                    "iteration": iteration,
                    "n_samples_modified": n_samples_modified,
                    "column_aucs": column_aucs,
                    "n_columns_corrected": n_columns_corrected,
                    "iter_total_time": time.time() - t_iter_start,
                }

            elif self.mode == "regressor":
                predictions = self.predictor.predict(X_corr)
                predictor_metrics = self.predictor.get_metrics(X_corr, miss_pct_corr)
                r2_correction = predictor_metrics.get("r2", 0.0)
                r2_train = predictor_metrics.get("train_r2", 0.0)
                r2_val = predictor_metrics.get("val_r2", 0.0)
                r2_corr_clipped = max(0, r2_correction)
                r2_train_clipped = max(0, r2_train)
                r2_val_clipped = max(0, r2_val)

                rmse_correction = float(np.sqrt(np.mean((predictions - miss_pct_corr) ** 2)))

                # R² knee (decreasing) — same savgol+KneeLocator as mask.
                # R² starts high and decays toward 0 as X_cur becomes
                # indistinguishable; the knee is where the decay stalls.
                if self.use_knee:
                    if not hasattr(self, "_r2_knee_history"):
                        self._r2_knee_history = []
                        self._r2_knee_window_mean_history = []

                    self._r2_knee_history.append(r2_corr_clipped)

                    if len(self._r2_knee_history) >= self.knee_savgol_window_length:
                        window_mean = np.mean(
                            self._r2_knee_history[-self.knee_savgol_window_length :]
                        )
                        self._r2_knee_window_mean_history.append(window_mean)

                        if len(self._r2_knee_window_mean_history) >= self.knee_min_points:
                            y = np.array(self._r2_knee_window_mean_history)
                            x = np.arange(len(y))

                            min_len = min(len(y), self.knee_savgol_window_length)
                            if min_len < 3:
                                min_len = 3
                            if min_len % 2 == 0:
                                min_len = min_len - 1
                            polyorder = min(self.knee_savgol_polyorder, min_len - 1)
                            if polyorder < 1:
                                polyorder = 1

                            try:
                                if len(y) >= min_len and min_len >= 3:
                                    y_smooth = savgol_filter(
                                        y, window_length=min_len, polyorder=polyorder
                                    )
                                else:
                                    y_smooth = y
                                kl = KneeLocator(
                                    x,
                                    y_smooth,
                                    curve="convex",
                                    direction="decreasing",
                                    S=self.knee_S,
                                )
                                if kl.knee is not None:
                                    if self.verbose:
                                        print(f"    R² knee at iteration {iteration}")
                                    self._global_r2_knee_detected = True
                            except (ValueError, FloatingPointError):
                                pass

                if self.verbose:
                    print(
                        f"    R² train={r2_train_clipped:.3f} "
                        f"val={r2_val_clipped:.3f} "
                        f"correction={r2_corr_clipped:.3f}  "
                        f"RMSE={rmse_correction:.5f}"
                    )

                self.last_columns_to_skip = (
                    set(range(self.n_features))
                    if r2_corr_clipped < self.column_auc_threshold
                    else set()
                )
                if r2_corr_clipped < self.column_auc_threshold:
                    if self.verbose:
                        print(
                            f"    R² {r2_corr_clipped:.3f} < "
                            f"{self.column_auc_threshold}, skipping corrections"
                        )
                    n_samples_modified = 0
                    correction_applied = False
                else:
                    X_corr_new = self.correct_subset(X_corr, mask_corr, miss_pct_sub=miss_pct_corr)
                    self.X_cur[self.idx_correct] = X_corr_new
                    n_samples_modified = int(np.any(X_corr_new != X_corr, axis=1).sum())
                    correction_applied = n_samples_modified > 0

                iter_metrics = {
                    "iteration": iteration,
                    "r2": r2_corr_clipped,
                    "r2_train": r2_train_clipped,
                    "r2_val": r2_val_clipped,
                    "rmse_correction": rmse_correction,
                    "n_samples_modified": n_samples_modified,
                    "correction_applied": correction_applied,
                    "iter_total_time": time.time() - t_iter_start,
                }

            self.n_iter = iteration + 1
            if self.compute_iteration_metrics:
                self.iteration_metrics.append(iter_metrics)

            if self.verbose:
                t_total = time.time() - t_iter_start
                if self.mode == "mask":
                    n_done = sum(self.column_done[col] for col in self.cols_with_missing)
                    print(
                        f"  Iter {iteration}: "
                        f"fit={t_fit_end - t_iter_start:.3f}s, "
                        f"total={t_total:.3f}s "
                        f"[{n_done}/{len(self.cols_with_missing)} cols done]"
                    )
                else:
                    print(
                        f"  Iter {iteration}: "
                        f"fit={t_fit_end - t_iter_start:.3f}s, "
                        f"total={t_total:.3f}s "
                        f"[modified={n_samples_modified}]"
                    )

        if self.verbose:
            print(f"\n=== Completed {self.n_iter} iterations ===")

    def split_data_asymmetric(self, iteration: int):
        """Deterministic split with different seed per iteration."""
        indices = np.arange(self.n_samples)
        self.idx_train, self.idx_correct = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=(self.random_state + iteration) % (2**32),
        )

    def correct_subset(
        self,
        X_sub: np.ndarray,
        mask_sub: np.ndarray,
        miss_pct_sub: np.ndarray = None,
        columns_to_skip: set = None,
        generator=None,
    ) -> np.ndarray:
        """
        Generate candidates, score them, and select the best for each sample.

        For each sample, generates n_codebook + n_exact_copy candidates.
        The original row is kept as candidate 0 to ensure monotonicity
        (score never increases). Only replaces if the improvement exceeds
        the improvement_threshold.

        Parameters
        ----------
        X_sub : ndarray of shape (n_sub_samples, n_features)
            Subset of data to correct.
        mask_sub : ndarray of shape (n_sub_samples, n_features)
            Boolean mask for the subset.
        miss_pct_sub : ndarray of shape (n_sub_samples,), optional
            Per-sample missing percentage.
        columns_to_skip : set, optional
            Column indices to exclude from correction.

        Returns
        -------
        X_corr : ndarray of shape (n_sub_samples, n_features)
            Corrected subset.
        """
        X_corr = X_sub.copy()
        n_samples, n_features = X_corr.shape

        if columns_to_skip is None:
            columns_to_skip = set()

        has_missing = np.any(mask_sub, axis=1)
        samples_idx = np.where(has_missing)[0]

        if len(samples_idx) == 0:
            return X_corr

        batch_size = max(1, getattr(self, "batch_size", 4))
        n_batches = (len(samples_idx) + batch_size - 1) // batch_size
        generator = self.candidates if generator is None else generator

        is_classifier = getattr(self.predictor, "is_classifier", False)
        cols_to_update = [c for c in range(n_features) if c not in columns_to_skip]

        if not cols_to_update:
            return X_corr

        for batch_num in range(n_batches):
            batch_start = batch_num * batch_size
            batch_end = (
                min((batch_num + 1) * batch_size, len(samples_idx))
                if batch_num < n_batches - 1
                else len(samples_idx)
            )
            batch_indices = samples_idx[batch_start:batch_end]

            X_batch = X_corr[batch_indices]
            mask_batch = mask_sub[batch_indices]
            miss_pct_batch = miss_pct_sub[batch_indices] if miss_pct_sub is not None else None

            generated = generator.generate(X_batch, mask_batch, self.codebooks, columns_to_skip)
            n_per_sample = generator.n_candidates_per_sample

            all_candidate_rows = []
            metadata = []

            for idx in range(len(batch_indices)):
                all_candidate_rows.append(X_batch[idx : idx + 1])
                s, e = idx * n_per_sample, (idx + 1) * n_per_sample
                all_candidate_rows.append(generated[s:e])
                metadata.append((idx, 1 + n_per_sample))

            if len(all_candidate_rows) == 0:
                continue

            predict_batch = np.vstack(all_candidate_rows)
            allowed = mask_batch.copy()
            if columns_to_skip:
                allowed[:, list(columns_to_skip)] = False
            allowed = np.repeat(allowed, 1 + n_per_sample, axis=0)
            originals = np.repeat(X_batch, 1 + n_per_sample, axis=0)
            predict_batch = np.where(allowed, predict_batch, originals)
            raw_predictions = self.predictor.predict(predict_batch)
            metadata_tuples = [tuple(m) for m in metadata]

            if len(cols_to_update) < n_features and len(cols_to_update) > 0:
                mask_batch_filtered = mask_batch[:, cols_to_update]
                if is_classifier:
                    raw_predictions = raw_predictions[:, cols_to_update]
            else:
                mask_batch_filtered = mask_batch

            if is_classifier:
                scores = self.scorer.score(
                    raw_predictions, mask_batch=mask_batch_filtered, metadata=metadata_tuples
                )
            else:
                scores = self.scorer.score(
                    raw_predictions, miss_pct=miss_pct_batch, metadata=metadata_tuples
                )

            cursor = 0
            for idx, num_candidates in metadata_tuples:
                c_scores = scores[cursor : cursor + num_candidates]
                if len(c_scores) == 0:
                    cursor += num_candidates
                    continue

                original_score = c_scores[0]
                best_idx = (
                    np.argmin(c_scores) if self.scorer.lower_is_better else np.argmax(c_scores)
                )
                best_score = c_scores[best_idx]

                if original_score != 0:
                    improvement = (original_score - best_score) / abs(original_score)
                else:
                    improvement = 0

                if improvement > self.improvement_threshold:
                    best_candidate = predict_batch[cursor + best_idx]
                    X_batch[idx] = best_candidate

                cursor += num_candidates

            X_corr[batch_indices] = X_batch

        return X_corr

    def _print_info(self):
        """Print initialization summary."""
        total_entries = sum(len(v) for v in self.codebooks.values())
        print("  MIAMI Imputer initialized:")
        print(f"    Mode: {self.mode}")
        print(f"    Features: {self.n_features}, Samples: {self.n_samples}")
        print(f"    Codebooks: {len(self.codebooks)} features, {total_entries} total unique values")
        print(f"    Missing rate: {self.mask.mean():.2%}")
        print(
            f"    Candidates/sample: {self.candidates.n_candidates_per_sample} "
            f"(codebook={self.n_codebook}, exact={self.n_exact_copy})"
        )
        print(f"    Improvement threshold: {self.improvement_threshold}")
        if self.mode == "mask":
            print(f"    AUC threshold: {self.column_auc_threshold}")
            print(f"    Knee detection: {'enabled' if self.use_knee else 'disabled'}")
        elif self.mode == "regressor":
            print(f"    R² gate threshold: {self.column_auc_threshold}")

    def get_iteration_metrics(self):
        """Return iteration metrics collected during run_iterations()."""
        return self.iteration_metrics
