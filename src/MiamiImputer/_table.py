"""Internal encoding and restoration of mixed tabular inputs."""

from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
from sklearn.utils.validation import check_array


class _TableCodec:
    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features

    @staticmethod
    def _array(X):
        if isinstance(X, pd.DataFrame):
            raw = X.to_numpy(dtype=object, na_value=np.nan, copy=True)
        elif isinstance(X, (list, tuple)):
            raw = np.asarray(X, dtype=object)
        else:
            raw = X
        return check_array(raw, dtype=None, ensure_all_finite=False, copy=True)

    def fit_transform(self, X):
        raw = self._array(X)
        self.n_features = raw.shape[1]
        explicit = set()
        if self.categorical_features is not None:
            if not isinstance(self.categorical_features, (list, tuple, np.ndarray)):
                raise ValueError("categorical_features must be a list of column names or indices")
            for column in self.categorical_features:
                if isinstance(column, str) and isinstance(X, pd.DataFrame):
                    matches = np.flatnonzero(X.columns == column)
                    if len(matches) != 1:
                        raise ValueError(f"Categorical column {column!r} must exist exactly once")
                    explicit.add(int(matches[0]))
                elif isinstance(column, Integral) and not isinstance(column, bool):
                    if not 0 <= column < self.n_features:
                        raise ValueError(f"Categorical column index {column} is out of range")
                    explicit.add(int(column))
                else:
                    raise ValueError("Use column names for DataFrames or integer column indices")
        self.categories = {}
        for j in range(self.n_features):
            values = raw[:, j]
            observed = values[~pd.isna(values)]
            if not all(isinstance(v, (str, Real, np.bool_)) for v in observed):
                raise TypeError(
                    "Column values must be a string or a number; nested objects and dates are unsupported"
                )
            dtype = X.dtypes.iloc[j] if isinstance(X, pd.DataFrame) else values.dtype
            categorical = (
                j in explicit
                or isinstance(dtype, pd.CategoricalDtype)
                or is_bool_dtype(dtype)
                or (len(observed) > 0 and all(isinstance(v, (bool, np.bool_)) for v in observed))
                or (len(observed) > 0 and not all(isinstance(v, Real) for v in observed))
                or isinstance(dtype, pd.StringDtype)
            )
            if categorical:
                if not len(observed):
                    raise ValueError(
                        f"Categorical column {j} has no observed values to impute from"
                    )
                self.categories[j] = pd.unique(observed).tolist()
        return self.transform(X)

    def transform(self, X):
        raw = self._array(X)
        if raw.shape[1] != self.n_features:
            raise ValueError(
                f"X has {raw.shape[1]} features, but MIAMIImputer is expecting "
                f"{self.n_features} features as input."
            )
        encoded = np.full(raw.shape, np.nan, dtype=float)
        for j in range(self.n_features):
            observed = ~pd.isna(raw[:, j])
            values = raw[observed, j]
            if j in self.categories:
                mapping = {value: k for k, value in enumerate(self.categories[j])}
                # Unseen observed labels are preserved when decoding. They never
                # become donors or candidate labels for a missing coordinate.
                encoded[observed, j] = [mapping.get(value, -1) for value in values]
            else:
                encoded[observed, j] = values.astype(float)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(encoded, index=X.index, columns=X.columns)
        return encoded

    def inverse_transform(self, completed, X):
        raw = self._array(X)
        missing = pd.isna(raw)
        output = (
            completed.astype(object)
            if self.categories or isinstance(X, pd.DataFrame)
            else completed.copy()
        )
        for j, labels in self.categories.items():
            codes = np.clip(np.rint(completed[:, j]).astype(int), 0, len(labels) - 1)
            output[:, j] = np.asarray(labels, dtype=object)[codes]
        output[~missing] = raw[~missing]
        if not isinstance(X, pd.DataFrame):
            return output
        columns = []
        for j in range(self.n_features):
            values = output[:, j]
            dtype = X.dtypes.iloc[j]
            if isinstance(dtype, pd.CategoricalDtype):
                labels = list(dtype.categories)
                labels.extend(v for v in pd.unique(values) if v not in labels)
                dtype = pd.CategoricalDtype(labels, ordered=dtype.ordered)
            elif is_integer_dtype(dtype) and not is_bool_dtype(dtype):
                numeric = np.asarray(values, dtype=float)
                if np.any(numeric != np.trunc(numeric)):
                    dtype = (
                        "Float64" if isinstance(dtype, pd.api.extensions.ExtensionDtype) else float
                    )
            elif is_numeric_dtype(dtype) and j not in self.categories:
                values = np.asarray(values, dtype=float)
            columns.append(pd.Series(values, index=X.index, dtype=dtype))
        result = pd.concat(columns, axis=1)
        result.columns = X.columns.copy()
        return result
