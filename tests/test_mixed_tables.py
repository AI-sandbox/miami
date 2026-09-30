"""User-facing mixed table formats, category support and dtype restoration."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from MiamiImputer import MIAMIImputer

from .test_devices import cuda_device  # noqa: F401


def imputer(**kwargs):
    options = dict(max_iter=2, n_estimators=5, n_codebook=8, n_exact_copy=8, use_knee=False)
    options.update(kwargs)
    return MIAMIImputer(**options)


def mixed_table():
    frame = pd.DataFrame(
        {
            "amount": pd.array([1.5, None, 3.0, 4.5, 5.0, 6.0] * 6, dtype="Float64"),
            "city": pd.Categorical(["Paris", None, "Rome", "Rome", "Paris", "Rome"] * 6),
            "label": pd.array(["a", "b", pd.NA, "b", "a", "a"] * 6, dtype="string"),
            "flag": pd.array([True, False, pd.NA, True, False, True] * 6, dtype="boolean"),
            "count": pd.array([1, 2, None, 4, 5, 6] * 6, dtype="Int64"),
        },
        index=pd.Index(range(100, 136), name="row"),
    )
    return frame


@pytest.mark.parametrize("mode", ["mask", "regressor"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_mixed_dataframe_real_fit(mode, device, request):
    if device.startswith("cuda"):
        request.getfixturevalue("cuda_device")
    X = mixed_table()
    original = X.copy(deep=True)
    model = imputer(mode=mode, device=device)
    result = model.fit_transform(X)
    pd.testing.assert_frame_equal(X, original)
    assert isinstance(result, pd.DataFrame)
    assert not result.isna().any().any()
    pd.testing.assert_index_equal(result.index, X.index)
    pd.testing.assert_index_equal(result.columns, X.columns)
    assert result.dtypes.equals(X.dtypes)
    for col in X:
        observed = X[col].notna()
        pd.testing.assert_series_equal(result.loc[observed, col], X.loc[observed, col])
    for col in ["city", "label", "flag"]:
        assert set(result[col]) <= set(X[col].dropna())
    restored = pickle.loads(pickle.dumps(model))
    pd.testing.assert_frame_equal(restored.transform(X), result)
    assert clone(model).get_params() == model.get_params()


@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_mixed_array_and_list_inputs(container):
    rows = [[1.0, "red", True], [2.0, "blue", False], [np.nan, None, None]]
    X = np.array(rows, dtype=object) if container is np.array else container(rows)
    result = imputer(max_iter=0).fit_transform(X)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    assert not pd.isna(result).any()
    assert result[0, 0] == 1.0 and not isinstance(result[0, 0], str)
    assert set(result[:, 1]) <= {"red", "blue"}
    assert set(result[:, 2]) <= {True, False}


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_numeric_missing_representations(missing):
    X = [[1.0, 2.0], [3.0, 4.0], [missing, 6.0]]
    result = imputer(max_iter=0).fit_transform(X)
    assert result.dtype.kind == "f"
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[:2], np.asarray(X[:2], dtype=float))


def test_numeric_category_codes_and_nondefault_knn():
    X = pd.DataFrame({"kind": [10.0, 20.0, 30.0, np.nan], "x": [1.0, 1.0, 1.0, 1.0]})
    model = imputer(max_iter=0, categorical_features=["kind"], n_neighbors=2)
    result = model.fit_transform(X)
    assert result.loc[3, "kind"] in {10, 20, 30}
    engine = model._engine_
    assert set(engine.X_imputed[:, 0]) <= set(engine.codebooks[0])
    array_result = imputer(max_iter=0, categorical_features=[0], n_neighbors=2).fit_transform(
        X.to_numpy()
    )
    assert array_result[3, 0] in {10, 20, 30}


def test_integer_columns_promote_without_truncating_imputations():
    X = pd.DataFrame({"count": pd.array([1, 2, None], dtype="Int64"), "x": [0.0, 0.0, 0.0]})
    result = imputer(max_iter=0, n_neighbors=2).fit_transform(X)
    assert result.loc[2, "count"] == 1.5
    assert str(result["count"].dtype) == "Float64"


def test_new_categories_preserved_and_training_categories_used_for_missing():
    X = pd.DataFrame({"city": pd.Categorical(["Rome", "Paris", None]), "x": [1.0, 2.0, 1.0]})
    model = imputer(max_iter=0).fit(X)
    new = pd.DataFrame({"city": pd.Categorical(["Tokyo", None]), "x": [3.0, 2.0]}, index=[9, 8])
    before = new.copy(deep=True)
    result = model.transform(new)
    assert result.loc[9, "city"] == "Tokyo"
    assert result.loc[8, "city"] in {"Rome", "Paris"}
    assert not result.isna().any().any()
    pd.testing.assert_frame_equal(new, before)
    pd.testing.assert_index_equal(result.index, new.index)


def test_pandas_output_option_and_reordered_columns():
    X = mixed_table()
    model = imputer(max_iter=0).set_output(transform="pandas")
    result = model.fit_transform(X)
    assert result.dtypes.equals(X.dtypes)
    with pytest.raises(ValueError, match="feature names"):
        model.transform(X[["city", "amount", "label", "flag", "count"]])


def test_all_missing_categorical_column_has_clear_error():
    X = pd.DataFrame({"city": pd.Categorical([None, None], categories=["Rome"]), "x": [1, 2]})
    with pytest.raises(ValueError, match="no observed values"):
        imputer().fit_transform(X)


def test_refitting_on_numeric_data_resets_category_mapping():
    model = imputer(max_iter=0).fit(mixed_table())
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    result = model.fit_transform(X)
    assert result.dtype.kind == "f"
    assert model._codec_.categories == {}


@pytest.mark.parametrize("columns", [["missing"], [20], "city"])
def test_invalid_categorical_features(columns):
    with pytest.raises(ValueError, match="[Cc]ategorical|column"):
        imputer(categorical_features=columns).fit(mixed_table())
