"""Public estimator contracts and numerical behavior on small real fits."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from MiamiImputer import MIAMIImputer


def toy(seed=8, rows=36):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 4))
    X[rng.random(X.shape) < 0.2] = np.nan
    return X


def small(mode="mask", **kwargs):
    options = dict(
        mode=mode, max_iter=2, n_estimators=5, n_codebook=8, n_exact_copy=8, use_knee=False
    )
    options.update(kwargs)
    return MIAMIImputer(**options)


@pytest.mark.parametrize("mode", ["mask", "regressor"])
def test_fit_transform_and_unseen_rows(mode):
    X, new = toy(), toy(9, 7)
    original = X.copy()
    imputer = small(mode)
    result = imputer.fit_transform(X, np.arange(len(X)))
    np.testing.assert_equal(X, original)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[~np.isnan(X)], X[~np.isnan(X)])
    np.testing.assert_array_equal(imputer.transform(X), result)
    state = pickle.dumps(imputer)
    output = imputer.transform(new)
    assert output.shape == new.shape
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output[~np.isnan(new)], new[~np.isnan(new)])
    np.testing.assert_array_equal(imputer.transform(new), output)
    assert pickle.dumps(imputer) == state
    np.testing.assert_array_equal(imputer.transform(new[::-1])[::-1], output)
    np.testing.assert_array_equal(np.vstack([imputer.transform(row[None]) for row in new]), output)
    np.testing.assert_array_equal(pickle.loads(state).transform(new), output)


@pytest.mark.parametrize("mode", ["mask", "regressor"])
def test_refit_resets_state_and_preserves_global_rng(mode):
    X, Y = toy(), toy(12, 32)
    np.random.seed(987)
    expected = np.random.RandomState(987).random_sample(5)
    estimator = small(mode)
    estimator.fit(X)
    np.testing.assert_array_equal(np.random.random_sample(5), expected)
    reused = estimator.fit_transform(Y)
    np.testing.assert_array_equal(reused, small(mode).fit_transform(Y))
    assert len(estimator.iteration_metrics_) <= estimator.max_iter


def test_constructor_clone_and_grouped_configuration():
    settings = {"max_depth": 3, "n_estimators": 7}
    estimator = small("regressor", discriminator=settings, general={"r2_threshold": 0.4})
    params = deepcopy(estimator.get_params())
    cloned = clone(estimator).set_params(max_iter=1)
    cloned.fit(toy())
    assert cloned.effective_params_["max_depth"] == 3
    assert cloned.effective_params_["n_estimators"] == 7
    assert cloned.effective_params_["column_auc_threshold"] == 0.4
    assert settings == {"max_depth": 3, "n_estimators": 7}
    assert estimator.get_params() == params
    assert not hasattr(estimator, "n_features_in_")
    with pytest.raises(ValueError, match="Unknown"):
        small(discriminator={"typo": 3}).fit(toy())


@pytest.mark.parametrize("mode", ["mask", "regressor"])
def test_empty_columns_and_complete_tables(mode):
    X = toy()
    X[:, 1] = np.nan
    imputer = small(mode)
    result = imputer.fit_transform(X)
    assert result.shape == X.shape
    np.testing.assert_array_equal(result[:, 1], 0)
    new = np.array([[np.nan, 99, 3, np.nan], [1, np.nan, 2, 4]], dtype=float)
    output = imputer.transform(new)
    assert output[0, 1] == 99
    assert output[1, 1] == 0
    assert np.isfinite(output).all()
    complete = np.arange(24, dtype=float).reshape(6, 4)
    np.testing.assert_array_equal(imputer.fit_transform(complete), complete)
    np.testing.assert_array_equal(imputer.transform(complete + 100), complete + 100)
    assert imputer.n_iter_ == 0


@pytest.mark.parametrize("mode", ["mask", "regressor"])
def test_one_missing_column_and_small_samples(mode):
    X = np.arange(80, dtype=float).reshape(20, 4)
    X[::3, 0] = np.nan
    assert np.isfinite(small(mode).fit_transform(X)).all()
    for data in [np.array([[np.nan, 1.0]]), np.full((4, 2), np.nan)]:
        assert small(mode).fit_transform(data).shape == data.shape


def test_pipeline_grid_search_and_feature_names():
    frame = pd.DataFrame(toy(), columns=list("abcd"))
    imputer = small(max_iter=0).set_output(transform="pandas")
    output = imputer.fit_transform(frame)
    assert output.columns.tolist() == frame.columns.tolist()
    np.testing.assert_array_equal(imputer.get_feature_names_out(), frame.columns)
    with pytest.raises(ValueError, match="feature names"):
        imputer.transform(frame[["b", "a", "c", "d"]])
    pipeline = Pipeline([("imputer", small(max_iter=1)), ("model", Ridge())])
    cv = GridSearchCV(pipeline, {"imputer__improvement_threshold": [0, 0.1]}, cv=2)
    cv.fit(frame, np.arange(len(frame), dtype=float))
    assert np.isfinite(cv.predict(frame.iloc[:5])).all()


def test_validation_and_fitted_state():
    with pytest.raises(NotFittedError):
        small().transform(toy())
    for kwargs in [
        dict(max_iter=-1),
        dict(batch_size=0),
        dict(mode="bad"),
        dict(test_size=1),
        dict(n_features_to_mutate=0),
    ]:
        with pytest.raises(ValueError):
            small(**kwargs).fit(toy())
    with pytest.raises(ValueError):
        small().fit(np.array([[1, np.inf], [2, 3]]))
    estimator = small().fit(toy())
    with pytest.raises(ValueError, match="features"):
        estimator.transform(np.zeros((3, 2)))


@pytest.mark.parametrize("mode", ["mask", "regressor"])
def test_sklearn_estimator_contract(mode):
    check_estimator(
        small(mode, max_iter=1),
        expected_failed_checks={
            "check_transformer_n_iter": "Complete tables require zero correction iterations; n_iter_ reports this truthfully."
        },
    )
