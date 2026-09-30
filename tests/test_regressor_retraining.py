"""The regressor must forget previously fitted trees between outer iterations."""

import unittest
from unittest.mock import patch

import numpy as np


def _check_fresh_regressor():
    from MiamiImputer import MIAMIImputer, discriminator

    regressor = discriminator.XGBRegressor

    def cpu_regressor(**kwargs):
        kwargs.update(device="cpu", n_jobs=1)
        return regressor(**kwargs)

    original_fit = discriminator.IterativeXGBoostPredictor.fit
    fitted_models = []
    training_sets = []

    def capture_fit(self, X, y):
        original_fit(self, X, y)
        fitted_models.append(self._model)
        training_sets.append((X.copy(), y.copy()))
        return self

    with (
        patch.object(discriminator, "XGBRegressor", cpu_regressor),
        patch.object(discriminator.IterativeXGBoostPredictor, "fit", capture_fit),
    ):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(80, 4))
        X[rng.random(X.shape) < 0.25] = np.nan
        imputer = MIAMIImputer(
            mode="regressor",
            verbose=False,
            max_iter=3,
            use_knee=False,
            n_estimators=5,
            max_depth=2,
            early_stopping_rounds=100,
            r2_threshold=2,
            n_codebook=0,
            n_exact_copy=0,
        )
        completed = imputer.fit_transform(X)

        assert len(fitted_models) == 3
        assert len({id(model) for model in fitted_models}) == 3
        assert [model.get_booster().num_boosted_rounds() for model in fitted_models] == [5, 5, 5]
        assert np.isfinite(completed).all()
        np.testing.assert_allclose(completed[~np.isnan(X)], X[~np.isnan(X)])

        # A fit after prior iterations must match an independent fresh fit on the
        # same final training data, not an ensemble carrying earlier information.
        X_last, y_last = training_sets[-1]
        independent = discriminator.IterativeXGBoostPredictor(
            n_estimators=5,
            max_depth=2,
            early_stopping_rounds=100,
        )
        original_fit(independent, X_last, y_last)
        np.testing.assert_allclose(
            imputer._engine_.predictor.predict(X_last),
            independent.predict(X_last),
            rtol=0,
            atol=0,
        )


class RegressorRetrainingTest(unittest.TestCase):
    def test_each_iteration_matches_a_fresh_regressor(self):
        _check_fresh_regressor()


if __name__ == "__main__":
    unittest.main()
