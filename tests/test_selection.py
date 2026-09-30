"""Exercise accepted repairs, score gates, batching and original donor masks."""

import numpy as np
import pytest

from MiamiImputer._engine import _MIAMIEngine
from MiamiImputer.generator import MixedGenerator
from MiamiImputer.scorer import get_scorer


@pytest.mark.parametrize("mode", ["mask", "regressor"])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_accepted_repairs_reduce_fixed_score_and_preserve_observations(mode, batch_size):
    class Predictor:
        is_classifier = mode == "mask"

        def predict(self, X):
            if self.is_classifier:
                return np.column_stack([X[:, 0], np.zeros(len(X))])
            return X[:, 0]

    engine = _MIAMIEngine(mode=mode, batch_size=batch_size)
    engine.predictor = Predictor()
    engine.scorer = get_scorer(mode)
    engine.codebooks = {0: np.array([0.0])}
    engine.candidates = MixedGenerator(0, 1, "all", random_state=42)
    engine.candidates.X_full = np.array([[0.0, 99.0]])
    engine.candidates.feature_available = {0: np.array([0]), 1: np.array([0])}
    X = np.column_stack([np.full(7, 0.8), np.arange(7)])
    mask = np.column_stack([np.ones(7, dtype=bool), np.zeros(7, dtype=bool)])
    mask[2, 0] = False
    result = engine.correct_subset(X, mask)
    np.testing.assert_array_equal(result[~mask], X[~mask])
    np.testing.assert_array_equal(result[mask], 0.0)
    assert np.all(result[:, 0] <= X[:, 0])
    np.testing.assert_array_equal(X[:, 0], 0.8)


@pytest.mark.parametrize("original,proposal", [(0.0, -1.0), (-1.0, -1.25)])
def test_zero_score_and_strict_relative_threshold(original, proposal):
    class Predictor:
        is_classifier = False

        def predict(self, X):
            return X[:, 0]

    engine = _MIAMIEngine(mode="regressor", improvement_threshold=0.25)
    engine.predictor = Predictor()
    engine.scorer = get_scorer("regressor")
    engine.codebooks = {}
    engine.candidates = MixedGenerator(0, 1, "all", random_state=42)
    engine.candidates.X_full = np.array([[proposal]])
    engine.candidates.feature_available = {0: np.array([0])}
    X = np.array([[original]])
    np.testing.assert_array_equal(engine.correct_subset(X, np.ones_like(X, dtype=bool)), X)


def test_previously_imputed_donor_coordinates_are_ineligible():
    generator = MixedGenerator(0, 3, "all", random_state=42)
    # Donor zero contains an imputed 999; only donor one was observed there.
    generator.X_full = np.array([[999.0, 2.0], [7.0, 3.0]])
    generator.feature_available = {0: np.array([1]), 1: np.array([0, 1])}
    result = generator.generate(np.array([[0.0, 8.0]]), np.array([[True, False]]), {})
    assert result[0, 0] == 7
    assert 999 not in result[:, 0]
    np.testing.assert_array_equal(result[:, 1], 8)
