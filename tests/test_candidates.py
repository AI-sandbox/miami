"""Regression tests for scored/applied candidates and observed donors."""

import numpy as np

from MiamiImputer._engine import _MIAMIEngine
from MiamiImputer.generator import CodebookGenerator, MixedGenerator
from MiamiImputer.scorer import MaskScorer


def test_skipped_coordinates_are_frozen_before_scoring():
    class Predictor:
        is_classifier = True

        def predict(self, X):
            return np.column_stack(
                [np.clip(0.2 + 0.4 * X[:, 0] + 0.5 * (1 - X[:, 1]), 0, 1), np.zeros(len(X))]
            )

    engine = _MIAMIEngine(mode="mask", verbose=False)
    engine.predictor = Predictor()
    engine.scorer = MaskScorer()
    engine.codebooks = {0: np.array([1.0]), 1: np.array([1.0])}
    engine.candidates = MixedGenerator(0, 1, "all", random_state=42)
    engine.candidates.X_full = np.array([[1.0, 1.0]])
    engine.candidates.feature_available = {0: np.array([0]), 1: np.array([0])}
    original = np.array([[0.0, 0.0]])
    corrected = engine.correct_subset(original, np.array([[True, True]]), columns_to_skip={1})
    np.testing.assert_array_equal(corrected, original)
    assert engine.predictor.predict(corrected)[0, 0] <= 0.7


def test_donor_zero_is_valid_for_a_new_row():
    generator = MixedGenerator(0, 3, "all", random_state=42)
    generator.X_full = np.array([[7.0, 9.0]])
    generator.feature_available = {0: np.array([0]), 1: np.array([0])}
    result = generator.generate(np.array([[0.0, 3.0]]), np.array([[True, False]]), {})
    assert result[0, 0] == 7
    np.testing.assert_array_equal(result[:, 1], 3)


def test_integer_mutation_budget_and_skip_gate():
    generator = CodebookGenerator(100, 1, random_state=42)
    result = generator.generate(
        np.zeros((1, 4)),
        np.ones((1, 4), dtype=bool),
        {i: np.array([1.0]) for i in range(4)},
        {0, 1, 2},
    )
    np.testing.assert_array_equal(result[:, :3], 0)
    assert result[:, 3].sum() > 0
