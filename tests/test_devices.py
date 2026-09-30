"""Explicit CPU/CUDA routing and rejection of silent CPU fallback."""

import json
import os
import warnings

import numpy as np
import pytest
from xgboost import XGBRegressor, build_info

from MiamiImputer import MIAMIImputer
from MiamiImputer.discriminator import _check_requested_device


def actual_device(model):
    return json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]


@pytest.mark.parametrize("device", ["gpu", "cudafoo", "cuda:-1", "cuda:", "cuda:0:1", 0])
def test_invalid_device_is_rejected(device):
    with pytest.raises(ValueError, match="device must be"):
        MIAMIImputer(device=device).fit(np.ones((4, 2)))


def test_cuda_request_cannot_silently_fall_back_to_cpu():
    class CPUBooster:
        def get_booster(self):
            return self

        def save_config(self):
            return json.dumps({"learner": {"generic_param": {"device": "cpu"}}})

    with pytest.raises(RuntimeError, match="XGBoost used 'cpu'"):
        _check_requested_device(CPUBooster(), "cuda:0")
    assert _check_requested_device(CPUBooster(), "cpu") == "cpu"


@pytest.fixture(scope="module")
def cuda_device():
    try:
        if not build_info().get("USE_CUDA", False):
            raise RuntimeError("XGBoost was built without CUDA")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = XGBRegressor(n_estimators=1, device="cuda:0", tree_method="hist")
            probe.fit(np.arange(24).reshape(12, 2), np.arange(12))
        if not actual_device(probe).startswith("cuda"):
            raise RuntimeError("No usable CUDA GPU; XGBoost fell back to CPU")
    except Exception as exc:
        if os.environ.get("MIAMI_REQUIRE_CUDA") == "1":
            pytest.fail(f"CUDA is required for this run: {exc}")
        pytest.skip(f"CUDA unavailable: {exc}")
    return "cuda:0"


def exercise_device(device, mode):
    rng = np.random.default_rng(19)
    X = rng.normal(size=(48, 4))
    X[rng.random(X.shape) < 0.25] = np.nan
    imputer = MIAMIImputer(
        mode=mode,
        device=device,
        max_iter=2,
        n_estimators=5,
        n_codebook=8,
        n_exact_copy=8,
        use_knee=False,
        column_auc_threshold=0 if mode == "mask" else None,
        r2_threshold=0,
    )
    completed = imputer.fit_transform(X)
    model = imputer._engine_.predictor._model
    assert actual_device(model) == device
    assert np.isfinite(completed).all()
    np.testing.assert_array_equal(completed[~np.isnan(X)], X[~np.isnan(X)])
    state = model.get_booster().save_raw()
    new = X[:7].copy()
    new[:, 1] += 0.123
    transformed = imputer.transform(new)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(transformed[~np.isnan(new)], new[~np.isnan(new)])
    np.testing.assert_allclose(imputer.transform(new[::-1])[::-1], transformed)
    assert state == model.get_booster().save_raw()


@pytest.mark.parametrize("mode", ["mask", "regressor"])
def test_cpu_execution(mode):
    exercise_device("cpu", mode)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["mask", "regressor"])
def test_real_cuda_execution(cuda_device, mode):
    exercise_device(cuda_device, mode)
