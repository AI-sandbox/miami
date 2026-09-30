"""Run either Miami variant on CPU or an explicitly selected NVIDIA GPU."""

import argparse

import numpy as np

from MiamiImputer import MIAMIImputer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<ordinal>")
    parser.add_argument("--mode", choices=["mask", "regressor"], default="mask")
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 5))
    X[rng.random(X.shape) < 0.25] = np.nan
    imputer = MIAMIImputer(
        mode=args.mode,
        device=args.device,
        max_iter=3,
        n_estimators=30,
        n_codebook=32,
        n_exact_copy=32,
        random_state=42,
    )
    completed = imputer.fit_transform(X)
    assert np.isfinite(completed).all()
    np.testing.assert_array_equal(completed[~np.isnan(X)], X[~np.isnan(X)])
    print(f"{args.mode} on {args.device}: completed dataset {completed.shape}")


if __name__ == "__main__":
    main()
