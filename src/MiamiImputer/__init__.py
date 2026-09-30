"""
Miami — Missingness Imputation via Adversarial Multichoice Iterations.

Public API
----------
>>> from MiamiImputer import MIAMIImputer
>>> imputer = MIAMIImputer(mode='mask')
>>> X_corrected = imputer.fit_transform(X_with_nans)

Two modes: ``mask`` (per-column classifier) and ``regressor``
(regressor on the missing fraction). See the paper for method details.
"""

from .config import DEFAULTS, FLAT_DEFAULTS
from .miami_imputer import MIAMIImputer

__all__ = ["MIAMIImputer", "DEFAULTS", "FLAT_DEFAULTS"]
__version__ = "1.2.0"
