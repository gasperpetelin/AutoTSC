"""RDST shapelet features."""

import numpy as np
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform


class RDSTFloat64(RandomDilatedShapeletTransform):
    """RDST wrapper that casts input to float64 (numba requires it)."""

    def _fit(self, X, y=None):
        return super()._fit(np.asarray(X, dtype=np.float64), y)
