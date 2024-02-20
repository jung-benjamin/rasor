#! /usr/bin/env python3
"""Surrogate models for isotopic ratios."""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp2d


class Surrogate:

    GRID_SIZE = (25, 25)

    def __init__(self, x, y):
        """Set the data.
        
        Parameters
        ----------
        x: np.ndarray
            X values of the interpolation data. Shape is (num, dim).
        y: np.ndarray
            Y values of the interpolation data Shape is (num,)
        """
        self.x = x
        self.y = y
        if not self.y.shape == self.GRID_SIZE:
            self._reshape_y()
        self._fill_nan()
        self._set_interpolator()

    @classmethod
    def ratio_from_isotopes(cls, x, y, r):
        """Create surrogate of a ratio given nuclide data."""
        i, j = r.split('/')
        return cls(x, y[i] / y[j])

    def _reshape_y(self):
        """Reshape the y data points."""
        self.y = self.y.reshape(self.GRID_SIZE, order='C')

    def _fill_nan(self, num=1e308):
        """Replace NaN values."""
        self.y = np.nan_to_num(self.y, nan=num)

    def _get_x_space(self):
        """Return regular grid axes."""
        dims = len(self.GRID_SIZE)
        reshi = self.x.reshape(*(*self.GRID_SIZE, dims))
        spaces = []
        for i in range(dims):
            sl = [slice(None)] * (dims + 1)
            sl[-1] = i
            sl[(dims - 1 - i)] = 0
            space = reshi[tuple(sl)]
            spaces.append(space.copy(order='C'))
        return spaces

    def _set_interpolator(self):
        """Create the Regular Grid Interpolator."""
        self.interpolator = RegularGridInterpolator(tuple(self._get_x_space()),
                                                    self.y)

    def __call__(self, *args):
        """Predict y values with interpolator"""
        return self.interpolator(args)


class LegacySurrogate(Surrogate):
    """Use the deprecated 'interp2d' interpolator."""

    def _reshape_y(self):
        """Reshape the y data points."""
        self.y = self.y.reshape(self.GRID_SIZE, order='F')

    def _set_interpolator(self):
        """Create the interpolator."""
        self.interpolator = interp2d(*self._get_x_space(), self.y)

    def __call__(self, *args):
        """Predict y values with interpolator"""
        return self.interpolator(*args)
