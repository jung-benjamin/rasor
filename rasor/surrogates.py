#! /usr/bin/env python3
"""Surrogate models for isotopic ratios."""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp2d


def load_json(fp):
    """Load content of a json file."""
    with open(fp) as f:
        content = json.load(f)
    return content


def array_from_file(fp):
    """Load a numpy array from a file.
    
    Infers file type from suffix. Handles .npy and
    .json files.
    """
    fp = Path(fp)
    _loaders = {
        '.json': lambda x: np.array(load_json(x)),
        '.npy': lambda x: np.load(x, allow_pickle=True)
    }
    loader = _loaders.get(fp.suffix)
    if loader is None:
        ValueError(f'Invalid file ending: {fp.suffix}')
    return loader(fp)


def dict_from_file(fp):
    """Load a dictionary from a file.

    Infers file type from suffix. Handles .npy and
    .json files.
    """
    fp = Path(fp)
    _loaders = {
        '.json': lambda x: load_json(x),
        '.npy': lambda x: np.load(x, allow_pickle=True).item()
    }
    loader = _loaders.get(fp.suffix)
    if loader is None:
        ValueError(f'Invalid file ending: {fp.suffix}')
    return loader(fp)


class Surrogate:

    GRID_SIZE = (25, 25)

    def __init__(self, x, y, key=None, grid_size=None):
        """Set the data.
        
        Parameters
        ----------
        x: np.ndarray
            X values of the interpolation data. Shape is (num, dim).
        y: np.ndarray
            Y values of the interpolation data Shape is (num,)
        """
        self._set_x(x)
        self._set_y(y, key=key)
        if grid_size:
            self.set_grid_size(grid_size)
        if not self.y.shape == self.GRID_SIZE:
            self._reshape_y()
        self._fill_nan()
        self._set_interpolator()

    def _set_x(self, x):
        """Set the x values."""
        if isinstance(x, (Path, str)):
            self.x = array_from_file(Path(x))
        elif isinstance(x, (list, np.ndarray)):
            self.x = np.array(x)
        else:
            TypeError('Invalid data type for x.')

    def _set_y(self, y, key=None):
        """Set the y values."""
        if isinstance(y, (Path, str)) and key is None:
            self.y = array_from_file(Path(y))
        elif isinstance(y, (Path, str)) and key:
            d = dict_from_file(y)
            self.y = np.array(d[key])
        elif isinstance(y, (list, np.ndarray)):
            self.y = np.array(y)
        elif key and isinstance(y, dict):
            self.y = np.array(y[key])
        else:
            TypeError('Invalid data type for y.')

    @classmethod
    def ratio_from_isotopes(cls, x, y, r, grid_size=None):
        """Create surrogate of a ratio given nuclide data."""
        i, j = r.split('/')
        return cls(x, y[i] / y[j], grid_size=grid_size)

    @classmethod
    def set_grid_size(cls, gs):
        """Change the grid size."""
        cls.GRID_SIZE = gs

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
            sl = [0] * (dims + 1)
            sl[-1] = i
            sl[i] = slice(None)
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


class SurrogateSoup:

    def __init__(self, x, y, names):
        self.models = {n: Surrogate(x, y[n]) for n in names}

    @classmethod
    def from_ratiolist(cls, x, y, ratios):
        """Create models given istope data and ratio IDs."""
        y_dict = {}
        for r in ratios:
            i, j = r.split('/')
            y_dict[r] = y[i] / y[j]
        return cls(x, y_dict, ratios)

    def __call__(self, *args, ratios='all'):
        """Evaluate models on parameter values."""
        if ratios == 'all':
            return {n: m(*args) for n, m in self.models.items()}
        else:
            return {r: self.models[r](*args) for r in ratios}
