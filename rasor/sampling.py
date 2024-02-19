#! /usr/bin/env python3
"""Generate input parameter samples

Contains two options for constructing a set
of input parameter samples:
1. Construct a ND grid of samples of the parameter space.
2. Construct a Sobol sequence on the parameter space.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import qmc


def make_bin_edges(limits, bin_number):
    """Create an array of bin edges."""
    return np.linspace(*limits, bin_number + 1)


def calc_bin_centers(edges):
    """Calculate bin centers given the bin edges."""
    return edges[:-1, :] + np.diff(edges, axis=0) / 2


class Sampler(ABC):
    """Generate a set of input parameter samples."""

    def __init__(self, limits, **call_args):
        """Define the parameter space for the sampler.
        
        Parameters
        ----------
        limits: np.ndarray
            Array of shape (N, 2) indicating the lower and
            upper bounds of the parameter space in each
            dimension.
        call_args: keyword arguments
            Default arguments used by the __call__ method for
            generating samples. If none are specified, the
            default values in the method definition are used.
        """
        self.limits = limits
        self.dims = limits.shape[1]
        self.call_args = {}
        self.call_args.update(call_args)

    @abstractmethod
    def _generate_samples(self, **kwargs):
        """Create a set of samples from the parameter space."""
        ...

    def __call__(self, **kwargs):
        """Call the sampler to generate samples."""
        keywords = self.call_args.copy()
        keywords.update(kwargs)
        return self._generate_samples(**keywords)


class LegacyGridSampler(Sampler):
    """Generate input parameter samples on a regular grid.
    
    Uses the legacy method where the edges of the parameter
    space are included as grid points.
    """

    def _generate_samples(self, bin_number=100, two_dim=True):
        """Generate samples on a regular grid.
        
        Divides parameter space into equallly spaced points. The
        edges of the parameter space are included as grid points.
        """
        centers = np.linspace(*self.limits, bin_number).T
        grid = np.array(np.meshgrid(*centers))
        if two_dim:
            return grid.reshape((self.dims, bin_number**self.dims)).T
        else:
            return grid


class GridSampler(Sampler):
    """Generate input parameter samples on a regular grid."""

    def _generate_samples(self, bin_number=100, two_dim=True):
        """Generate samples on a regular grid.
        
        Divides the parameter space axes into equal-width bins and
        uses the bin centers to create a meshgrid that covers the
        space.
        """
        edges = make_bin_edges(limits=self.limits, bin_number=bin_number)
        centers = calc_bin_centers(edges=edges).T
        grid = np.array(np.meshgrid(*centers, indexing='ij'))
        if two_dim:
            return grid.reshape((self.dims, bin_number**self.dims)).T
        else:
            return grid


class SobolSampler(Sampler):
    """Generate input parameter samples with the Sobol sequence."""

    def _generate_samples(self, m=10, **kwargs):
        """Generate a Sobol sequence on the parameter space."""
        sampler = qmc.Sobol(self.dims, **kwargs)
        samples = sampler.random_base2(m=m)
        return qmc.scale(samples, *self.limits)


class SamplerFactory:
    """Factory class for input parameter samples"""

    _samplers = {
        'sobol': SobolSampler,
        'grid': GridSampler,
        'legacy': LegacyGridSampler
    }

    def get_sampler(self, method, **kwargs):
        """Create an instance of a sampler."""
        sampler = self._samplers.get(method)
        if not sampler:
            raise ValueError(method)
        return sampler(**kwargs)
