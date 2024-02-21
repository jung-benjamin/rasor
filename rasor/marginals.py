#! /usr/bin/env python3
"""Approximate marginal distributions."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm, qmc

from .sampling import SamplerFactory


def make_bin_edges(limits, bin_number):
    """Create an array of bin edges."""
    return np.linspace(*limits, bin_number + 1)


def calc_bin_centers(edges):
    """Calculate bin centers given the bin edges."""
    return edges[:-1, :] + np.diff(edges, axis=0) / 2


class Marginals(ABC):
    """Base class for approximating marginal distributions."""

    def __init__(self, limits, bin_number=100, **kwargs):
        """Specify the parameter space."""
        self.limits = limits
        self.dims = limits.shape[1]
        self.bin_number = bin_number
        self._make_space()
        self._set_sampler(**kwargs)
        self._samples = None

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, s):
        self._samples = s

    @abstractmethod
    def _set_sampler(self, **kwargs):
        """Set instance of sampler."""
        ...

    def _make_space(self):
        """Create points on the parameter space axes."""
        bins = make_bin_edges(limits=self.limits, bin_number=self.bin_number)
        space = calc_bin_centers(bins)
        self.bins = bins
        self.space = space

    def _create_samples(self, **kwargs):
        """Create a samples of the parameter space.
        
        The samples are created with a meshgrid structure and
        reshaped into a 2D array dimension (dims, num^dims).
        """
        return self.sampler(**kwargs)

    @abstractmethod
    def _set_samples(self, *args):
        """Set externally created samples"""
        ...

    def create_samples(self, **kwargs):
        self.samples = self._create_samples(**kwargs)

    def set_samples(self, *args):
        self._set_samples(*args)

    @abstractmethod
    def _marginalize(self, pdf):
        """Marginalize the probability distribution function."""
        ...

    def calculate_marginals(self, pdf):
        """Approximate marginal distributions of pdf in each dimension."""
        if self.samples is None:
            self.create_samples()
        marginals = self._marginalize(pdf)
        marginals = marginals.T / (marginals *
                                   np.diff(self.bins, axis=0).T).sum(axis=1)
        self.marginals = marginals.T
        return self.marginals

    def parameterize_marginals(self):
        """Calculate mean and sd of marginal distributions."""
        mean, st_dev = [], []
        for x, y in zip(self.space.T, self.marginals):
            m = np.dot(x, y) / y.sum()
            sd = np.sqrt(np.dot((x - m)**2, y / y.sum()))
            mean.append(m)
            st_dev.append(sd)
        return np.array(mean), np.array(st_dev)


class GridMarginals(Marginals):
    """Approximate marginals with grid sampling."""

    def _set_sampler(self, **kwargs):
        """Set grid sampler instance."""
        self.sampler = SamplerFactory().get_sampler(method='grid',
                                                    limits=self.limits,
                                                    bin_number=self.bin_number,
                                                    **kwargs)

    def _set_samples(self, samples):
        self.samples = samples

    def _marginalize(self, pdf):
        """Approximate marginal distributions of pdf in each dimension."""
        vals = pdf(self.samples).reshape([self.bin_number] * self.dims, )
        marginals = np.empty((self.dims, self.bin_number))
        for i in range(self.dims):
            sum_ax = tuple(j for j in range(self.dims) if j != i)
            marginals[i, :] = vals.sum(axis=sum_ax)
        return marginals


class LegacyMarginals(GridMarginals):
    """Approximate marginals with legacy grid
    
    The legacy grid used a different method for defining grid points
    (bin centers) and edges. It does not produce the same grid as the
    Sobol sampler.
    """

    def _set_sampler(self, **kwargs):
        """Set legacy grid sampler instance."""
        self.sampler = SamplerFactory().get_sampler(method='legacy',
                                                    limits=self.limits,
                                                    bin_number=self.bin_number,
                                                    **kwargs)

    def _make_space(self):
        """Set bin edges and centers of the parameter space."""
        bins = make_bin_edges(limits=self.limits,
                              bin_number=self.bin_number - 1)
        widths = np.diff(bins, axis=0)[0]
        bin_edges = np.ones((bins.shape[0] + 1, bins.shape[1]))
        bin_edges[:-1, :] = bins - widths / 2
        bin_edges[-1, :] = bins[-1, :] + widths / 2
        self.bins = bin_edges
        self.space = bins


class SobolMarginals(Marginals):
    """Approximate marginals with the Sobol sequence."""

    def _set_sampler(self, **kwargs):
        """Set the Sobol sampler instance."""
        self.sampler = SamplerFactory().get_sampler(method='sobol',
                                                    limits=self.limits,
                                                    **kwargs)

    def _calc_bin_indices(self):
        """Calculate the sample indices associated with each bin."""
        idx = []
        bin_num = self.bin_number
        for i, dim in enumerate(self.samples.T):
            bin_arg = np.digitize(dim, self.bins[:, i])
            idx.append(
                [np.where(bin_arg == (j + 1))[0] for j in range(bin_num)])
        return idx

    def _set_samples(self, samples, indices):
        self.samples = samples
        self.idx = indices

    def create_samples(self, **kwargs):
        self.samples = self._create_samples(**kwargs)
        self.idx = self._calc_bin_indices()

    def _marginalize(self, pdf):
        """Approximate marginal distributions of pdf in each dimension."""
        vals = pdf(self.samples)
        marginals = np.empty((self.dims, self.bin_number))
        for i in range(self.dims):
            marginals[i, :] = np.array(
                [vals[idx].sum() for idx in self.idx[i]])
        return marginals


class MarginalsFactory:
    """A factory class for marginals calculators."""

    _marginals = {
        'sobol': SobolMarginals,
        'grid': GridMarginals,
        'legacy': LegacyMarginals,
    }

    def get_marginals(self, method, **kwargs):
        """Create and instance of a marginals calculator."""
        marginals = self._marginals.get(method)
        if not marginals:
            raise ValueError(method)
        return marginals(**kwargs)
