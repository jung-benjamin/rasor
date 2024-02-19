#! /usr/bin/env python3
"""Tests for the sampling module"""

import numpy as np
import pytest
from scipy.stats import qmc

from rasor import sampling


def test_sobol_sampler(bounds):
    """Test the Sobol sequence sample generator."""
    sobol = qmc.scale(qmc.Sobol(2, seed=1234).random_base2(10), *bounds)
    sampler = sampling.SobolSampler(limits=bounds, m=10, seed=1234)
    np.testing.assert_array_equal(sobol, sampler())
    sampler = sampling.SobolSampler(limits=bounds)
    np.testing.assert_array_equal(sobol, sampler(m=10, seed=1234))


def test_grid_sampler(bounds):
    """Test the grid sample generator."""
    edges = np.linspace(*bounds, 101)
    centers = edges[:-1, :] + np.diff(edges, axis=0) / 2
    grid = np.array(np.meshgrid(*centers.T, indexing='ij')).reshape(
        (2, 100**2)).T
    sampler = sampling.GridSampler(limits=bounds, bin_number=100)
    np.testing.assert_array_equal(grid, sampler())


def test_legacy_sampler(bounds):
    """Test the legacy grid sample generator"""
    centers = np.linspace(*bounds, 100).T
    grid = np.array(np.meshgrid(*centers)).reshape(2, 100**2).T
    sampler = sampling.LegacyGridSampler(limits=bounds, bin_number=100)
    np.testing.assert_array_equal(grid, sampler())
