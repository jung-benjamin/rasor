#! /usr/bin/env python3
"""Tests for the genetic algorithm mutations."""

import numpy as np
import pytest

from rasor import mutations


@pytest.fixture
def gene_pool():
    """Alphabet as gene pool for mutations."""
    uppercase = [chr(i) for i in range(65, 65 + 26)]
    lowercase = [chr(i) for i in range(97, 97 + 26)]
    return uppercase + lowercase


@pytest.fixture
def fittest(gene_pool):
    """Five fittest members, each of length 5."""
    return [gene_pool[i * 5:(i + 1) * 5] for i in range(5)]


@pytest.fixture
def rng():
    """Random number generator with fixed seed."""
    return np.random.default_rng(seed=12345)


@pytest.fixture
def mutation(rng, gene_pool):
    """Return a mutation factory function."""
    freq = 0.5
    pop_size = 100
    gene_len = 5
    factory_kwargs = {
        'gene_pool': gene_pool,
        'population_size': pop_size,
        'frequency': freq,
        'gene_length': gene_len,
        'rng': rng
    }

    def factory(method):
        return mutations.MutationFactory().get_mutation(method=method,
                                                        **factory_kwargs)

    return factory


def test_cross_over(fittest, mutation):
    """Test for the CrossOver mutation."""
    cross_over = mutation('cross_over')
    crossed = cross_over(fittest)
    assert all([len(cr) == 5 for cr in crossed])


def test_gene_swap(fittest, mutation):
    """Test for the GeneSwap mutation"""
    gene_swap = mutation('gene_swap')
    swapped = gene_swap(fittest)
    assert all([len(sw) == 5 for sw in swapped])


def test_deletion(fittest, mutation):
    """Test for the Deletion mutation."""
    deletion = mutation('deletion')
    deletion.set_min_len(2)
    deleted = deletion(fittest)
    assert all([len(de) == 4 for de in deleted])


def test_addition(fittest, mutation):
    """Test for the Addition mutation."""
    addition = mutation('addition')
    added = addition(fittest)
    assert all([len(ad) == 6 for ad in added])
