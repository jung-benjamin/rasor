#! /usr/bin/env python3
"""Tests for the marginals module"""

from rasor import marginals


def test_marginals_init(bounds):
    """Test initialization of the marginals classes"""
    for m in ['legacy', 'grid', 'sobol']:
        marginals.MarginalsFactory().get_marginals(method=m, limits=bounds)
    assert True
