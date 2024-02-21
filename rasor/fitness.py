#! /usr/bin/env python3
"""Fitness functions for the genetic algorithm"""

from ..marginals import MarginalsFactory
from ..posterior_uncertainty import PosteriorUncertaintyApproximator


def create_samples(method, limits, **kwargs):
    marge = MarginalsFactory().get_marginals(method=method,
                                             limits=limits,
                                             **kwargs)
    marge.create_samples()
    if method == 'grid':
        return (marge.samples, )
    elif method == 'sobol':
        return marge.samples, marge.idx


class MaximumUncertaintyFitness:
    """Evaluate maximum posterior uncertainty as fitness."""

    def __init__(self,
                 x,
                 y,
                 sample_limits,
                 test_point,
                 method='grid',
                 uncertainy_model='legacy',
                 uncertainty_kwargs=None,
                 **kwargs):
        self.x = x
        self.y = y
        self.sample_limits = sample_limits
        self.marginal_samples = create_samples(method=method,
                                               limits=sample_limits,
                                               **kwargs)
        self.method = method
        self.uncertainy_model = uncertainy_model
        self.uncertainty_kwargs = uncertainty_kwargs
        self.test_point = test_point

    def __call__(self, genes):
        approximator = PosteriorUncertaintyApproximator(
            ratios=genes,
            xgrid=self.x,
            ydict=self.y,
            sample_limits=self.sample_limits,
            marginal_samples=self.marginal_samples,
            method=self.method,
            uncertainty_model=self.uncertainy_model,
            uncertainty_kwargs=self.uncertainty_kwargs)
        pos_unc = approximator.approximate_uncertainty(self.test_point)
        return max(pos_unc)
