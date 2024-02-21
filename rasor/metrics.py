#! /usr/bin/env python3
"""Metrics for evaluating suitability of a list of ratios."""
from abc import ABC, abstractmethod

import numpy as np

from .likelihood import GaussianLikelihood
from .marginals import MarginalsFactory
from .surrogates import Surrogate


class Metric(ABC):
    """Calculate an evaluation metric for a list of ratios."""

    def __init__(self):
        pass

    @abstractmethod
    def _metric_function(self, test_point):
        """A metric"""
        pass

    def __call__(self, test_point):
        """Calculate the metric on a single test point."""
        return self._metric_function(test_point)


class MarginalLikelihoodUncertainty:
    """Approximate relative uncertainty of marginals of likelihood."""

    def __init__(self, likelihood, marginals):
        """Instantiate the metric."""
        self.likelihood = likelihood
        self.marginals = marginals

    def approximate_uncertainty(self, test_point):
        """Approximate uncertainty of likelihood marginals"""
        self.likelihood.test_point = test_point
        self.marginals.calculate_marginals(self.likelihood)
        mu, sd = self.marginals.parametrize_marginals()
        return 100 * sd / mu

    @classmethod
    def from_dict(cls, d):
        """Instantiate class from dictionary."""
        m_fac = MarginalsFactory()
        m_dict = d['Metric'].copy()
        m_dict['limits'] = d['Metric'].get('limits', d['Problem']['limits'])
        marginals = m_fac.get_marginals(**m_dict)
        l_dict = d['Likelihood'].copy()
        x = np.load(d['Likelihood']['surrogates']['x_file'], allow_pickle=True)
        y = np.load(d['Likelihood']['surrogates']['y_file'], allow_pickle=True)
        surrogate = {
            r: Surrogate.ratio_from_isotopes(x=x, y=y, r=r)
            for r in d['Problem']['ratios']
        }
        likelihood = GaussianLikelihood(surrogates=surrogate,
                                        **l_dict['uncertainty'])
        return cls(likelihood=likelihood, marginals=marginals)


class MaxLikelihoodUncertainty(MarginalLikelihoodUncertainty, Metric):
    """Maximum of marginal likelihood uncertainty"""

    def __init__(self, likelihood, marginals):
        MarginalLikelihoodUncertainty.__init__(likelihood=likelihood,
                                               marginals=marginals)

    def _metric_function(self, test_point):
        """Calculate maximum of marginal likelihood uncertainty."""
        return max(self.approximate_uncertainty(test_point=test_point))


class SqSumLikelihoodUncertainty(MarginalLikelihoodUncertainty, Metric):
    """Squared sum of marginal likelihood uncertainties"""

    def __init__(self, likelihood, marginals):
        MarginalLikelihoodUncertainty.__init__(likelihood=likelihood,
                                               marginals=marginals)

    def _metric_function(self, test_point):
        """Calculate maximum of marginal likelihood uncertainty."""
        unc = self.approximate_uncertainty(test_point=test_point)
        return sum(u**2 for u in unc)
