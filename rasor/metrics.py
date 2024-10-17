#! /usr/bin/env python3
"""Metrics for evaluating suitability of a list of ratios."""
from abc import ABC, abstractmethod

import numpy as np

from .likelihood import GaussianLikelihood
from .marginals import MarginalsFactory
from .surrogates import SurrogateCollection


class Metric(ABC):
    """Calculate an evaluation metric for a list of ratios."""

    def __init__(self):
        pass

    @abstractmethod
    def _metric_function(self, test_point):
        """A metric"""
        pass

    def __call__(self):
        """Calculate the metric on a single test point."""
        return self._metric_function()


class MarginalLikelihoodUncertainty:
    """Approximate relative uncertainty of marginals of likelihood."""

    def __init__(self, likelihood, marginals):
        """Instantiate the metric."""
        self.likelihood = likelihood
        self.marginals = marginals

    def approximate_uncertainty(self):
        """Approximate uncertainty of likelihood marginals"""
        self.marginals.calculate_marginals(self.likelihood)
        mu, sd = self.marginals.parameterize_marginals()
        return 100 * sd / mu

    @classmethod
    def from_dict(cls, d):
        """Instantiate class from dictionary."""
        m_fac = MarginalsFactory()
        m_dict = d['Metric'].copy()
        m_dict['limits'] = d['Metric'].get('limits', d['Problem']['limits'])
        marginals = m_fac.get_marginals(**m_dict)
        l_dict = d['Likelihood'].copy()
        models = SurrogateCollection.from_ratiolist(
            **l_dict['surrogates'],
            ratios=d['Problem']['ratios'],
        )
        likelihood = GaussianLikelihood(surrogates=models.modellist(),
                                        **l_dict['uncertainty'])
        return cls(likelihood=likelihood, marginals=marginals)


class MaxLikelihoodUncertainty(MarginalLikelihoodUncertainty, Metric):
    """Maximum of marginal likelihood uncertainty"""

    def __init__(self, likelihood, marginals):
        MarginalLikelihoodUncertainty.__init__(self,
                                               likelihood=likelihood,
                                               marginals=marginals)

    def _metric_function(self):
        """Calculate maximum of marginal likelihood uncertainty."""
        return max(self.approximate_uncertainty())


class SqSumLikelihoodUncertainty(MarginalLikelihoodUncertainty, Metric):
    """Squared sum of marginal likelihood uncertainties"""

    def __init__(self, likelihood, marginals):
        MarginalLikelihoodUncertainty.__init__(self,
                                               likelihood=likelihood,
                                               marginals=marginals)

    def _metric_function(self):
        """Calculate maximum of marginal likelihood uncertainty."""
        unc = self.approximate_uncertainty()
        return sum(u**2 for u in unc)
