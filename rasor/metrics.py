#! /usr/bin/env python3
"""Metrics for evaluating suitability of a list of ratios."""
import warnings
from abc import ABC, abstractmethod
from multiprocessing import Pool

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


class MarginalScore(ABC):
    """Metric score derived from marginal likelihoods."""

    def __init__(self, likelihood, marginals):
        """Instantiate the metric."""
        self.likelihood = likelihood
        self.marginals = marginals

    def _parametrize_marginals(self):
        """Parameterize the marginals.
        
        Calculate the mean and standard deviation of the marginal likelihoods.

        Returns
        -------
        mu : np.ndarray(float)
            Mean of the marginal likelihoods.
        sd : np.ndarray(float)
            Standard deviation of the marginal likelihoods.
        """
        self.marginals.calculate_marginals(self.likelihood)
        mu, sd = self.marginals.parameterize_marginals()
        return mu, sd

    @abstractmethod
    def _score(self):
        """Calculate the metric score."""
        pass

    def __call__(self):
        """Calculate the metric score."""
        return self._score()


class MarginalLikelihoodUncertainty(MarginalScore):
    """Approximate relative uncertainty of marginals of likelihood."""

    def _score(self):
        """Calculate the metric score."""
        mu, sd = self._parametrize_marginals()
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


class MarginalMeanDistance(MarginalScore):
    """Relative distance of marginal mean to the test point."""

    def _score(self):
        """Calculate the metric score."""
        mu, _ = self._parametrize_marginals()
        return 100 * np.abs(
            mu - self.likelihood.test_point) / self.likelihood.test_point


class MarginalSumNormedDistance(MarginalScore):
    """Relative distance of marginal mean to the test point.
    
    Uses the sum of the mean and the test point as the normalization.
    """

    def _score(self):
        """Calculate the metric score."""
        mu, _ = self._parametrize_marginals()
        return 2 * np.abs(mu - self.likelihood.test_point) / (
            mu + self.likelihood.test_point)


class CombinedDistanceUncertaintyScore(MarginalScore):
    """Combined score of distance and uncertainty."""

    def _score(self):
        """Calculate the metric score."""
        mu, sd = self._parametrize_marginals()
        distance = np.abs(
            mu - self.likelihood.test_point) / self.likelihood.test_point
        uncertainty = sd / mu
        return 100 * (distance + uncertainty)


class MarginalScoreFactory:
    """Factory for creating marginal score metrics."""

    score_dispatcher = {
        'uncertainty': MarginalLikelihoodUncertainty,
        'distance': MarginalMeanDistance,
        'sum_normed': MarginalSumNormedDistance,
        'combined': CombinedDistanceUncertaintyScore
    }

    def __init__(self, marginals):
        """Instantiate the factory."""
        self.marginals = marginals

    def get_score(self, score_type, likelihood, **kwargs):
        """Create a marginal score metric."""
        return self.score_dispatcher[score_type](likelihood=likelihood,
                                                 marginals=self.marginals,
                                                 **kwargs)


class MaxScore(Metric):
    """Use maximum of marginal scores of each parameter as the metric."""

    def __init__(self, likelihood, marginals, score_type, **kwargs):
        """Instantiate the metric."""
        self.score = MarginalScoreFactory(marginals).get_score(
            score_type=score_type, likelihood=likelihood, **kwargs)

    def _metric_function(self):
        return max(self.score())


class SquaredSumScore(Metric):
    """Use squared sum of marginal scores as the metric."""

    def __init__(self, likelihood, marginals, score_type, **kwargs):
        """Instantiate the metric."""
        self.score = MarginalScoreFactory(marginals).get_score(
            score_type=score_type, likelihood=likelihood, **kwargs)

    def _metric_function(self):
        return sum(s**2 for s in self.score())


def caller(f):
    return f()


class SingleMetric(Metric):
    """Calculate a metric for a single test point."""

    _metric_types = {
        'max': MaxScore,
        'sum': SquaredSumScore,
    }

    def __init__(self,
                 likelihood,
                 marginals,
                 metric_type='max',
                 score_type='uncertainty'):
        """Set the likelihoods and the metric type."""
        self.likelihood = likelihood
        self.marginals = marginals
        self.metric_func = self._metric_types[metric_type](
            likelihood=likelihood, marginals=marginals, score_type=score_type)

    def _metric_function(self):
        return self.metric_func()

    @classmethod
    def from_dict(cls, d):
        """Instantiate class from dictionary."""
        m_fac = MarginalsFactory()
        m_dict = d['Marginals'].copy()
        m_dict['limits'] = np.array(d['Metric'].get('limits',
                                                    d['Problem']['limits']))
        marginals = m_fac.get_marginals(**m_dict)
        l_dict = d['Likelihood'].copy()
        models = SurrogateCollection.from_ratiolist(
            **l_dict['surrogates'],
            ratios=d['Problem']['ratios'],
        )
        likelihood = GaussianLikelihood(surrogates=models.modellist(),
                                        **l_dict['uncertainty'])
        return cls(likelihood=likelihood, marginals=marginals, **d['Metric'])


class MultiMetric(Metric):

    _metric_types = {
        'max': MaxScore,
        'sum': SquaredSumScore,
    }

    def __init__(self,
                 likelihoods,
                 marginals,
                 num_proc=1,
                 metric_type='max',
                 score_type='uncertainty'):
        self.likelihoods = likelihoods
        self.marginals = marginals
        self.num_proc = num_proc
        self.metric_func = self._metric_types[metric_type]
        self.score_type = score_type

    def _metric_function(self):

        mlu = [
            self.metric_func(likelihood=ll,
                             marginals=self.marginals,
                             score_type=self.score_type)
            for ll in self.likelihoods
        ]
        if self.num_proc > 1:
            with Pool(self.num_proc) as p:
                # Distribute the metric function calls across processes
                metric_values = list(p.map(caller, mlu))
        else:
            metric_values = [mf() for mf in mlu]
        nan_loc = np.where(np.isnan(metric_values))[0]
        mean = np.nanmean(metric_values)
        if len(nan_loc) > (0.5 * len(metric_values)):
            msg = ("Warning: More than 50% of the metric values are NaN.\n" +
                   f"    The mean metric value is: {mean}")
            warnings.warn(msg, category=UserWarning)
        return mean, nan_loc
