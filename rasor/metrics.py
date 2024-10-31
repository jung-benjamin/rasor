#! /usr/bin/env python3
"""Metrics for evaluating suitability of a list of ratios."""
from abc import ABC, abstractmethod
from multiprocessing import Pool

import numpy as np
from mpi4py import MPI

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

    def approximate_uncertainty(self):
        """Approximate uncertainty of likelihood marginals"""
        self.marginals.calculate_marginals(self.likelihood)
        mu, sd = self._parametrize_marginals()
        return 100 * sd / mu

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
        self.likelihoods = likelihood
        self.marginals = marginals
        self.metric_func = self._metric_types[metric_type](
            likelihood=likelihood, marginals=marginals, score_type=score_type)

    def _metric_function(self):
        return self.metric_func()


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

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if size > 1:
            if rank == 0:
                # split the list of likelihoods in size parts
                # and send each part to a different process
                chunks = np.array_split(self.likelihoods, size)
            else:
                chunks = None
            # scatter the chunks to all processes
            chunk = comm.scatter(chunks, root=0)
            mlu = [
                self.metric_func(likelihood=ll,
                                 marginals=self.marginals,
                                 score_type=self.score_type)() for ll in chunk
            ]
            # gather the results from all processes
            mlu = comm.gather(mlu, root=0)
            if rank == 0:
                mlu_all = []
                for m in mlu:
                    mlu_all.extend(m)
                metr = np.mean(mlu_all)
            else:
                metr = None
            metr = comm.bcast(metr, root=0)
            return metr
        elif self.num_proc > 1:
            with Pool(self.num_proc) as p:
                mlu = [
                    self.metric_func(likelihood=ll,
                                     marginals=self.marginals,
                                     score_type=self.score_type)
                    for ll in self.likelihoods
                ]
                # args = [(ll, self.marginals) for ll in self.likelihoods]
                return np.mean(p.map(caller, mlu))
        else:
            return np.mean([
                self.metric_func(likelihood=ll,
                                 marginals=self.marginals,
                                 score_type=self.score_type)()
                for ll in self.likelihoods
            ])
