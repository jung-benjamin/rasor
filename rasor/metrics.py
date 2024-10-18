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


def caller(f):
    return f()


class MultiMetric(Metric):

    def __init__(self, likelihoods, marginals, num_proc=1):
        self.likelihoods = likelihoods
        self.marginals = marginals
        self.num_proc = num_proc

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
                MaxLikelihoodUncertainty(likelihood=ll,
                                         marginals=self.marginals)()
                for ll in chunk
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
                    MaxLikelihoodUncertainty(likelihood=ll,
                                             marginals=self.marginals)
                    for ll in self.likelihoods
                ]
                # args = [(ll, self.marginals) for ll in self.likelihoods]
                return np.mean(p.map(caller, mlu))
        else:
            return np.mean([
                MaxLikelihoodUncertainty(likelihood=ll,
                                         marginals=self.marginals)()
                for ll in self.likelihoods
            ])
