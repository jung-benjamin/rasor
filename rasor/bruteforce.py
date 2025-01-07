#! /usr/bin/env python3
"""Select isotopic ratios by evaluating all combinations."""

import json
from itertools import combinations
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from .likelihood import GaussianLikelihoodLookUp
from .marginals import MarginalsFactory
from .metrics import SingleMetric
from .sampling import SamplerFactory
from .surrogates import FrozenSurrogateLookUp, SurrogateCollection


class SolubilityMatrix:
    """Calculate metric on a grid of test points."""

    def __init__(self, metric, test_points):
        """Define the test space and set the metric."""
        test_point_dispatcher = {
            np.ndarray: self.set_test_points,
            dict: self.sample_test_points
        }
        self.metric = metric
        t = type(test_points)
        test_point_dispatcher[t](test_points)

    def set_test_points(self, tp):
        """Set the test point array."""
        self.test_points = tp

    def sample_test_points(self, kwarg_dict):
        """Create a sampler and create the test point array."""
        self.sampler = SamplerFactory().get_sampler(**kwarg_dict)
        self.set_test_points(self.sampler())

    def fill(self, test_point_lookup=None):
        """Evaluate the metric on each test_point."""
        self.matrix = np.empty(self.test_points.shape[0])
        if test_point_lookup:
            for i in range(self.test_points.shape[0]):
                self.metric.likelihood.mu = test_point_lookup.select_idx(i)
                self.metric.likelihood.calc_sigma()
                self.matrix[i] = self.metric()
        else:
            for i, tp in enumerate(self.test_points):
                self.metric.likelihood.test_point = tp
                self.matrix[i] = self.metric()

    @classmethod
    def from_dict(cls, d):
        """Construct class from parameters in a dictionary"""
        metric = SingleMetric.from_dict(d=d)
        return cls(metric=metric, test_points=d['Test_points'])


def get_metric(ratios, param_dict):
    param_dict['Problem'].update({'ratios': ratios})
    return SingleMetric.from_dict(param_dict)


def evaluate_solubility_matrix(ratios,
                               test_points,
                               metric_params,
                               use_combined=False,
                               surrogate_lookup=None,
                               test_point_lookup=None,
                               marginals=None):
    """Create and fill the solubility matrix"""
    if surrogate_lookup:
        likelihood = GaussianLikelihoodLookUp(
            surrogates=surrogate_lookup,
            test_point_mu=test_point_lookup,
            test_point=test_points,
            **metric_params["Likelihood"]['uncertainty'])
        metric = SingleMetric(likelihood=likelihood,
                              marginals=marginals,
                              **metric_params['Metric'])
    else:
        metric = get_metric(ratios, metric_params)
    solu = SolubilityMatrix(metric=metric, test_points=test_points)
    if surrogate_lookup:
        solu.fill(test_point_lookup=test_point_lookup)
    else:
        solu.fill()
    if use_combined:
        return np.mean(solu.matrix)
    return solu.matrix


class Minotaur:
    """Iterate through ratio combinations with brute force."""

    combo_length = 2

    def __init__(self,
                 ratios,
                 test_points,
                 metric_params,
                 use_combined=False,
                 use_lookup=False):
        self.ratios = ratios
        test_point_dispatcher = {
            np.ndarray: self.set_test_points,
            dict: self.sample_test_points
        }
        t = type(test_points)
        test_point_dispatcher[t](test_points)
        self.metric_params = metric_params
        self.use_combined = use_combined
        self.use_lookup = use_lookup
        if self.use_lookup:
            print(f'Using lookup tables...')
            m_fac = MarginalsFactory()
            m_dict = self.metric_params['Marginals'].copy()
            m_dict['limits'] = np.array(self.metric_params['Metric'].get(
                'limits', self.metric_params['Problem']['limits']))
            self.marginals = m_fac.get_marginals(**m_dict)
            self.marginals.create_samples()
            self.models = SurrogateCollection.from_ratiolist(
                **self.metric_params["Likelihood"]["surrogates"],
                ratios=self.ratios)
            self.surrogate_lookup = FrozenSurrogateLookUp.from_surrogate_collection(
                self.models, self.marginals.samples)
            self.test_point_lookup = FrozenSurrogateLookUp.from_surrogate_collection(
                self.models, self.test_points)

    def set_test_points(self, tp):
        """Set the test point array."""
        self.test_points = tp

    def sample_test_points(self, kwarg_dict):
        """Create a sampler and create the test point array."""
        self.sampler = SamplerFactory().get_sampler(**kwarg_dict)
        self.set_test_points(self.sampler())

    def combinations(self):
        """Iterator over all combinations of ratios."""
        return combinations(self.ratios, r=self.combo_length)

    @classmethod
    def set_combo_length(cls, length):
        """Set the length of the ratio combinations."""
        cls.combo_length = int(length)

    def _scan(self):
        matrices = {}
        if self.use_lookup:
            for r in tqdm(self.combinations(), disable=None):
                matrices[','.join(r)] = evaluate_solubility_matrix(
                    ratios=r,
                    test_points=self.test_points,
                    metric_params=self.metric_params,
                    use_combined=self.use_combined,
                    surrogate_lookup=self.surrogate_lookup.get_subset(r),
                    test_point_lookup=self.test_point_lookup.get_subset(r),
                    marginals=self.marginals)
        else:
            for r in tqdm(self.combinations(), disable=None):
                matrices[','.join(r)] = evaluate_solubility_matrix(
                    ratios=r,
                    test_points=self.test_points,
                    metric_params=self.metric_params,
                    use_combined=self.use_combined)
        return list(matrices.keys()), list(matrices.values())

    def _scan_multiproc(self, num_proc):
        if self.use_lookup:
            args = {
                ','.join(c):
                (c, self.test_points, self.metric_params, self.use_combined,
                 self.surrogate_lookup.get_subset(c),
                 self.test_point_lookup.get_subset(c), self.marginals)
                for c in self.combinations()
            }
        else:
            args = {
                ','.join(c):
                (c, self.test_points, self.metric_params, self.use_combined)
                for c in self.combinations()
            }
        with Pool(processes=num_proc) as pool:
            matrices = pool.starmap(evaluate_solubility_matrix,
                                    list(args.values()))
        return list(args), list(matrices)

    def fight(self, num_proc=1):
        """Calculate solubility matrix for each ratio combination."""
        if num_proc > 1:
            keys, solubility = self._scan_multiproc(num_proc=num_proc)
        else:
            keys, solubility = self._scan()
        return keys, solubility


class BabyMinotaur(Minotaur):
    """Iterate over pre-generated list of combinations
    
    Baby Minotaur is not as strong as his father and needs help
    calculating the combinations of ratios.
    """

    def __init__(self, combinations, test_points, metric_params):
        self._combinations = combinations
        self.test_points = test_points
        self.metric_params = metric_params

    def combinations(self):
        return self._combinations

    @classmethod
    def set_combo_length(cls, length):
        raise NotImplementedError('Baby Minotaur cannot change combo length')


class Aftermath:
    """Analyze the aftermath of a minotaur fight."""

    def __init__(self, keys, solubility):
        """Set solubility matrix and keys for each axis."""
        self.keys = np.array(keys)
        self.matrix = np.array(solubility)

    def find_best_ratios(self, depth=1):
        """Find best ratio set for each grid point.
        
        Finds the ratio set with the lowest metric value(s) for
        each grid point in the solubility matrix. Filters duplicate
        sets of ratios and sorts the results alphabetically (not by
        metric value!!!).

        Parameters
        ----------
        depth : int
            Number of best ratios to return for each grid point in the
            solubility matrix.

        Returns
        -------
        np.ndarray
            Array of ratio sets with the lowest metric values
        """
        sort_idx = self.matrix.argsort(axis=0)[:depth]
        return self.keys[sorted(set(sort_idx.flatten()))]


class LootCollector:
    """Find the unique ratios among the selection of best ratio sets."""

    def __init__(self, ratio_keys):
        self.wreckage = ratio_keys

    def find_unique(self):
        """Reduce the list of ratio sets to a list of unique ratios."""
        combined = []
        for w in self.wreckage:
            combined.extend(w.split(','))
        return sorted(set(combined))

    @classmethod
    def from_json(cls, json_file):
        """Create an instance from a JSON file."""
        with open(json_file, 'r') as f:
            return cls(ratio_keys=json.load(f))
