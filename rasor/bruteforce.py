#! /usr/bin/env python3
"""Select isotopic ratios by evaluating all combinations."""

import json
from itertools import combinations
from multiprocessing import Pool

import numpy as np

from .metrics import MaxLikelihoodUncertainty
from .sampling import SamplerFactory


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

    def fill(self):
        """Evaluate the metric on each test_point."""
        self.matrix = np.empty(self.test_points.shape[0])
        for i, tp in enumerate(self.test_points):
            self.matrix[i] = self.metric(tp)

    @classmethod
    def from_dict(cls, d):
        """Construct class from parameters in a dictionary"""
        metric = MaxLikelihoodUncertainty.from_dict(d=d)
        return cls(metric=metric, test_points=d['Test_points'])


def get_metric(ratios, param_dict):
    param_dict['Problem'].update({'ratios': ratios})
    return MaxLikelihoodUncertainty.from_dict(param_dict)


def evaluate_solubility_matrix(ratios, test_points, metric_params):
    """Create and fill the solubility matrix"""
    metric = get_metric(ratios, metric_params)
    solu = SolubilityMatrix(metric=metric, test_points=test_points)
    solu.fill()
    return solu.matrix


class Minotaur:
    """Iterate through ratio combinations with brute force."""

    combo_length = 2

    def __init__(self, ratios, test_points, metric_params):
        self.ratios = ratios
        self.test_points = test_points
        self.metric_params = metric_params

    def combinations(self):
        """Iterator over all combinations of ratios."""
        return combinations(self.ratios, r=self.combo_length)

    @classmethod
    def set_combo_length(cls, length):
        """Set the length of the ratio combinations."""
        cls.combo_length = int(length)

    def _scan(self):
        matrices = {}
        for r in self.combinations():
            matrices[','.join(r)] = evaluate_solubility_matrix(
                ratios=r,
                test_points=self.test_points,
                metric_params=self.metric_params)
        return list(matrices.keys()), list(matrices.values())

    def _scan_multiproc(self, num_proc):
        args = {
            ','.join(c): (c, self.test_points, self.metric_params)
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

    def find_best_ratios(self):
        """Find best ratio set for each grid point."""
        min_idx = self.matrix.argmin(axis=0)
        return self.keys[sorted(set(min_idx))]


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
