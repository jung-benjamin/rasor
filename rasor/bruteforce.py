#! /usr/bin/env python3
"""Select isotopic ratios by evaluating all combinations."""

import json
import warnings
from copy import deepcopy
from itertools import chain, combinations
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
                self.metric.likelihood.test_point = self.test_points[i]
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
        return list(combinations(self.ratios, r=self.combo_length))

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


class MultiMinotaur:
    """Run multiple minotaur fights.
    
    Should only be used with use_combined=False. Otherwise the models
    are averaged after the metric is averaged."""

    def __init__(self, **kwargs):
        surrogates = kwargs["metric_params"]["Likelihood"].pop("surrogates")
        assert isinstance(surrogates, (list, tuple))
        self.minotaurs = []
        for sur in surrogates:
            kwarg_cp = deepcopy(kwargs)
            kwarg_cp["metric_params"]["Likelihood"]["surrogates"] = sur
            self.minotaurs.append(Minotaur(**kwarg_cp))

    def fight(self, num_proc=1):
        """Calculate solubility matrix of each minotaur."""
        keys, solubility = [], []
        for m in self.minotaurs:
            k, s = m.fight(num_proc=num_proc)
            keys.append(k)
            solubility.append(s)

        # Average over the minotaur dimension
        # In the future it may be better to return the matrices
        # without averaging, as matrix could be useful by itself.
        # For now, this would probably break the processing code.
        solubility = np.mean(np.array(solubility), axis=0)

        # keys are all ordered the same way (preserved by starmap)
        return keys[0], solubility


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

    @classmethod
    def from_json(cls, fp):
        """Construct class from parameters in a JSON dictionary"""
        with open(fp, 'r') as f:
            d = json.load(f)
        return cls(keys=list(d.keys()), solubility=list(d.values()))

    def find_best_ratios(self, depth=1, target_number=None):
        """Find best ratio set for each grid point.
        
        Finds the ratio set with the lowest metric value(s) for
        each grid point in the solubility matrix. Filters duplicate
        sets of ratios and sorts the results alphabetically (not by
        metric value!!!).

        Parameters
        ----------
        depth : int, None
            Number of best ratios to return for each grid point in the
            solubility matrix. If None, only the target_number limits
            the number of ratios returned.
        target_number: int, None
            Desired length of the final ratio list. If None, only the
            depth limits the number of ratios returned.

        Returns
        -------
        np.ndarray
            Array of ratio sets with the lowest metric values
        """
        sort_idx = self.matrix.argsort(axis=0)[:depth]
        if target_number is None:
            ratios = [p.split(",") for p in self.keys[sort_idx].flatten()]
            return sorted(set(chain(*ratios)))
        else:
            selected = set()
            for i, idx in enumerate(sort_idx):
                if len(idx.shape) == 0:
                    ratios = self.keys[idx].split(",")
                    if len(selected) < target_number:
                        selected |= set(ratios)
                else:
                    vals = np.array(
                        [self.matrix[col, j] for j, col in enumerate(idx.T)])
                    for k in vals.argsort():
                        ratios = self.keys[idx[k]].split(",")
                        if len(selected) < target_number:
                            selected |= set(ratios)
            return sorted(selected)


class MultiAftermath(Aftermath):
    """Combine results of multiple Minotaur fights.
    
    Do not use for the results of the MultiMinotaur class!
    Those results are already averaged and can be handled
    with the regular aftermath.
    """

    def __init__(self, key_lists, solubility):
        """Set solubility matrix and keys for each axis."""
        # Ensure that key list all contain the same keys
        assert all(set(keys) == set(key_lists[0]) for keys in key_lists)

        # Sort all key lists and solubility in the same
        # order as the first key list
        key_order = {k: i for i, k in enumerate(key_lists[0])}
        sorted_key_lists = []
        sorted_solubility = []

        for keys, sol in zip(key_lists, solubility):
            idx = [key_order[k] for k in keys]
            sorted_keys = [keys[i] for i in np.argsort(idx)]
            sorted_sol = np.array(sol)[np.argsort(idx)]
            sorted_key_lists.append(sorted_keys)
            sorted_solubility.append(sorted_sol)

        self.keys = np.array(sorted_key_lists[0])
        self.matrix = np.mean(np.array(sorted_solubility), axis=0)

    @classmethod
    def from_json(cls, *fp):
        """Construct class from data in json files."""
        # If a single list is passed, assume it contains a list of files
        if len(fp) == 1 and isinstance(fp[0], list):
            fp = fp.pop(0)
        key_lists, solubility_matrices = [], []
        for f in fp:
            with open(f, 'r') as file:
                d = json.load(file)
            key_lists.append(list(d.keys()))
            solubility_matrices.append(list(d.values()))
        return cls(key_lists=key_lists, solubility=solubility_matrices)


class LootCollector:
    """Find the unique ratios among the selection of best ratio sets."""

    def __init__(self, ratio_keys):
        msg = "LootCollector is deprecated and will be removed in a future version."
        warnings.warn(msg, DeprecationWarning)
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
