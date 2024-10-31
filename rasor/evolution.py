#! /usr/bin/env python3
"""Simulate evolution with a genetic algorithm."""

import json
import logging
import multiprocessing as mp
from functools import reduce
from itertools import chain
from multiprocessing import Pool

import numpy as np

from .likelihood import GaussianLikelihood, GaussianLikelihoodLookUp
from .marginals import MarginalsFactory
from .metrics import MaxLikelihoodUncertainty, MultiMetric
from .mutations import MutationFactory
from .sampling import SamplerFactory
from .surrogates import FrozenSurrogateLookUp, SurrogateCollection


class Fitness:
    """Fitness function for genetic algorithms."""

    def __init__(self,
                 gene_pool,
                 data,
                 test_point,
                 uncertainty_kwargs,
                 marginal_kwargs,
                 num_proc=1):
        self.models = SurrogateCollection.from_ratiolist(**data,
                                                         ratios=gene_pool)
        self.test_point = test_point
        self.logger.info(f'Setting test point: {test_point}')
        self.uncertainty_kwargs = uncertainty_kwargs
        self.marginals = MarginalsFactory().get_marginals(**marginal_kwargs)
        self.marginals.create_samples()
        if len(test_point.shape) == 2:
            self.num_test_points = test_point.shape[0]
        elif len(test_point.shape) == 1:
            self.num_test_points = 1
        else:
            raise ValueError(f'Invalid test point shape: {test_point.shape}')

    @property
    def logger(self):
        """Get logger."""
        return logging.getLogger(self.__class__.__name__)

    @classmethod
    def config_logger(cls,
                      loglevel='INFO',
                      logpath=None,
                      formatstr='%(levelname)s:%(name)s:%(message)s'):
        """Configure the logger."""
        log = logging.getLogger(cls.__name__)
        log.propagate = False
        log.setLevel(getattr(logging, loglevel.upper()))
        log.handlers.clear()
        fmt = logging.Formatter(formatstr)
        sh = logging.StreamHandler()
        sh.setLevel(getattr(logging, loglevel.upper()))
        sh.setFormatter(fmt)
        log.addHandler(sh)
        if logpath:
            fh = logging.FileHandler(logpath)
            fh.setLevel(getattr(logging, loglevel.upper()))
            fh.setFormatter(fmt)
            log.addHandler(fh)

    def __call__(self, gene):
        """Evaluate the fitness function."""
        if self.num_test_points == 1:
            likelihood = GaussianLikelihood(
                surrogates=[self.models[g] for g in gene],
                test_point=self.test_point,
                **self.uncertainty_kwargs)
            metric = MaxLikelihoodUncertainty(likelihood=likelihood,
                                              marginals=self.marginals)
        else:
            likelihoods = [
                GaussianLikelihood(surrogates=[self.models[g] for g in gene],
                                   test_point=self.test_point[i],
                                   **self.uncertainty_kwargs)
                for i in range(self.num_test_points)
            ]
            metric = MultiMetric(likelihoods=likelihoods,
                                 marginals=self.marginals,
                                 num_proc=self.num_proc)
        return metric()


class FitnessLookup(Fitness):
    """Evalute fitness using lookup-table-based classes."""

    def __init__(self,
                 gene_pool,
                 data,
                 test_point,
                 uncertainty_kwargs,
                 marginal_kwargs,
                 num_proc=1):
        """Initialize the fitness function with lookkup tables.

        Creates to instances of FrozenSurrogateLookUp, one for the
        input samples and one for the test points.

        Parameters
        ----------
        gene_pool : list(str)
            List of isotopic ratios.
        data : dict
            Dictionary with x and y data for the interpolation-based
            surrogate models.
        test_points : np.ndarray
            Array of test points for the likelihood evaluation.
        uncertainty_kwargs : dict
            Keyword arguments for the uncertainty model.
        marginal_kwargs : dict
            Keyword arguments for the marginal likelihood approximation.
        """

        self.models = SurrogateCollection.from_ratiolist(**data,
                                                         ratios=gene_pool)
        self.logger.info(f'Setting test points: {test_point}')
        self.marginals = MarginalsFactory().get_marginals(**marginal_kwargs)
        self.marginals.create_samples()
        self.uncertainty_kwargs = uncertainty_kwargs
        self.surrogate_lookup = FrozenSurrogateLookUp.from_surrogate_collection(
            self.models, self.marginals.samples)
        self.test_point_lookup = FrozenSurrogateLookUp.from_surrogate_collection(
            self.models, test_point)
        if len(test_point.shape) == 2:
            self.num_test_points = test_point.shape[0]
        elif len(test_point.shape) == 1:
            self.num_test_points = 1
        else:
            raise ValueError(f'Invalid test point shape: {test_point.shape}')
        self.num_proc = num_proc

    def __call__(self, gene):
        """Evaluate the fitness function."""
        if self.num_test_points == 1:
            likelihood = GaussianLikelihoodLookUp(
                surrogates=self.surrogate_lookup.get_subset(gene),
                test_point_mu=self.test_point_lookup.get_subset(gene),
                **self.uncertainty_kwargs)
            metric = MaxLikelihoodUncertainty(likelihood=likelihood,
                                              marginals=self.marginals)
        else:
            likelihoods = [
                GaussianLikelihoodLookUp(
                    surrogates=self.surrogate_lookup.get_subset(gene),
                    test_point_mu=self.test_point_lookup.get_subset(
                        gene).select_idx(i),
                    **self.uncertainty_kwargs)
                for i in range(self.num_test_points)
            ]
            metric = MultiMetric(likelihoods=likelihoods,
                                 marginals=self.marginals,
                                 num_proc=self.num_proc)
        return metric()


class Evolution:
    """Genetic evolution overlord class."""

    def __init__(self,
                 gene_pool,
                 fitness_func,
                 mutations,
                 init_length=10,
                 init_size=100,
                 rng_seed=1234):
        """Set the gene pool and fitness function parameters."""
        self.gene_pool = gene_pool
        self.init_length = init_length
        self.init_size = init_size
        self.fitness_func = fitness_func
        self.rng = np.random.default_rng(seed=rng_seed)
        self._set_initial_population()
        self.evaluate_fitness()
        self._set_mutations(mutations=mutations)

    def _set_mutations(self, mutations):
        """Set the mutations and their frequency of occurence."""
        if isinstance(mutations, (list, tuple, set)):
            self.mutations = {
                m:
                MutationFactory().get_mutation(
                    m,
                    population_size=self.init_size,
                    gene_length=self.init_length,
                    rng=self.rng,
                    gene_pool=self.gene_pool,
                    frequency=self.mutation_fraction[m])
                for m in mutations
            }
        elif isinstance(mutations, dict):
            self.mutations = {
                m:
                MutationFactory().get_mutation(m,
                                               population_size=self.init_size,
                                               gene_length=self.init_length,
                                               rng=self.rng,
                                               gene_pool=self.gene_pool,
                                               frequency=freq)
                for m, freq in mutations.items()
            }

    @property
    def logger(self):
        """Get logger."""
        return logging.getLogger(self.__class__.__name__)

    @classmethod
    def config_logger(cls,
                      loglevel='INFO',
                      logpath=None,
                      formatstr='%(levelname)s:%(name)s:%(message)s'):
        """Configure the logger."""
        log = logging.getLogger(cls.__name__)
        log.propagate = False
        log.setLevel(getattr(logging, loglevel.upper()))
        log.handlers.clear()
        fmt = logging.Formatter(formatstr)
        sh = logging.StreamHandler()
        sh.setLevel(getattr(logging, loglevel.upper()))
        sh.setFormatter(fmt)
        log.addHandler(sh)
        if logpath:
            fh = logging.FileHandler(logpath)
            fh.setLevel(getattr(logging, loglevel.upper()))
            fh.setFormatter(fmt)
            log.addHandler(fh)

    def _set_initial_population(self):
        """Create the inital population of genes."""
        self.population = self.draw_from_pool(number=self.init_size,
                                              length=self.init_length)
        self.logger.debug(f'Initial population size: {len(self.population)}')

    def draw_from_pool(self, number, length):
        """Combine genes to form population members."""
        members = [
            self.rng.choice(self.gene_pool, size=length,
                            replace=False).tolist() for i in range(number)
        ]
        return members

    def evaluate_fitness(self):
        """Evaluate the fitness function for each gene in the population."""
        fitness = [self.fitness_func(gene) for gene in self.population]
        self.fitness = fitness

    def best_fitness_idx(self, n=5, cutoff=1e-4):
        """Select fittest members in the population.
        
        The cutoff is used to exclude extremely small values, as these
        are likely to result from numerical inaccuracies.
        """
        masked_fit = np.ma.array(self.fitness,
                                 mask=np.array(self.fitness) < cutoff)
        unmasked_idx = np.where(~masked_fit.mask)[0]
        unmasked_val = masked_fit.data[unmasked_idx]
        smallest_idx = unmasked_idx[np.argpartition(unmasked_val, n)[:n]]
        sorted_idx = smallest_idx[np.argsort(masked_fit[smallest_idx])]
        return sorted_idx

    def best_fitness_vals(self, n=5, cutoff=1e-4):
        fit = [
            self.fitness[i] for i in self.best_fitness_idx(n=n, cutoff=cutoff)
        ]
        return fit

    def elitism(self, n=5, cutoff=1e-4):
        """Select fittest members of the current generation."""
        fittest = [
            self.population[i]
            for i in self.best_fitness_idx(n=n, cutoff=cutoff)
        ]
        self.logger.debug(f'Lengths of elites: {[len(f) for f in fittest]}')
        return fittest

    mutation_fraction = {
        'elitism': 0.05,
        'cross_over': 0.05,
        'gene_swap': 0.2,
        'deletion': 0.2,
        'addition': 0.2
    }

    @classmethod
    def set_mutation_fraction(cls, d):
        """Change frequency of a mutation.
        
        Dict key must be on of the mutations, otherwise
        the fraction will be ignored.
        """
        cls.mutation_fraction.update(d)

    def evolve(self):
        """Build a new generation
        
        Select the fittest members and mutate to create the new
        population. The number of mutations of each type are specified
        as fractions of the initial population size. The difference
        between the specified fraction and the initial size is filled
        with novelty search.
        """
        population_size = self.init_size
        new_population = []
        elites = self.elitism(n=int(population_size *
                                    self.mutation_fraction['elitism']))
        new_population.extend(elites)
        for m, mutation in self.mutations.items():
            try:
                new_population.extend(mutation(elites))
            except Exception as e:
                self.logger.exception(f'Skipping mutation {m} due to error:')

        diff = population_size - len(new_population)
        if diff > 0:
            new_population.extend(
                self.draw_from_pool(number=diff, length=self.init_length))
        self.population = new_population
        self.evaluate_fitness()

    def darwinism(self, max_iter=20):
        fitness_evo = []
        for i in range(max_iter):
            self.logger.info(f'Generation {i}')
            fitness_evo.append(self.best_fitness_vals())
            try:
                self.evolve()
            except Exception as e:
                self.logger.exception(f'Iteration {i} skipped due to error:')
        best = self.elitism(n=1)
        return best, fitness_evo


def natural_selection(fitness_kws,
                      evolution_kws,
                      max_iter,
                      log_kwargs=None,
                      lookup=False):
    """Apply genetic selection"""
    logging.getLogger().debug(f'Log kwargs in natural selection: {log_kwargs}')
    if log_kwargs:
        if mp.current_process().name != 'MainProcess':
            fmt = log_kwargs.get('formatstr',
                                 '%(levelname)s:%(name)s:%(message)s')
            current_proc = mp.current_process().name
            fmt_split = fmt.split(':')
            fmt = ":".join([fmt_split[0]] + [current_proc] + fmt_split[-2:])
            log_kwargs.update({'formatstr': fmt})
        Evolution.config_logger(**log_kwargs)
        MutationFactory.config_logger(**log_kwargs)
        Fitness.config_logger(**log_kwargs)
    if lookup:
        fitness_func = FitnessLookup(**fitness_kws)
    else:
        fitness_func = Fitness(**fitness_kws)
    island = Evolution(**evolution_kws, fitness_func=fitness_func)
    return island.darwinism(max_iter=max_iter)


class GalapagosIslands:
    """Run genetic evolution with different starting conditions."""

    def __init__(self,
                 gene_pool,
                 mutations,
                 test_points,
                 data,
                 uncertainty_kwargs,
                 marginal_kwargs,
                 init_size=100,
                 init_length=10,
                 max_iter=20,
                 rng_seed=12345,
                 use_lookup=False,
                 use_combined=False):
        """Define the test space and set the metric."""
        test_point_dispatcher = {
            np.ndarray: self.set_test_points,
            dict: self.sample_test_points
        }
        t = type(test_points)
        test_point_dispatcher[t](test_points)
        self.gene_pool = gene_pool
        self.mutations = mutations
        self.init_size = init_size
        self.init_length = init_length
        self.rng_seed = rng_seed
        self.data = data
        self.uncertainty_kwargs = uncertainty_kwargs
        self.marginal_kwargs = marginal_kwargs
        self.max_iter = max_iter
        self.use_lookup = use_lookup
        self.use_combined = use_combined

    def set_test_points(self, tp):
        """Set the test point array."""
        self.test_points = tp

    def sample_test_points(self, kwarg_dict):
        """Create a sampler and create the test point array."""
        self.sampler = SamplerFactory().get_sampler(**kwarg_dict)
        self.set_test_points(self.sampler())

    @property
    def logger(self):
        """Get logger."""
        return logging.getLogger(self.__class__.__name__)

    @classmethod
    def config_logger(cls,
                      loglevel='INFO',
                      logpath=None,
                      formatstr='%(levelname)s:%(name)s:%(message)s'):
        """Configure the logger."""
        log = logging.getLogger(cls.__name__)
        log.propagate = False
        log.setLevel(getattr(logging, loglevel.upper()))
        log.handlers.clear()
        fmt = logging.Formatter(formatstr)
        sh = logging.StreamHandler()
        sh.setLevel(getattr(logging, loglevel.upper()))
        sh.setFormatter(fmt)
        log.addHandler(sh)
        if logpath:
            fh = logging.FileHandler(logpath)
            fh.setLevel(getattr(logging, loglevel.upper()))
            fh.setFormatter(fmt)
            log.addHandler(fh)

    def _combine(self, num_proc=1):
        """Run evolution with combined test points.
        
        The fitness fuction uses a metric that combines all
        test points into a single value.
        """
        fit_kws = {
            'gene_pool': self.gene_pool,
            'data': self.data,
            'marginal_kwargs': self.marginal_kwargs,
            'uncertainty_kwargs': self.uncertainty_kwargs,
            'test_point': self.test_points,
            'num_proc': num_proc
        }
        evo_kws = {
            'gene_pool': self.gene_pool,
            'mutations': self.mutations,
            'init_size': self.init_size,
            'init_length': self.init_length,
            'rng_seed': self.rng_seed
        }
        best, fitness = natural_selection(fitness_kws=fit_kws,
                                          evolution_kws=evo_kws,
                                          max_iter=self.max_iter,
                                          lookup=self.use_lookup)
        return best, fitness

    def _scan(self):
        """Iterate over the test points and run evolution"""
        best_genes, fitness_evolution = [], []
        for tp in self.test_points:
            fit_kws = {
                'gene_pool': self.gene_pool,
                'data': self.data,
                'marginal_kwargs': self.marginal_kwargs,
                'uncertainty_kwargs': self.uncertainty_kwargs,
                'test_point': tp
            }
            evo_kws = {
                'gene_pool': self.gene_pool,
                'mutations': self.mutations,
                'init_size': self.init_size,
                'init_length': self.init_length,
                'rng_seed': self.rng_seed
            }
            best, fitness = natural_selection(fitness_kws=fit_kws,
                                              evolution_kws=evo_kws,
                                              max_iter=self.max_iter,
                                              lookup=self.use_lookup)
            best_genes.append(best)
            fitness_evolution.append(fitness)
        return best_genes, fitness_evolution

    def _scan_multiproc(self, num_proc, log_kwargs):
        args = []
        for tp in self.test_points:
            args.append(({
                'gene_pool': self.gene_pool,
                'data': self.data,
                'marginal_kwargs': self.marginal_kwargs,
                'uncertainty_kwargs': self.uncertainty_kwargs,
                'test_point': tp
            }, {
                'gene_pool': self.gene_pool,
                'mutations': self.mutations,
                'init_size': self.init_size,
                'init_length': self.init_length,
                'rng_seed': self.rng_seed
            }, self.max_iter, log_kwargs, self.use_lookup))
        self.logger.debug(f'Length of multiprocessing args: {len(args)}')
        with Pool(processes=num_proc) as pool:
            output = pool.starmap(natural_selection, list(args))
        best_genes, fitness_evolution = [], []
        for b, f in output:
            best_genes.append(b)
            fitness_evolution.append(f)
        return best_genes, fitness_evolution

    def speciate(self, num_proc=1, log_kwargs=None):
        """Run evolution for each test point."""
        if self.use_combined:
            best_genes, fitness_evolution = self._combine(num_proc=num_proc)
        else:
            if num_proc > 1:
                best_genes, fitness_evolution = self._scan_multiproc(
                    num_proc, log_kwargs=log_kwargs)
            else:
                best_genes, fitness_evolution = self._scan()
        return best_genes, fitness_evolution


class GeneSequencer:
    """Find unique set of ratios from the evolution results."""

    def __init__(self, best_genes):
        self.best_genes = best_genes

    def find_unique(self):
        """Find unique set of genes."""
        unique_genes = set(chain(*chain.from_iterable(self.best_genes)))
        return sorted(unique_genes)

    @classmethod
    def from_json(cls, json_file):
        """Create a GeneSequencer from a json file."""
        with open(json_file, 'r') as f:
            best_genes = json.load(f)
        return cls(best_genes=best_genes)
