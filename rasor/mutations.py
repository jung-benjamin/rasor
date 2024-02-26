#! /usr/bin/env python3
"""Mutate ratiolists for the genetic algorithm."""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy


def pool_genes(population):
    pool = []
    for p in population:
        pool.extend(p)
    return sorted(set(pool))


class Mutation(ABC):

    def __init__(self, population_size, frequency, gene_length, gene_pool,
                 rng):
        """Set size of overall population and frequency of mutation."""
        self.population_size = population_size
        self.frequency = frequency
        self.gene_length = gene_length
        self.gene_pool = gene_pool
        self.rng = rng

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

    @abstractmethod
    def _mutate(self, genes):
        ...

    def __call__(self, genes):
        """Mutate each gene in the list."""
        return self._mutate(genes)


class CrossOver(Mutation):

    def _mutate(self, genes):
        n = int(self.population_size * self.frequency)
        fittest_pool = pool_genes(genes)
        self.logger.debug(f'Size of fittest pool: {len(fittest_pool)}')
        self.logger.debug(f'Length of genes: {self.gene_length}')
        crossed = [
            self.rng.choice(fittest_pool,
                            size=min(self.gene_length, len(fittest_pool)),
                            replace=False).tolist() for i in range(n)
        ]
        return crossed


class GeneSwap(Mutation):

    def _mutate(self, genes):
        n = int(self.population_size * self.frequency / len(genes))
        fittest_pool = pool_genes(genes)
        rest_pool = sorted(set(self.gene_pool) - set(fittest_pool))
        swapped = []
        for f in deepcopy(genes):
            self.logger.debug(f'Length of gene: {len(f)}')
            _ = f.pop(self.rng.integers(len(f)))
            add = self.rng.choice(rest_pool, size=n, replace=False)
            for a in add:
                swapped.append(f + [a])
        return swapped


class Addition(Mutation):
    max_len = 20

    @classmethod
    def set_max_len(cls, l):
        cls.max_len = l

    def _mutate(self, genes):
        n = int(self.population_size * self.frequency / len(genes))
        fittest_pool = pool_genes(genes)
        rest_pool = sorted(set(self.gene_pool) - set(fittest_pool))
        elongated = []
        for f in deepcopy(genes):
            if len(f) < self.max_len:
                add = self.rng.choice(rest_pool, size=n, replace=False)
                for a in add:
                    elongated.append(f + [a])
            else:
                continue
        return elongated


class Deletion(Mutation):
    min_len = 5

    @classmethod
    def set_min_len(cls, l):
        cls.min_len = l

    def _mutate(self, genes):
        n = int(self.population_size * self.frequency / len(genes))
        shortened = []
        for f in deepcopy(genes):
            if len(f) > self.min_len:
                for i in range(n):
                    _ = f.pop(self.rng.integers(len(f)))
                    shortened.append(f)
            else:
                continue
        return shortened


class MutationFactory:
    """Factory class for genetic algorithm mutations."""

    _mutations = {
        'cross_over': CrossOver,
        'gene_swap': GeneSwap,
        'addition': Addition,
        'deletion': Deletion
    }

    def get_mutation(self, method, **kwargs):
        """Create instance of genetic algorithm mutation."""
        mutation = self._mutations.get(method)
        if not mutation:
            raise ValueError(mutation)
        return mutation(**kwargs)

    @classmethod
    def config_logger(cls, **kwargs):
        """Configure logger of all mutation classes."""
        for m, c in cls._mutations.items():
            c.config_logger(**kwargs)
