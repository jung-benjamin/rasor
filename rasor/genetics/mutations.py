#! /usr/bin/env python3
"""Mutate ratiolists for the genetic algorithm."""

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
        crossed = [
            self.rng.choice(fittest_pool, size=self.gene_length,
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
