#! /usr/bin/env python3
"""Simulate evolution with a genetic algorithm."""

import numpy as np

from .mutations import MutationFactory


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
        self.mutations = {
            m:
            MutationFactory().get_mutation(m,
                                           population_size=init_size,
                                           gene_length=init_length,
                                           rng=self.rng,
                                           gene_pool=self.gene_pool,
                                           frequency=self.mutation_fraction[m])
            for m in mutations
        }

    def _set_initial_population(self):
        """Create the inital population of genes."""
        self.population = self.draw_from_pool(number=self.init_size,
                                              length=self.init_length)

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
            new_population.extend(mutation(elites))

        diff = population_size - len(new_population)
        if diff > 0:
            new_population.extend(
                self.draw_from_pool(number=diff, length=self.init_length))
        self.population = new_population
        self.evaluate_fitness()

    def darwinism(self, max_iter=20):
        fitness_evo = []
        for i in range(max_iter):
            fitness_evo.append(self.best_fitness_vals())
            self.evolve()
        best = self.elitism(n=1)
        return best, fitness_evo
