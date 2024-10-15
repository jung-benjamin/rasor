#! /usr/bin/env python3
"""Likelihood function models"""

from abc import ABC, abstractmethod

import numpy as np


class UncertaintyModel(ABC):
    """Model for relative uncertianty of isotopic ratios."""

    @abstractmethod
    def _uncertainty(self, ratio):
        """Calculate the relative uncertainty given a raito"""
        ...

    def __call__(self, ratio):
        """Calculate relative uncertainty and multiply with ratio."""
        return self._uncertainty(ratio) * ratio


class ConstantUncertainty(UncertaintyModel):

    def __init__(self, value=0.1):
        """Set value of constant relative uncertainty."""
        self.value = value

    def _uncertainty(self, ratio):
        return self.value


class UncertaintyFactory:

    _uncertainty = {
        'constant': ConstantUncertainty,
    }

    def get_model(self, model, **kwargs):
        """Create an instance of an uncertainty model."""
        model = self._uncertainty.get(model)
        if not model:
            raise ValueError(model)
        return model(**kwargs)


class LikelihoodModel(ABC):
    """Approximate the joint probabilitiy density function."""

    def __init__(self,
                 surrogates,
                 uncertainty_model,
                 test_point=None,
                 **kwargs):
        """Set the test point and the surrogate models."""
        self.surrogates = surrogates
        self.uncertainty = UncertaintyFactory().get_model(
            uncertainty_model, **kwargs)
        if test_point is not None:
            self.test_point = test_point
        else:
            self._test_point = test_point

    @property
    def test_point(self):
        """Test point on which likelihood is conditional."""
        return self._test_point

    @test_point.setter
    def test_point(self, tp):
        """Set a test point and calculate mu and sigma."""
        self._test_point = tp
        self.calc_mu()
        self.calc_sigma()

    def calc_mu(self):
        """Evaluate the surrogates on the test point."""
        self.mu = [sur(self._test_point) for sur in self.surrogates]

    def calc_sigma(self):
        self.sigma = [self.uncertainty(m) for m in self.mu]

    @abstractmethod
    def pdf(self, x):
        """Probability distribution function for variable"""
        ...

    def joint_pdf(self, x):
        """Joint probability distribution function
        
        Assumes all variables are independent and the joint probability
        is the product of each probability distribution function.
        """
        prob = 1
        for func, mu, sigma in zip(self.surrogates, self.mu, self.sigma):
            prob *= self.pdf(func(x), mu, sigma)
        return prob

    def __call__(self, x):
        """Calculate the joint probability for x
        
        Parameters
        ----------
        x : np.ndarray(float)
            Values for which to calculate the joint probability.
            Shape is (num, dim) where num is the number of points
            and"""
        return self.joint_pdf(x)


class LikelihoodLookUp(LikelihoodModel):
    """Approximate the using lookup tables."""

    def __init__(self, surrogates, uncertainty_model, test_point_mu, **kwargs):
        """Set the test point and the surrogate models.
        
        Surrogates and test_point mu must be lists of surrogate evaluations
        as calculated by the lookup tables.
        """
        self.surrogates = surrogates
        self.uncertainty = UncertaintyFactory().get_model(
            uncertainty_model, **kwargs)
        self.mu = np.array(list(test_point_mu.values()))
        self.calc_sigma()

    def joint_pdf(self, x):
        """Joint probability distribution function
        
        Assumes all variables are independent and the joint probability
        is the product of each probability distribution function.
        """
        prob = 1
        for func, mu, sigma in zip(self.surrogates.values(), self.mu,
                                   self.sigma):
            prob *= self.pdf(func, mu, sigma)
        return prob

    @property
    def test_point(self):
        """Test point on which likelihood is conditional."""
        return self._test_point

    @test_point.setter
    def test_point(self, tp):
        """Set a test point and calculate mu and sigma."""
        self._test_point = tp


class GaussianLikelihood(LikelihoodModel):
    """Likelihood function with normal distributions."""

    def pdf(self, x, mu, sigma):
        """Normal probability distribution function."""
        return np.exp(-(x - mu)**2 / (2 * (sigma**2)))


class GaussianLikelihoodLookUp(LikelihoodLookUp):
    """Likelihood function with normal distributions using lookup tables."""

    def pdf(self, x, mu, sigma):
        """Normal probability distribution function."""
        return np.exp(-(x - mu)**2 / (2 * (sigma**2)))
