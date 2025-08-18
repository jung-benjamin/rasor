#! /usr/bin/env python3
"""Manage test input values for evaluating the models."""

from copy import deepcopy

import numpy as np

from .sampling import SamplerFactory


def get_test_point_matrix(tp):
    """Create a TestPointMatrix"""
    if isinstance(tp, (np.ndarray, list)):
        return np.array(tp)
    elif isinstance(tp, dict):
        assert np.array(tp["limits"]).ndim == 2, "TestPointMatrix must be 2D"
        sampler = SamplerFactory().get_sampler(**tp)
        return sampler()
    else:
        raise TypeError(
            f"Expected np.ndarray or dict, got {type(tp)} instead.")


def get_test_point_multi_set(tp):
    """Create a TestPointMultiSet."""
    if isinstance(tp, (np.ndarray, list)):
        return np.array(tp)
    elif isinstance(tp, dict):
        assert np.array(
            tp['limits']).ndim == 3, "TestPointMultiSet requires 3D limits"
        kwarg_dict_cp = deepcopy(tp)
        points = []
        for limits in kwarg_dict_cp.pop('limits'):
            matrix = get_test_point_matrix({"limits": limits, **kwarg_dict_cp})
            points.append(matrix)
        return np.array(points)
    else:
        raise TypeError(
            f"Expected np.ndarray or dict, got {type(tp)} instead.")
