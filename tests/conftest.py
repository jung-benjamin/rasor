#! /usr/bin/env python3
"""Configure the pytest unit tests"""

import json
from pathlib import Path

import numpy as np
import pytest


def get_ratio_pairs(ratios):
    return [[ratios[i], ratios[j]] for i in range(len(ratios))
            for j in range(i + 1, len(ratios))]


@pytest.fixture
def rootdir():
    return Path(__file__).resolve().parent


@pytest.fixture
def bounds():
    return np.array([[0.2, 1000], [0.7, 9000]])
