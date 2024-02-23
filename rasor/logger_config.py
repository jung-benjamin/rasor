#! /usr/bin/env python3
"""Configure loggers across modules."""

from .evolution import Evolution, GalapagosIslands
from .mutations import MutationFactory


def config_global_logging(**kwargs):
    """Configure all loggers in rasor."""
    with_logger = [Evolution, MutationFactory, GalapagosIslands]
    for c in with_logger:
        c.config_logger(**kwargs)
