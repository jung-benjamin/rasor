#! /usr/bin/env python3
"""Configure loggers across modules."""

from .evolution import Evolution
from .mutations import MutationFactory


def config_global_logging(**kwargs):
    """Configure all loggers in rasor."""
    with_logger = [Evolution, MutationFactory]
    for c in with_logger:
        c.config_logger(**kwargs)
