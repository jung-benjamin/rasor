#! /usr/bin/env python3
"""Configure loggers across modules."""

import logging

from .evolution import Evolution, GalapagosIslands
from .mutations import MutationFactory


def config_global_logging(**kwargs):
    """Configure all loggers in rasor."""
    with_logger = [Evolution, MutationFactory, GalapagosIslands]
    for c in with_logger:
        c.config_logger(**kwargs)


def configure_logger(cls,
                     loglevel='INFO',
                     logpath=None,
                     formatstr='%(levelname)s:%(name)s:%(message)s'):
    """Configure the logger."""
    if isinstance(cls, str):
        log = logging.getLogger(cls)
    else:
        log = logging.getLogger(cls.__name__)
    log.propagate = False
    log.setLevel(getattr(logging, loglevel.upper()))
    log.handlers.clear()
    fmt = logging.Formatter(formatstr)
    if logpath:
        fh = logging.FileHandler(logpath)
        fh.setLevel(getattr(logging, loglevel.upper()))
        fh.setFormatter(fmt)
        log.addHandler(fh)
    else:
        sh = logging.StreamHandler()
        sh.setLevel(getattr(logging, loglevel.upper()))
        sh.setFormatter(fmt)
        log.addHandler(sh)
