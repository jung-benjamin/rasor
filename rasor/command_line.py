#! /usr/bin/env python3
"""Provide access to functions via scripts."""

from . import plot_genetic_evolution as pge
from . import post_processing as pp
from . import pre_selection as ps
from . import ratio_selection as rs


def run_ratio_selection():
    """Run isotope ratio selection algorithm."""
    args = rs.argparser()
    rs.config_logging(loglevel=args.log_level, logpath=args.log_file)
    rs.run_ratio_selection(args)


def run_pre_selection():
    """Run pre selection algorithm."""
    ps.select_candidates()


def run_post_processing():
    """Run post processing algorithm."""
    args = pp.argparser()
    pp.run_post_processing(args)


def plot_genetic_evolution():
    """Plot genetic algorithm evolution."""
    pge.plot(pge.argparser())
