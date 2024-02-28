#! /usr/bin/env python3
"""Provide access to functions via scripts."""

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
