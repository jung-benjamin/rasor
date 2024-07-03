#! /usr/bin/env python3
"""Process the results of the ratio selection algorithm."""

import argparse
import json
from pathlib import Path

from rasor.bruteforce import LootCollector


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    infile = 'Input file with results of ratio selection algorithm.'
    parser.add_argument('infile', help=infile, type=Path)
    outfile = 'Output file path for storing the results.'
    parser.add_argument('outfile', help=outfile, type=Path)
    algorithm = 'Algorithm used for selection.'
    parser.add_argument('-a',
                        '--algorithm',
                        help=algorithm,
                        choices=['bruteforce'],
                        default='bruteforce')
    return parser.parse_args()


def reduce_brute_force_selection(selection_file):
    """Reduce the results of the brute force selection."""
    looter = LootCollector.from_json(selection_file)
    return looter.find_unique()


def run_post_processing(args):
    """Run the post processing."""
    if args.algorithm == 'bruteforce':
        unique_ratios = reduce_brute_force_selection(args.infile)
        with open(args.outfile.with_suffix('.json'), 'w') as f:
            json.dump(unique_ratios, f, indent=4)


if __name__ == '__main__':
    args = argparser()
    run_post_processing(args)
