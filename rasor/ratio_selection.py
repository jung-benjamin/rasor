#! /usr/bin/env python3
"""Select sets of isotopic ratios."""

import argparse
import json
import logging
import time
from json import JSONEncoder
from pathlib import Path

import numpy as np
from mpi4py import MPI

from rasor import config_global_logging
from rasor.bruteforce import Aftermath, BabyMinotaur, Minotaur
from rasor.evolution import GalapagosIslands


class NumpyArrayEncoder(JSONEncoder):
    """JSONEncoder that supports numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    infile = 'Input file in json format'
    parser.add_argument('infile', help=infile, type=Path)
    cores = 'Number of parallel processes.'
    parser.add_argument('-c', '--cores', help=cores, default=1, type=int)
    output = 'Directory to store the results.'
    parser.add_argument('-o', '--output', help=output, type=Path)
    loglevel = 'Set global logging level.'
    parser.add_argument('-l', '--log-level', help=loglevel, default='WARNING')
    logfile = 'Set path to logfile.'
    parser.add_argument('-f', '--log-file', help=logfile, type=Path)
    return parser.parse_args()


ALGORITHMS = {
    'brute_force': Minotaur,
    'genetic_evolution': GalapagosIslands,
    'baby_brute': BabyMinotaur
}


def parse_input_file(infile):
    """Parse the input file."""
    with infile.open('r') as f:
        arg_dict = json.load(f)
    if isinstance(arg_dict['Algorithm'], str):
        algorithm = arg_dict['Algorithm']
        algorithm_kws = {}
    else:
        algorithm_kws = arg_dict['Algorithm']
        algorithm = list(algorithm_kws)[0]
    ratios = arg_dict['Problem']['ratios']
    logging.info(f'Number of unique ratios: {len(set(ratios))}')
    limits = arg_dict['Problem']['limits']
    marginal_kws = arg_dict['Marginals']
    marginal_kws.update(
        {'limits': np.array(marginal_kws.get('limits', limits))})
    test_points = arg_dict['Test_points']
    test_points.update({'limits': np.array(test_points.get('limits', limits))})
    if algorithm == 'genetic_evolution':
        islands = GalapagosIslands(
            gene_pool=ratios,
            test_points=test_points,
            data=arg_dict['Likelihood']['surrogates'],
            marginal_kwargs=marginal_kws,
            metric_kwargs=arg_dict['Metric'],
            uncertainty_kwargs=arg_dict['Likelihood']['uncertainty'],
            **algorithm_kws[algorithm])
        return islands
    elif algorithm == 'brute_force':
        battering_ram = Minotaur(
            ratios=ratios,
            test_points=test_points,
            metric_params=arg_dict,
            use_combined=algorithm_kws[algorithm].get('use_combined', False),
            use_lookup=algorithm_kws[algorithm].get('use_lookup', False))
        if algorithm_kws.get(algorithm):
            Minotaur.set_combo_length(algorithm_kws[algorithm]['combo_length'])
        return battering_ram
    elif algorithm == 'baby_brute':
        with open(algorithm_kws[algorithm].get('combination_file'), 'r') as f:
            combinations = json.load(f)
        baby = BabyMinotaur(combinations=combinations,
                            test_points=test_points,
                            metric_params=arg_dict)
        return baby


def select_ratios(algorithm, ncores=1, log_kwargs=None):
    """Use the algorithm to select isotopic ratios."""
    if isinstance(algorithm, Minotaur):
        afterwards = Aftermath(*algorithm.fight(num_proc=ncores))
        selected = afterwards.find_best_ratios()
        metric_vals = dict(zip(afterwards.keys, afterwards.matrix))
    elif isinstance(algorithm, GalapagosIslands):
        selected, metric_vals = algorithm.speciate(num_proc=ncores,
                                                   log_kwargs=log_kwargs)
    return selected, metric_vals


def store_results(selected, metric_vals, output_dir):
    """Store selected ratios and the metric to json files."""
    if not output_dir.exists():
        output_dir.mkdir()
    with open(output_dir / 'selected.json', 'w') as f:
        json.dump(selected, f, indent=True, cls=NumpyArrayEncoder)
    with open(output_dir / 'metric_vals.json', 'w') as f:
        json.dump(metric_vals, f, indent=True, cls=NumpyArrayEncoder)


def run_ratio_selection(args):
    """Run isotope ratio selection."""
    tick = time.perf_counter()
    algorithm = parse_input_file(args.infile)
    selected, metric = select_ratios(algorithm=algorithm,
                                     ncores=args.cores,
                                     log_kwargs={
                                         'loglevel': args.log_level,
                                         'logpath': args.log_file
                                     })
    if MPI.COMM_WORLD.Get_rank() == 0:
        store_results(selected=selected,
                      metric_vals=metric,
                      output_dir=args.output)
    tock = time.perf_counter()
    logging.info(f'Finished in {tock - tick} seconds.')


def config_logging(loglevel='INFO',
                   logpath=None,
                   formatstr='%(levelname)s:%(name)s:%(message)s'):
    """Configure root and module loggers."""
    config_global_logging(loglevel=loglevel,
                          logpath=logpath,
                          formatstr=formatstr)
    log = logging.getLogger()
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


if __name__ == '__main__':
    args = argparser()
    config_logging(loglevel=args.log_level, logpath=args.log_file)
    run_ratio_selection(args)
