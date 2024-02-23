#! /usr/bin/env python3
"""Select sets of isotopic ratios."""

import argparse
import json
import logging
from json import JSONEncoder
from pathlib import Path

import numpy as np

from rasor.bruteforce import Aftermath, Minotaur
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
    return parser.parse_args()


ALGORITHMS = {'brute_force': Minotaur, 'genetic_evolution': GalapagosIslands}


def parse_input_file(infile):
    """Parse the input file."""
    with infile.open('r') as f:
        arg_dict = json.load(f)
    if isinstance(arg_dict['Algorithm'], str):
        algorithm = arg_dict['Algorithm']
    else:
        algorithm_kws = arg_dict['Algorithm']
        algorithm = list(algorithm_kws)[0]
    ratios = arg_dict['Problem']['ratios']
    print(ratios)
    print(f'Number of unique ratios: {len(set(ratios))}')
    limits = arg_dict['Problem']['limits']
    metric_kws = arg_dict['Metric']
    metric_kws.update({'limits': np.array(metric_kws.get('limits', limits))})
    test_points = arg_dict['Test_points']
    test_points.update({'limits': np.array(test_points.get('limits', limits))})
    if algorithm == 'genetic_evolution':
        x_data = np.load(arg_dict['Likelihood']['surrogates']['x_file'],
                         allow_pickle=True)
        y_data = np.load(arg_dict['Likelihood']['surrogates']['y_file'],
                         allow_pickle=True).item()
        data = (x_data, y_data)
        islands = GalapagosIslands(
            gene_pool=ratios,
            test_points=test_points,
            data=data,
            marginal_kwargs=metric_kws,
            uncertainty_kwargs=arg_dict['Likelihood']['uncertainty'],
            **algorithm_kws[algorithm])
        print(islands.init_length)
        print(islands.init_size)
        print(len(islands.gene_pool))
        return islands
    elif algorithm == 'brute_force':
        battering_ram = Minotaur(ratios=ratios,
                                 test_points=test_points,
                                 metric_params=arg_dict)
        return battering_ram


def select_ratios(algorithm, ncores=1):
    """Use the algorithm to select isotopic ratios."""
    if isinstance(algorithm, Minotaur):
        afterwards = Aftermath(*algorithm.fight(num_proc=ncores))
        selected = afterwards.find_best_ratios()
        metric_vals = dict(zip(afterwards.keys, afterwards.matrix))
    elif isinstance(algorithm, GalapagosIslands):
        selected, metric_vals = algorithm.speciate(num_proc=ncores)
    return selected, metric_vals


def store_results(selected, metric_vals, output_dir):
    """Store selected ratios and the metric to json files."""
    if not output_dir.exists():
        output_dir.mkdir()
    with open(output_dir / 'selected.json', 'w') as f:
        json.dump(selected, f, indent=True)
    with open(output_dir / 'metric_vals.json', 'w') as f:
        json.dump(metric_vals, f, indent=True, cls=NumpyArrayEncoder)


def run_ratio_selection(args):
    """Run isotope ratio selection."""
    algorithm = parse_input_file(args.infile)
    selected, metric = select_ratios(algorithm=algorithm, ncores=args.cores)
    store_results(selected=selected,
                  metric_vals=metric,
                  output_dir=args.output)


if __name__ == '__main__':
    args = argparser()
    logging.getLogger('CrossOver').setLevel(args.log_level)
    logging.getLogger('CrossOver').addHandler(logging.StreamHandler())
    logging.getLogger('Evolution').setLevel(args.log_level)
    logging.getLogger('Evolution').addHandler(logging.StreamHandler())
    run_ratio_selection(args)
