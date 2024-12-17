import argparse
import json
import os
from collections import Counter
from itertools import chain
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    metric_file = "JSON file containing the fitness of the genetic algorithm."
    parser.add_argument('metrics', type=Path, help=metric_file)
    return parser.parse_args()


def load_json(fp):
    with fp.open('r') as f:
        return json.load(f)


def load_metrics(fp):
    return np.array(load_json(fp))


def plot_fitness(fitness, save=""):
    """Plot fitness function values."""
    fig, ax = plt.subplots()
    steps = range(fitness.shape[0])
    for fit in fitness.T:
        ax.plot(steps, fit)
    if save:
        plt.savefig(save)


def plot(args):
    fitness = load_metrics(args.metrics)
    plot_path = args.metrics.with_name("fitness.png")
    plot_fitness(fitness, save=plot_path)


if __name__ == '__main__':
    args = argparser()
    plot(args)
