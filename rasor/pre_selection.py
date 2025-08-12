#! /usr/bin/env python3
"""Run pre-selection of nuclides and ratios."""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from . import filters

NUCLIDE_REGEX = re.compile(r'([A-Za-z]+)(-)?(\d+)_?(\*|m\d?)?')


class PathEncoder(json.JSONEncoder):
    """JSONEncoder that supports Path objects."""

    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def argparser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    datafile = 'Csv file with nuclide quantity data.'
    parser.add_argument('datafile', help=datafile, type=Path)
    threshold = 'Threshold value for nuclide concentrations.'
    parser.add_argument('-t',
                        '--threshold',
                        help=threshold,
                        default=10e-9,
                        type=float)
    data_fraction = 'Number of datapoints that need to be above threshold.'
    parser.add_argument('-f',
                        '--fraction',
                        help=data_fraction,
                        default=1,
                        type=float)
    keep_noble = 'Set flag to keep noble gases'
    parser.add_argument('--keep-noble',
                        help=keep_noble,
                        action='store_false',
                        dest='drop_noble')
    keep_noble_progeny = 'Set flag to keep noble gas decay products.'
    parser.add_argument('--keep-noble-progeny',
                        help=keep_noble_progeny,
                        action='store_false',
                        dest='drop_noble_progeny')
    keep_oxygen = 'Set flag to keep oxygen isotopes in data.'
    parser.add_argument('--keep-oxygen',
                        help=keep_oxygen,
                        action='store_false',
                        dest='drop_oxygen')
    act_reduction = 'Factor by which to reduce U and Pu.'
    parser.add_argument('--actinide-reduction',
                        help=act_reduction,
                        default=0.99,
                        type=float)
    drop_element = 'List of elements to be removed from the data.'
    parser.add_argument('--drop-element',
                        help=drop_element,
                        nargs='*',
                        default=[])
    drop_element_progeny = ('List of elements whose decay products are' +
                            ' to be removed. The elements themselves' +
                            ' are not removed by this option (and explicitly'
                            ' kept, even if they are in other decay chains.).')
    parser.add_argument('--drop-element-progeny',
                        help=drop_element_progeny,
                        nargs='*',
                        default=[])
    excited_states = 'Select method for dealing with excited states.'
    parser.add_argument('--excited-states',
                        help=excited_states,
                        choices=['keep', 'add', 'drop'],
                        default='keep')
    exclude_file = 'JSON file with a list of nuclides to exclude.'
    parser.add_argument('--exclude-file', help=exclude_file, type=Path)
    outfile = 'File path for storing the output in Json format.'
    parser.add_argument('--outfile',
                        help=outfile,
                        type=Path,
                        default=Path('ratio_candidates.json'))
    write_data = 'File for storing the filtered data to csv.'
    parser.add_argument('--write-data', help=write_data, type=Path)
    parser.add_argument("--v1",
                        help="Use old pre-selection method.",
                        action="store_true")
    return parser.parse_args()


def format_nuclide_id(nuclide_id):
    """Format nuclide ID to a standard form."""
    match = NUCLIDE_REGEX.match(nuclide_id)
    if not match:
        raise ValueError(f"Invalid nuclide ID format: {nuclide_id}")
    element, dash, mass_number, modifier = match.groups()
    match modifier:
        case "*" | "m" | "m1":
            excited = f"m"
        case "m2":
            excited = f"n"
        case _:
            excited = ""
    return f"{element}{mass_number}{excited}"


def get_ratio_candidates(args):
    """Filter nuclides and determine possible ratios."""
    nuclide_filter = filters.NuclideFilter.from_csv(args.datafile)
    if args.exclude_file:
        with args.exclude_file.open() as f:
            exclude = json.load(f)
        nuclide_filter.filter_nuclides(exclude)
    if args.drop_noble:
        nuclide_filter.drop_noble_gases()
    if args.drop_noble_progeny:
        nuclide_filter.drop_noble_gas_progeny()
    if args.drop_oxygen:
        nuclide_filter.drop_oxygen()
    nuclide_filter.actinide_reduction = args.actinide_reduction
    nuclide_filter.excited_states_handler = args.excited_states
    nuclide_filter.drop_progeny = args.drop_element_progeny
    nuclide_filter.drop_elements = args.drop_element
    nuclide_filter.filter(args.threshold, fraction=args.fraction)
    if args.write_data:
        nuclide_filter.revert_actinide_reduction()
        nuclide_filter.to_csv(args.write_data)
    ratios = nuclide_filter.get_ratio_options()
    return ratios


def get_ratio_candidates_v2(args):
    """Filter nuclides and determine possible ratios."""

    if args.write_data:
        print("Warning: --write-data option is not supported in v2 mode." +
              "\n Ignoring this flag.")
    if args.excited_states != 'keep':
        print("Warning: --excited-states option is not supported in v2 mode." +
              "\n Ignoring this flag.")

    if isinstance(args.datafile, list):
        dlist = [pd.read_csv(i, index_col=0) for i in args.datafile]
        data = pd.concat(dlist, axis=1)
    else:
        data = pd.read_csv(args.datafile, index_col=0)
    data.index.name = "nuclide"
    data.index = data.index.map(format_nuclide_id)

    element_threshold_filter = filters.ElementThresholdFilter(
        data, actinide_reduction=args.actinide_reduction)
    decay_progeny_filter = filters.DecayProgenyFilter(data)
    element_filter = filters.ElementFilter(data)

    # Some elements should be kept
    protected_elements = ["U", "Pu"]
    protected_nuclides = []
    for element in protected_elements:
        protected_nuclides.extend(element_filter.elements.get(element))

    nuclides = set(data.index)
    drop_nuclides = set(
        element_threshold_filter(args.threshold, percentile=args.fraction))
    if args.drop_noble:
        drop_nuclides |= set(element_filter(filters.NOBLE_GASES))
    if args.drop_oxygen:
        drop_nuclides |= set(element_filter("O"))
    if args.drop_noble_progeny:
        drop_nuclides |= set(decay_progeny_filter(filters.NOBLE_GASES))
    if args.drop_element_progeny:
        drop_nuclides |= set(decay_progeny_filter(args.drop_element_progeny))
    if args.drop_element:
        drop_nuclides |= set(element_filter(args.drop_element))
    if args.exclude_file:
        with args.exclude_file.open() as f:
            exclude = json.load(f)
        drop_nuclides |= set(exclude)

    drop_nuclides -= set(protected_nuclides)  # Keep U and Pu isotopes
    nuclides -= set(format_nuclide_id(n) for n in drop_nuclides)

    if not nuclides:
        raise ValueError('No nuclides left after filtering.')
    ratios = list(filters.yield_ratio_options(nuclides))
    return ratios


def write_output(ratios, fp):
    """Write ratio candidates to output file."""
    if not fp.exists():
        with fp.open('w') as f:
            json.dump(ratios, f, indent=True)
    else:
        raise FileExistsError(fp)


def path_relto_home(x):
    """Convert paths relative to home directory."""
    home = Path(os.environ['HOME'])
    if isinstance(x, Path):
        try:
            return x.absolute().relative_to(home)
        except ValueError:
            return x
    return x


def store_metadata(args):
    """Store pre-selection configuration parameters."""
    d = {n: path_relto_home(it) for n, it in vars(args).items()}
    meta_name = f'{args.outfile.stem}_meta.json'
    with args.outfile.with_name(meta_name).open('w') as f:
        json.dump(d, f, indent=True, cls=PathEncoder)


def select_candidates():
    """Pre-select isotope ratio candidates"""
    args = argparser()
    if args.v1:
        print(f"Using v1 pre-selection method.")
        ratios = get_ratio_candidates(args)
    else:
        print(f"Using v2 pre-selection method.")
        ratios = get_ratio_candidates_v2(args)
    print(f'Selected {len(ratios)} candidate ratios.')
    write_output(ratios=ratios, fp=args.outfile)
    store_metadata(args=args)


if __name__ == '__main__':
    select_candidates()
