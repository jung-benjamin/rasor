#! /usr/bin/env python3
"""Run pre-selection of nuclides and ratios."""

import argparse
import json
import os
from pathlib import Path

from .filters import NuclideFilter


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
    excited_states = 'Select method for dealing with excited states.'
    parser.add_argument('--excited-states',
                        help=excited_states,
                        choices=['keep', 'add', 'drop'],
                        default='keep')
    outfile = 'File path for storing the output in Json format.'
    parser.add_argument('--outfile',
                        help=outfile,
                        type=Path,
                        default=Path('ratio_candidates.json'))
    write_data = 'File for storing the filtered data to csv.'
    parser.add_argument('--write-data', help=write_data, type=Path)
    return parser.parse_args()


def get_ratio_candidates(args):
    """Filter nuclides and determine possible ratios."""
    nuclide_filter = NuclideFilter.from_csv(args.datafile)
    nuclide_filter.drop_noble_gases = args.drop_noble
    nuclide_filter.drop_noble_gas_progeny = args.drop_noble_progeny
    nuclide_filter.drop_oxygen = args.drop_oxygen
    nuclide_filter.actinide_reduction = args.actinide_reduction
    nuclide_filter.excited_states_handler = args.excited_states
    nuclide_filter.filter(args.threshold, fraction=args.fraction)
    if args.write_data:
        nuclide_filter.to_csv(args.write_data)
    ratios = nuclide_filter.get_ratio_options()
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
    ratios = get_ratio_candidates(args)
    print(f'Selected {len(ratios)} candidate ratios.')
    write_output(ratios=ratios, fp=args.outfile)
    store_metadata(args=args)


if __name__ == '__main__':
    select_candidates()
