#! /usr/bin/env python3
"""Filter nuclides and ratios based on simulation data."""

import re
from itertools import combinations, groupby

import pandas as pd
import radioactivedecay as rd

NUCLIDE_REGEX = re.compile(r'([A-Za-z]+)(-)(\d+)(\*|m)?')
NOBLE_GASES = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']


def isotope_regex(element):
    """Compile regex for selecting isotopes of an element."""
    return re.compile(f'{element}(-)?(\d)+(\*|m)?')


def get_ground_state(nuclide):
    """Get ground state of an excited nuclide."""
    n = NUCLIDE_REGEX.fullmatch(nuclide)
    return ''.join([n.group(1), n.group(2), n.group(3)])


def get_element(nuclide):
    """Determine element of a nuclide."""
    n = NUCLIDE_REGEX.fullmatch(nuclide)
    return n.group(1)


def is_excited(nuclide):
    """Determine if a nuclide is an excited state."""
    n = NUCLIDE_REGEX.fullmatch(nuclide)
    return bool(n.group(4))


def yield_ratio_options(nuclides):
    """Determine possible ratios from isotope list."""
    isolist = sorted(nuclides, key=lambda x: x.split('-')[0])
    for k, g in groupby(isolist, lambda x: x.split('-')[0]):
        for p, q in combinations(g, 2):
            yield f'{p}/{q}'


def fill_chain(nucl, chain):
    prog = rd.Nuclide(nucl).progeny()
    if prog == []:
        return
    chain |= set(prog)
    for p in prog:
        fill_chain(p, chain)


def get_decay_chain(nuclide):
    """Find set of nuclides in decay chain of a nuclide."""
    chain = set()
    try:
        _ = rd.Nuclide(nuclide)
    except ValueError:
        msg = f'Warning! {nuclide} not found in decay data.'
        print(msg)
    else:
        fill_chain(nucl=nuclide, chain=chain)
    return chain


class NuclideFilter:
    """Filter nuclides based on simulation data."""

    def __init__(self, data):
        """Create data"""
        self.data = data
        self.data.fillna(0, inplace=True)
        self.drop_noble_gases = True
        self.drop_oxygen = True
        self.drop_noble_gas_progeny = True
        self.actinide_reduction = 0.99
        self.excited_states_handler = 'keep'

    def __len__(self):
        return len(self.data.index)

    @classmethod
    def from_csv(cls, csvfiles):
        """Instantiate class from data in csv files."""
        if isinstance(csvfiles, list):
            dlist = [pd.read_csv(i, index_col=0) for i in csvfiles]
            dframe = pd.concat(dlist, axis=1)
        else:
            dframe = pd.read_csv(csvfiles, index_col=0)
        return cls(dframe)

    @property
    def nuclides(self):
        """Return the isotopes in the data."""
        return list(self.data.index)

    @property
    def drop_noble_gases(self):
        """Boolean flag for removing noble gases from data."""
        return self._drop_noble_gases

    @drop_noble_gases.setter
    def drop_noble_gases(self, b):
        """Set flag for removing noble gases from data."""
        self._drop_noble_gases = b

    @property
    def drop_noble_gas_progeny(self):
        """Boolean flag for removing noble gas decay products."""
        return self._drop_noble_gas_progeny

    @drop_noble_gas_progeny.setter
    def drop_noble_gas_progeny(self, b):
        """Set flag for dropping noble gas decay products."""
        self._drop_noble_gas_progeny = b

    @property
    def drop_oxygen(self):
        """Bolean flag for removing oxygen from data."""
        return self._drop_oxygen

    @drop_oxygen.setter
    def drop_oxygen(self, b):
        """Set flag for removing oxygen from data."""
        self._drop_oxygen = b

    @property
    def actinide_reduction(self):
        """Factor by which U and Pu are reduced."""
        return self._actinide_reduction

    @actinide_reduction.setter
    def actinide_reduction(self, f):
        """Set factor by which U and Pu are reduced."""
        self._actinide_reduction = f

    def _element_filter(self, element):
        """List all nuclides, dropping isotopes of element"""
        ele_regex = isotope_regex(element)
        nuclides = [e for e in self.nuclides if not ele_regex.fullmatch(e)]
        return nuclides

    def get_element_isotopes(self, elements):
        """List isotopes of elements in available in the data."""
        if isinstance(elements, (tuple, list, set)):
            elements = '({})'.format('|'.join(elements))
        ele_regex = isotope_regex(elements)
        isotopes = [n for n in self.nuclides if ele_regex.fullmatch(n)]
        return isotopes

    def filter_nuclides(self, nuclides):
        """Remove one or more nuclides from the data."""
        if isinstance(nuclides, str):
            nuclides = [nuclides]
        isotopes = [e for e in self.nuclides if e not in nuclides]
        self.data = self.data.loc[isotopes]

    def filter_elements(self, element):
        """Remove all isotopes of one or more elements from the data."""
        if isinstance(element, str):
            isotopes = self._element_filter(element)
            self.data = self.data.loc[isotopes]
        else:
            for ele in element:
                isotopes = self._element_filter(ele)
                self.data = self.data.loc[isotopes]

    def filter_decay_chain(self, nuclide):
        """Remove all nuclides in decay chain."""
        chain = set()
        if isinstance(nuclide, str):
            chain |= get_decay_chain(nuclide=nuclide)
        elif isinstance(nuclide, (tuple, list, set)):
            for n in nuclide:
                chain |= get_decay_chain(nuclide=n)
        self.filter_nuclides(nuclides=list(chain))

    def reduce_elements(self, element, factor=0.99):
        """Reduce quantities of an element by a factor."""
        if isinstance(element, str):
            reggy = isotope_regex(element)
            isotopes = [e for e in self.nuclides if reggy.fullmatch(e)]
            self.data.loc[isotopes] *= (1 - factor)
        else:
            if isinstance(factor, (float, int)):
                factor = [factor] * len(element)
            if len(factor) != len(element):
                msg = 'Factor should be float or same lengths as elements.'
                raise Exception(msg)
            for ele, fac in zip(element, factor):
                reggy = isotope_regex(ele)
                isotopes = [e for e in self.nuclides if reggy.fullmatch(e)]
                self.data.loc[isotopes] *= (1 - fac)

    def select_by_concentration(self, threshold=10e-9, fraction=1):
        """Select nuclides if the concentration is above the threshold."""
        conc = self.data / self.data.sum(axis=0)
        data_fraction = (conc >= threshold).sum(axis=1) / conc.shape[1]
        self.data = self.data[(data_fraction >= fraction)]

    def get_ratio_options(self):
        """Determine possible ratios from isotope list"""
        return list(yield_ratio_options(self.nuclides))

    def select_excited_nuclides(self):
        """Select activated nuclides in data."""
        exc = [n for n in self.nuclides if is_excited(n)]
        return exc

    def select_excited_pairs(self):
        """Select excited nuclides and their ground state counterparts."""
        exc = self.select_excited_nuclides()
        cp = [get_ground_state(n) for n in exc]
        return list(zip(cp, exc))

    def _add_excited_pairs(self):
        """Add quantity of excited states to the respective ground states."""
        for g, e in self.select_excited_pairs():
            try:
                self.data.loc[g] += self.data.loc[e]
            except KeyError:
                pass
            else:
                self.data.drop(e, axis=0, inplace=True)

    def _drop_excited_pairs(self):
        """Drop the excited nuclides and their ground state counterpart."""
        for g, e in self.select_excited_pairs():
            try:
                self.data.drop(g, axis=0, inplace=True)
            except KeyError:
                pass
            finally:
                self.data.drop(e, axis=0, inplace=True)

    @property
    def excited_states_handler(self):
        """Method for dealing with excited states.
        
        If method returns none, excited states are
        left untouched.
        """
        return self._excited_states_handler

    @excited_states_handler.setter
    def excited_states_handler(self, method):
        """Set processing method for nuclides in excited states."""
        _methods = {
            'drop': self._drop_excited_pairs,
            'add': self._add_excited_pairs,
            'keep': lambda: None
        }
        handler = _methods.get(method)
        if handler is None:
            raise ValueError(method)
        else:
            self._excited_states_handler = handler

    def reduce_actinides(self, factor=0.99):
        """Reduce U and Pu content by the separation efficiency."""
        self.reduce_elements(['U', 'Pu'], factor=factor)

    def filter(self, threshold, fraction=1):
        """Filter nuclides by concentration threshold."""
        noble_isotopes = self.get_element_isotopes(NOBLE_GASES)
        if self.drop_noble_gases:
            self.filter_elements(NOBLE_GASES)
        if self.drop_oxygen:
            self.filter_elements('O')
        self.excited_states_handler()
        self.reduce_actinides(self.actinide_reduction)
        self.select_by_concentration(threshold=threshold, fraction=fraction)
        if self.drop_noble_gas_progeny:
            self.filter_decay_chain(noble_isotopes)
