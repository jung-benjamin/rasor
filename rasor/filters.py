#! /usr/bin/env python3
"""Filter nuclides and ratios based on simulation data."""

import re
from itertools import combinations, groupby

import pandas as pd
import radioactivedecay as rd

NUCLIDE_REGEX = re.compile(r'([A-Za-z]+)(-)?(\d+)_?(\*|m\d?)?')
NOBLE_GASES = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']


def isotope_regex(element):
    """Compile regex for selecting isotopes of an element."""
    return re.compile(element + r'(-)?(\d)+(\*|m)?')


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
        if p == 'SF':
            continue
        else:
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
        self._drop_progeny = []
        self._drop_elements = []
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

    def to_csv(self, fp, **kwargs):
        """Write the filtered data to a csv file."""
        self.data.to_csv(fp, **kwargs)

    @property
    def nuclides(self):
        """Return the isotopes in the data."""
        return list(self.data.index)

    # @property
    def drop_noble_gases(self):
        """Add noble gases to drop list."""
        self.drop_elements = NOBLE_GASES

    @property
    def drop_progeny(self):
        """Drop decay products of these elements."""
        return self._drop_progeny

    @drop_progeny.setter
    def drop_progeny(self, elements):
        """Add elements whose decay products are removed.
        
        The list of elements is sorted and duplicates are
        removed.
        """
        if isinstance(elements, str):
            self._drop_progeny.extend([elements])
        elif isinstance(elements, (list, tuple, set)):
            self._drop_progeny.extend(list(elements))
        else:
            raise ValueError(elements)
        self._drop_progeny = sorted(set(self._drop_progeny))

    # @property
    def drop_noble_gas_progeny(self):
        """Add noble gas decay products to drop list.
        
        Calling this method adds all noble gases to the list
        of elements whose decay products are removed.
        """
        self.drop_progeny = NOBLE_GASES

    @property
    def drop_elements(self):
        """List of elements that are removed from the data."""
        return self._drop_elements

    @drop_elements.setter
    def drop_elements(self, elements):
        """Set elements to be removed from the data."""
        if isinstance(elements, str):
            self._drop_elements.extend([elements])
        elif isinstance(elements, (list, tuple, set)):
            self._drop_elements.extend(list(elements))
        else:
            raise ValueError(elements)
        self._drop_elements = sorted(set(self._drop_elements))

    def drop_oxygen(self):
        """Add oxygen to the drop list."""
        self.drop_elements = 'O'

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
        """Remove all decay products of one or more nuclides.
        
        The nuclides themselves are not removed, even if they
        appear in the decay chain of other nuclides. If they
        should be removed, remove them separately.
        """
        chain = set()
        if isinstance(nuclide, str):
            chain |= get_decay_chain(nuclide=nuclide)
        elif isinstance(nuclide, (tuple, list, set)):
            for n in nuclide:
                chain |= get_decay_chain(nuclide=n)
            chain -= set(nuclide)
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

    def revert_element_reduction(self, element, factor=0.99):
        """Reduce quantities of an element by a factor."""
        if isinstance(element, str):
            reggy = isotope_regex(element)
            isotopes = [e for e in self.nuclides if reggy.fullmatch(e)]
            self.data.loc[isotopes] /= (1 - factor)
        else:
            if isinstance(factor, (float, int)):
                factor = [factor] * len(element)
            if len(factor) != len(element):
                msg = 'Factor should be float or same lengths as elements.'
                raise Exception(msg)
            for ele, fac in zip(element, factor):
                reggy = isotope_regex(ele)
                isotopes = [e for e in self.nuclides if reggy.fullmatch(e)]
                self.data.loc[isotopes] /= (1 - fac)

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

    def revert_actinide_reduction(self):
        """Revert reduction of U and Pu content by the separation efficiency."""
        self.revert_element_reduction(['U', 'Pu'],
                                      factor=self.actinide_reduction)

    def filter(self, threshold, fraction=1):
        """Filter nuclides by concentration threshold."""
        if self.drop_progeny:
            drop_isotopes = self.get_element_isotopes(self.drop_progeny)
        if self.drop_elements:
            self.filter_elements(self.drop_elements)
        self.excited_states_handler()
        self.reduce_actinides(self.actinide_reduction)
        self.select_by_concentration(threshold=threshold, fraction=fraction)
        if self.drop_progeny:
            self.filter_decay_chain(drop_isotopes)


class ElementFilter:

    major_actinides = ["U", "Pu"]

    def __init__(self,
                 data,
                 dilution_factor=60,
                 actinide_reduction=0.9999,
                 har_density=1.3):
        """Initialize ElementFilter with data.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing nuclide data with nuclide IDs as the
            index. Units of the data should be in g/cm3.
        dilution_factor : float, optional
            Factor by which the data is diluted during reprocessing.
            Default is 60.
        actinide_reduction : float, optional
            Factor by which U and Pu are reduced to account for
            reprocessing. Default is 0.9999 (99.99% reduction).
        har_density : float, optional
            Density of the HAR solution in g/cm3. Default is 1.3 g/cm3.
        """
        self.data = data.copy()
        self._sum_isotopes()
        self.dilution_factor = dilution_factor
        self.actinide_reduction = actinide_reduction
        self.har_density = har_density
        self._process()

    def _sum_isotopes(self):
        """Prepare the DataFrame for element filtering.
        
        This method first adapts the index and then sums over all
        isotopes of each element, resulting in a DataFrame with elements
        as the index and their total mass densities.
        """
        self.data.index.name = "nuclide"
        self.data.reset_index(inplace=True)
        self.data["element"] = self.data["nuclide"].apply(get_element)
        self.data.set_index(["element", "nuclide"], inplace=True)
        self.data = self.data.groupby("element").sum()

    def dilute(self):
        """Approximate dilution during reprocessing.

        Divides the data by the specified dilution factor.
        """
        self.data /= self.dilution_factor

    def reduce_actinides(self):
        """Reduce actinide concentrations to account for reprocessing."""
        self.data.loc[self.major_actinides] *= (1 - self.actinide_reduction)

    def calc_mass_fractions(self):
        """Convert mass densities to mass fractions.
        
        Uses a given density of the HAR solution to convert the diluted
        mass densities into mass fractions.
        """
        self.data /= self.har_density

    def _process(self):
        """Process the data through before filtering."""
        self.dilute()
        self.reduce_actinides()
        self.calc_mass_fractions()

    def filter(self, threshold=1e-9, percentile=0.25):
        """Filter elements based on a mass fraction threshold.
        
        Parameters
        ----------
        threshold : float, optional
            The minimum mass fraction for an element to be included in
            the filtered data. Default is 1e-9. (1 ppb)
        percentile : float, optional
            The percentile to use for filtering. Default is 0.25 (25th
            percentile).
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the elements with mass fractions
            above the specified threshold.
        """
        if (percentile * 100).is_integer():
            percent_str = f"{int(percentile * 100)}%"
        else:
            percent_str = f"{percentile * 100}%"
        return self.data[self.data.T.describe(
            percentiles=[percentile]).loc[percent_str].T > threshold]
