#! /usr/bin/env python3
"""Filter nuclides and ratios based on simulation data."""

import importlib.resources as pkg_resources
import json
import logging
import re
from itertools import combinations, groupby

import numpy as np
import pandas as pd
import radioactivedecay as rd

NUCLIDE_REGEX = re.compile(r'([A-Za-z]+)(-)?(\d+)_?(\*|m\d?|n)?')
NOBLE_GASES = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']

with pkg_resources.path(__package__, 'atomic_numbers.json') as p:
    with open(p, 'r') as f:
        ATOMIC_NUMBERS = json.load(f)


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


def get_mass_number(nuclide):
    """Get mass number of a nuclide."""
    n = NUCLIDE_REGEX.fullmatch(nuclide)
    return int(n.group(3))


def is_excited(nuclide):
    """Determine if a nuclide is an excited state."""
    n = NUCLIDE_REGEX.fullmatch(nuclide)
    return bool(n.group(4))


def yield_ratio_options(nuclides):
    """Determine possible ratios from isotope list."""
    isolist = sorted(nuclides, key=get_element)
    for k, g in groupby(isolist, get_element):
        for p, q in combinations(g, 2):
            yield f'{p}/{q}'


def fill_chain(nucl, chain, threshold=np.inf):
    """Recursively fill decay chain of a nuclide.
    
    A threshold can be set to stop the recursion if the half-life
    of a nuclide is above the threshold (in years).
    
    Parameters
    ----------
    nucl : str
        The nuclide for which to find the decay chain.
    chain : set
        A set to store the nuclides in the decay chain.
    threshold : float, optional
        The half-life threshold in years to stop recursion. Default is
        1e6 years (1 million years).
    
    Returns
    -------
    None
    """
    nuc = rd.Nuclide(nucl)
    prog = nuc.progeny()
    if prog == []:
        return
    elif nuc.half_life("y") > threshold:
        return
    chain |= (set(prog) - {"SF"})
    for p in prog:
        if p == 'SF':
            continue
        else:
            fill_chain(p, chain)


def get_decay_chain(nuclide, threshold=np.inf):
    """Find set of nuclides in decay chain of a nuclide.
    
    Nuclides with a half-life above the threshold are not treated
    as stable.
    """
    chain = set()
    try:
        _ = rd.Nuclide(nuclide)
    except ValueError:
        msg = f'Warning! {nuclide} not found in decay data.'
        logging.getLogger("DecayChain").warning(msg)
    else:
        fill_chain(nucl=nuclide, chain=chain, threshold=threshold)
    return chain


def group_elements(nuclides):
    """Group elements"""
    return {
        g: list(k)
        for g, k in groupby(sorted(nuclides, key=get_element), key=get_element)
    }


class Filter:
    """Base class for filters."""

    @property
    def logger(self):
        """Get logger."""
        return logging.getLogger(self.__class__.__name__)

    def collect_nuclides(self, elements):
        """Collect nuclides of the specified elements."""
        nuclides = []
        if isinstance(elements, str):
            elements = [elements]
        for element in elements:
            isotopes = self.elements.get(element, [])
            if not isotopes:
                msg = f"Warning! No isotopes found for element {element}."
                self.logger.warning(msg)
                continue
            nuclides.extend(isotopes)
        return sorted(nuclides)

    def __call__(self, *args, **kwargs):
        """Call the filter with the specified elements."""
        filter = self.filter(*args, **kwargs)
        self.logger.debug(f"Filtering : {filter}")
        return filter


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


class ElementThresholdFilter(Filter):

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
        self.nuclides = list(data.index)
        self.elements = group_elements(self.nuclides)
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

    def compare_threshold(self, threshold=1e-9, percentile=0.25):
        """Filter elements based on a mass fraction threshold.
        
        Parameters
        ----------
        threshold : float, optional
            The minimum mass fraction for an element to be included in
            the filtered data. Default is 1e-9. (1 ppb)
        percentile : float, optional
            The percentile to use for filtering. Default is 0.25 (25th
            percentile). Is rounded to first decimal of percent value.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the elements with mass fractions
            above the specified threshold.
        """
        percentile = np.round(percentile, 3)
        if (percentile * 100).is_integer():
            percent_str = f"{int(percentile * 100)}%"
        else:
            percent_str = f"{percentile * 100}%"
        below = self.data[self.data.T.describe(
            percentiles=[percentile]).loc[percent_str].T < threshold]
        return below

    def filter(self, threshold=1e-9, percentile=0.25):
        """Filter elements based on a mass fraction threshold."""
        below = self.compare_threshold(threshold=threshold,
                                       percentile=percentile)
        return self.collect_nuclides(below.index)


class DecayProgenyFilter(Filter):
    """Filter to remove decay progeny of specified nuclides.
    
    If isotopes of elements that are removed of changed during
    reprocessing decay, this does not affect the isotopic ratios
    of this element. However, it does affect the ratios of the
    decay product. The effect-size depends on the half-life.
    """

    def __init__(self, data, half_life_threshold=np.inf):
        """Initialize the filter with a list of nuclides.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing nuclide data with nuclide IDs as the
            index. Units of the data should be in g/cm3.
        half_life_threshold : float, optional
            Half-life threshold in years. Nuclides with a half-life
            above this threshold will not be included in the decay
            chain. Default is inf.
        """
        self.nuclides = list(data.index)
        self.elements = group_elements(self.nuclides)
        self.half_life_threshold = half_life_threshold

    def collect_droppable_progeny(self, elements):
        """Collect progeny of the specified elements.

        Find a list of decay progeny for all isotopes of the specified
        elements. The progeny that are an isotope of their decay parent
        are not included.
        
        Parameters
        ----------
        elements : list of str
            List of element symbols to collect progeny for.
        
        Returns
        -------
        set
            A set of nuclides that are progeny of the specified elements.
        """
        progeny = set()
        for element in elements:
            isotopes = self.elements.get(element, [])
            if not isotopes:
                msg = f"Warning! No isotopes found for element {element}."
                self.logger.warning(msg)
                continue
            element_progeny = set()
            for iso in isotopes:
                if iso.endswith("*"):
                    iso = iso.replace("*", "m")
                element_progeny |= get_decay_chain(
                    iso, threshold=self.half_life_threshold)

            # Decay progeny that are isotopes of the same element
            # are not included in the progeny set.
            if any([n.endswith("*") for n in element_progeny]):
                print(
                    f"Decay of element {element} produces excited states.!!!")
            drop_set = (element_progeny - set(isotopes))
            self.logger.info(f"Dropping progeny of {element}: {drop_set}")
            progeny |= drop_set
        return progeny

    def filter(self, elements):
        """Return all isotopes of the specified elements."""
        return self.collect_droppable_progeny(elements)


class ElementFilter(Filter):
    """Filter to remove all isotopes of specified elements."""

    def __init__(self, data):
        """Initialize the filter with a list of nuclides.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing nuclide data with nuclide IDs as the
            index. Units of the data should be in g/cm3.
        """
        self.nuclides = list(data.index)
        self.elements = group_elements(self.nuclides)

    def filter(self, elements):
        """Return all isotopes of the specified elements."""
        return self.collect_nuclides(elements)


class NuclideThresholdFilter(ElementThresholdFilter):
    """Filter to remove nuclides below a specified threshold."""

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
        self.nuclides = list(data.index)
        self.elements = group_elements(self.nuclides)
        self.data = data.copy()
        self.dilution_factor = dilution_factor
        self.actinide_reduction = actinide_reduction
        self.har_density = har_density
        self._process()

    def reduce_actinides(self):
        """Reduce actinide concentrations to account for reprocessing."""
        major_actinide_isotopes = []
        for actinide in self.major_actinides:
            major_actinide_isotopes.extend(self.elements.get(actinide, []))
        self.data.loc[major_actinide_isotopes] *= (1 - self.actinide_reduction)

    def _process(self):
        """Process the data through before filtering."""
        self.dilute()
        self.reduce_actinides()
        self.calc_mass_fractions()

    def filter(self, threshold=1e-9, percentile=0.25):
        """Filter elements based on a mass fraction threshold."""
        below = self.compare_threshold(threshold=threshold,
                                       percentile=percentile)
        return below.index.tolist()


class MassNumberFilter(Filter):
    """Filter to remove nuclides based on mass number."""

    def __init__(self, data):
        """Initialize ElementFilter with data.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing nuclide data with nuclide IDs as the
            index. Units of the data should be in g/cm3.
        """
        self.nuclides = list(data.index)
        self.mass_numbers = pd.Series(
            {n: int(get_mass_number(n))
             for n in self.nuclides})

    def filter(self, mass_number):
        """Return isotopes lower or equal to a specified mass number.
        
        Parameters
        ----------
        mass_number : int
            The mass number threshold.
        
        Returns
        -------
        list
            A list of nuclides with mass numbers less than or equal to the
            specified mass number.
        """
        return self.mass_numbers[self.mass_numbers <=
                                 mass_number].index.tolist()


class FilterFactory:
    """Factory class to create filters based on type."""

    filters = {
        "nuclide_threshold": NuclideThresholdFilter,
        "element_threshold": ElementThresholdFilter,
        "decay_progeny": DecayProgenyFilter,
        "element": ElementFilter,
        "mass_number": MassNumberFilter
    }

    @staticmethod
    def create_filter(filter_type, data, **kwargs):
        """Create a filter of the specified type."""
        filter_class = FilterFactory.filters.get(filter_type)
        if filter_class is None:
            raise ValueError(f"Unknown filter type: {filter_type}")
        return filter_class(data, **kwargs)
