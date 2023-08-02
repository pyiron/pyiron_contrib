# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

__author__ = "Sam Waseda"
__copyright__ = (
    "Copyright 2023, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sam Waseda"
__email__ = "waseda@mpie.de"
__status__ = "production"
__date__ = "April 18, 2023"

import pandas as pd
from pyiron_atomistics.lammps.potential import LammpsPotentialFile
import numpy as np
import warnings
import itertools
from functools import wraps


general_doc = """
How to combine potentials:

Example I: Hybrid potential for a single element

>>> from pyiron_atomistics.lammps.potentials import Library, Morse
>>> eam = Library("Al")
>>> morse = Morse("Al", D_0=0.5, alpha=1.1, r_0=2.1, cutoff=6)
>>> lammps_job.potential = eam + morse

Example II: Hybrid potential for multiple elements

>>> from pyiron_atomistics.lammps.potentials import Library, Morse
>>> eam = Library("Al")
>>> morse_Al_Ni = Morse("Al", "Ni", D_0=0.2, alpha=1.05, r_0=2.2, cutoff=6)
>>> morse_Ni = Morse("Ni", D_0=0.7, alpha=1.15, r_0=2.15, cutoff=6)
>>> lammps_job.potential = eam + morse_Al_Ni + morse_Ni  # hybrid/overlay
>>> lammps_job.potential = eam * morse_Al_Ni * morse_Ni  # hybrid
>>> lammps_job.potential = 0.4 * eam + 0.1 * morse_Al_Ni + morse_Ni  # hybrid/scaled

"""


doc_pyiron_df = """
- "Config": Lammps commands in a list. Lines are separated by list items and
    each entry must finish with a new line
- "Filename": Potential file name in either absolute path or relative to
    the pyiron-resources path (optional)
- "Model": Model name (optional)
- "Name": Name of the potential (optional)
- "Species": Order of species as defined in pair_coeff
- "Citations": Citations (optional)

Example

>>> import pandas as pd
>>> pd.DataFrame(
...     {
...         "Config": [[
...             'pair_style my_potential 3.2\n',
...             'pair_coeff 2 1 1.1 2.3 3.2\n'
...         ]],
...         "Filename": [""],
...         "Model": ["my_model"],
...         "Name": ["my_potential"],
...         "Species": [["Fe", "Al"]],
...         "Citations": [],
...     }
... )
"""


issue_page = "https://github.com/pyiron/pyiron_atomistics/issues"


class LammpsPotentials:
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        cls._df = None
        return obj

    @staticmethod
    def _harmonize_species(species_symbols) -> list:
        """
        Check whether species are set for the pairwise interactions. If only
        one chemical species is given, duplicate the species.
        """
        if len(species_symbols) == 0:
            raise ValueError("Chemical elements not specified")
        if len(species_symbols) == 1:
            species_symbols *= 2
        return list(species_symbols)

    def _initialize_df(
        self,
        pair_style,
        interacting_species,
        pair_coeff,
        preset_species=None,
        model=None,
        citations=None,
        filename=None,
        potential_name=None,
        scale=None,
        cutoff=None,
    ):
        def check_none_n_length(variable, default, length=len(pair_coeff)):
            if variable is None:
                variable = default
            if isinstance(variable, list) and len(variable) == 1 < length:
                variable = length * variable
            return variable

        arg_dict = {
            "pair_style": pair_style,
            "interacting_species": interacting_species,
            "pair_coeff": pair_coeff,
            "preset_species": check_none_n_length(preset_species, [[]]),
            "cutoff": check_none_n_length(cutoff, 0),
            "model": check_none_n_length(model, pair_style),
            "citations": check_none_n_length(citations, [[]]),
            "filename": check_none_n_length(filename, [""]),
            "potential_name": check_none_n_length(potential_name, pair_style),
        }
        if scale is not None:
            arg_dict["scale"] = scale
        try:
            self.set_data(pd.DataFrame(arg_dict))
        except ValueError:
            raise ValueError(
                f"Initialization failed - inconsistency in data: {arg_dict}"
            )

    def copy(self):
        new_pot = LammpsPotentials()
        new_pot.set_data(self.get_all_data())
        return new_pot

    @staticmethod
    def _unique(args):
        labels, indices = np.unique(args, return_index=True)
        return labels[np.argsort(indices)]

    @property
    def model(self) -> str:
        """Model name (required in pyiron df)"""
        return "_and_".join(self._unique(self.df.model))

    @property
    def potential_name(self) -> str:
        """Potential name (required in pyiron df)"""
        return "_and_".join(self._unique(self.df.potential_name))

    @property
    def species(self):
        """Species defined in the potential"""
        species = self._unique([ss for s in self.df.interacting_species for ss in s])
        preset = self._unique(
            ["___".join(s) for s in self.df.preset_species if len(s) > 0]
        )
        if len(preset) == 0:
            return list(species)
        elif len(preset) > 1:
            raise NotImplementedError(
                "Currently not possible to have multiple file-based potentials"
            )
        preset = list(preset)[0].split("___")
        comp_lst = [s for s in species if s not in self._unique(preset)]
        return [p for p in preset + comp_lst if p != "*"]

    @property
    def filename(self) -> list:
        """LAMMPS potential files"""
        return [f for f in self._unique(self.df.filename) if len(f) > 0]

    @property
    def citations(self) -> str:
        """Citations to be included"""
        return "".join(np.unique([c for c in self.df.citations if len(c) > 0]))

    @property
    def is_scaled(self) -> bool:
        """Scaling in pair_style hybrid/scaled and hybrid/overlay (which is scale=1)"""
        return "scale" in self.df

    @property
    def pair_style(self) -> str:
        """LAMMPS pair_style"""
        if len(set(self.df.pair_style)) == 1:
            pair_style = "pair_style " + list(set(self.df.pair_style))[0]
            if np.max(self.df.cutoff) > 0:
                pair_style += f" {np.max(self.df.cutoff)}"
            return pair_style + "\n"
        elif "scale" not in self.df:
            pair_style = "pair_style hybrid"
        elif all(self.df.scale == 1):
            pair_style = "pair_style hybrid/overlay"
        else:
            pair_style = "pair_style hybrid/scaled"
        for ii, s in enumerate(self.df[["pair_style", "cutoff"]].values):
            if pair_style.startswith("pair_style hybrid/scaled"):
                pair_style += f" {self.df.iloc[ii].scale}"
            pair_style += f" {s[0]}"
            if s[1] > 0:
                pair_style += f" {s[1]}"
        return pair_style + "\n"

    class _PairCoeff:
        def __init__(
            self,
            pair_style,
            interacting_species,
            pair_coeff,
            species,
            preset_species,
        ):
            self._interacting_species = interacting_species
            self._pair_coeff = pair_coeff
            self._species = species
            self._preset_species = preset_species
            self._pair_style = pair_style
            self._s_dict = None

        @property
        def is_hybrid(self):
            return len(set(self._pair_style)) > 1

        @property
        def counter(self):
            """
            Enumeration of potentials if a potential is used multiple
            times in hybrid (which is a requirement from LAMMPS)
            """
            key, count = np.unique(self._pair_style, return_counts=True)
            counter = {kk: 1 for kk in key[count > 1]}
            results = []
            for coeff in self._pair_style:
                if coeff in counter and self.is_hybrid:
                    results.append(str(counter[coeff]))
                    counter[coeff] += 1
                else:
                    results.append("")
            return results

        @property
        def pair_style(self):
            """pair_style to be output only in hybrid"""
            if self.is_hybrid:
                return self._pair_style
            else:
                return len(self._pair_style) * [""]

        @property
        def results(self):
            """pair_coeff lines to be used in pyiron df"""
            return [
                " ".join((" ".join(("pair_coeff",) + c)).split()) + "\n"
                for c in zip(
                    self.interacting_species,
                    self.pair_style,
                    self.counter,
                    self.pair_coeff,
                )
            ]

        @property
        def s_dict(self):
            if self._s_dict is None:
                self._s_dict = dict(
                    zip(self._species, (np.arange(len(self._species)) + 1).astype(str))
                )
                self._s_dict.update({"*": "*"})
            return self._s_dict

        @property
        def interacting_species(self) -> list:
            """
            Species in LAMMPS notation (i.e. in numbers instead of chemical
            symbols)
            """
            return [
                " ".join([self.s_dict[cc] for cc in c])
                for c in self._interacting_species
            ]

        @property
        def pair_coeff(self) -> list:
            """
            Args for pair_coeff. Elements defined in EAM files are
            complemented with the ones defined in other potentials in the
            case of hybrid (filled with NULL)
            """
            if not self.is_hybrid:
                return self._pair_coeff
            results = []
            for pc, ps in zip(self._pair_coeff, self._preset_species):
                if len(ps) > 0 and "eam" in pc:
                    s = " ".join(ps + (len(self._species) - len(ps)) * ["NULL"])
                    pc = s.join(pc.rsplit(" ".join(ps), 1))
                results.append(pc)
            return results

    @property
    def pair_coeff(self) -> list:
        """LAMMPS pair_coeff"""

        return self._PairCoeff(
            pair_style=self.df.pair_style,
            interacting_species=self.df.interacting_species,
            pair_coeff=self.df.pair_coeff,
            species=self.species,
            preset_species=self.df.preset_species,
        ).results

    def get_df(self):
        """df used in pyiron potential"""
        return pd.DataFrame(
            {
                "Config": [[self.pair_style] + self.pair_coeff],
                "Filename": [self.filename],
                "Model": [self.model],
                "Name": [self.potential_name],
                "Species": [self.species],
                "Citations": [self.citations],
            }
        )

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()

    def set_data(self, df):
        for key in [
            "pair_style",
            "interacting_species",
            "pair_coeff",
            "preset_species",
        ]:
            if key not in df:
                raise ValueError(f"{key} missing")
        self._df = df

    @property
    def df(self):
        """DataFrame containing all info for each pairwise interactions"""
        return self._df

    def get_all_data(self, default_scale=None):
        if default_scale is None or "scale" in self.df:
            return self.df.copy()
        df = self.df.copy()
        df["scale"] = default_scale
        return df

    def __mul__(self, scale_or_potential):
        if isinstance(scale_or_potential, LammpsPotentials):
            if self.is_scaled or scale_or_potential.is_scaled:
                raise ValueError("You cannot mix hybrid types")
            new_pot = LammpsPotentials()
            new_pot.set_data(
                pd.concat(
                    (self.get_all_data(), scale_or_potential.get_all_data()),
                    ignore_index=True,
                )
            )
            return new_pot
        if self.is_scaled:
            raise NotImplementedError("Currently you cannot scale twice")
        new_pot = self.copy()
        new_pot.df["scale"] = scale_or_potential
        return new_pot

    __rmul__ = __mul__

    def __add__(self, potential):
        new_pot = LammpsPotentials()
        new_pot.set_data(
            pd.concat(
                (
                    self.get_all_data(default_scale=1),
                    potential.get_all_data(default_scale=1),
                ),
                ignore_index=True,
            )
        )
        return new_pot


class Library(LammpsPotentials):
    """
    Potential class to choose a file based potential from an existing library
    (e.g. EAM).
    You can either specify the chemical species and/or the name of the
    potential.

    Example I: Via chemical species

    >>> eam = Library("Al")

    Example II: Via potential name

    >>> eam = Library(name="1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1")

    If the variable `eam` is used without specifying the potential name (i.e.
    in Example I), the first potential in the database corresponding with the
    specified chemical species will be selected. In order to see the list of
    potentials, you can also execute

    >>> eam = Library("Al")
    >>> eam.list_potentials()  # See list of potential names
    >>> eam.view_potentials()  # See potential names and metadata

    """

    def __init__(self, *chemical_elements, name=None):
        """
        Args:
            chemical_elements (str): chemical elements/species
            name (str): potential name in the database
        """
        if name is not None:
            self._df_candidates = LammpsPotentialFile().find_by_name(name)
        else:
            self._df_candidates = LammpsPotentialFile().find(list(chemical_elements))

    @staticmethod
    def _get_pair_style(config):
        if any(["hybrid" in c for c in config]):
            return [c.split()[3] for c in config if "pair_coeff" in c]
        for c in config:
            if "pair_style" in c:
                return [" ".join(c.replace("\n", "").split()[1:])] * sum(
                    ["pair_coeff" in c for c in config]
                )
        raise ValueError(
            f"pair_style could not determined: {config}.\n\n"
            "The reason why you are seeing this error is most likely because "
            "the potential you chose had a corrupt config. It is "
            "supposed to have at least one item which starts with 'pair_style'.\n"
            "If you are using the standard pyiron database, feel free to "
            f"submit an issue on {issue_page}. "
            "Typically you can get a reply within 24h.\n"
        )

    @staticmethod
    def _get_pair_coeff(config):
        try:
            if any(["hybrid" in c for c in config]):
                return [" ".join(c.split()[4:]) for c in config if "pair_coeff" in c]
            return [" ".join(c.split()[3:]) for c in config if "pair_coeff" in c]
        except IndexError:
            raise AssertionError(
                f"{config} does not follow the format 'pair_coeff element_1 element_2 args'"
            )

    @staticmethod
    def _get_interacting_species(config, species):
        def _convert(c, s):
            if c == "*":
                return c
            return s[int(c) - 1]

        return [
            [_convert(cc, species) for cc in c.split()[1:3]]
            for c in config
            if c.startswith("pair_coeff")
        ]

    @staticmethod
    def _get_scale(config):
        for c in config:
            if not c.startswith("pair_style"):
                continue
            if "hybrid/overlay" in c:
                return 1
            elif "hybrid/scaled" in c:
                raise NotImplementedError(
                    "Too much work for something inexistent in pyiron database for now"
                )
        return

    def list_potentials(self):
        return self._df_candidates.Name

    def view_potentials(self):
        return self._df_candidates

    @property
    def df(self):
        if self._df is None:
            df = self._df_candidates.iloc[0]
            if len(self._df_candidates) > 1:
                warnings.warn(
                    f"Potential not uniquely specified - use default {df.Name}"
                )
            self._initialize_df(
                pair_style=self._get_pair_style(df.Config),
                interacting_species=self._get_interacting_species(
                    df.Config, df.Species
                ),
                pair_coeff=self._get_pair_coeff(df.Config),
                preset_species=[df.Species],
                model=df.Model,
                citations=df.Citations,
                filename=df.Filename,
                potential_name=df.Name,
                scale=self._get_scale(df.Config),
            )
        return self._df


def check_cutoff(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "cutoff" not in kwargs or kwargs["cutoff"] == 0:
            raise ValueError(
                "It is not possible to set cutoff=0 for parameter-based "
                "potentials. If you think this should be possible, you have the "
                "following options:\n\n"
                f"- Open an issue on our GitHub page: {issue_page}\n"
                "- Write your own potential in pyiron format. Here's how:\n"
                f"{doc_pyiron_df}\n"
            )
        return f(*args, **kwargs)

    return wrapper


class Morse(LammpsPotentials):
    """
    Morse potential defined by:

    E = D_0*[exp(-2*alpha*(r-r_0))-2*exp(-alpha*(r-r_0))]
    """

    @check_cutoff
    def __init__(self, *chemical_elements, D_0, alpha, r_0, cutoff, pair_style="morse"):
        """
        Args:
            chemical_elements (str): Chemical elements
            D_0 (float): parameter (s. eq. above)
            alpha (float): parameter (s. eq. above)
            r_0 (float): parameter (s. eq. above)
            cutoff (float): cutoff length
            pair_style (str): pair_style name (default: "morse")

        Example:

        >>> morse = Morse("Al", "Ni", D_0=1, alpha=0.5, r_0=2, cutoff=6)
        """
        self._initialize_df(
            pair_style=[pair_style],
            interacting_species=[self._harmonize_species(chemical_elements)],
            pair_coeff=[" ".join([str(cc) for cc in [D_0, alpha, r_0, cutoff]])],
            cutoff=cutoff,
        )


Morse.__doc__ += general_doc


class CustomPotential(LammpsPotentials):
    """
    Custom potential class to define LAMMPS potential not implemented in
    pyiron
    """

    @check_cutoff
    def __init__(self, pair_style, *chemical_elements, cutoff, **kwargs):
        """
        Args:
            pair_style (str): pair_style name (default: "morse")
            chemical_elements (str): Chemical elements
            cutoff (float): cutoff length

        Example:

        >>> custom_pot = CustomPotential("lj/cut", "Al", "Ni", epsilon=0.5, sigma=1, cutoff=3)

        Important: the order of parameters is conserved in the LAMMPS input
        (except for `cutoff`, which is always the last argument).
        """
        self._initialize_df(
            pair_style=[pair_style],
            interacting_species=[self._harmonize_species(chemical_elements)],
            pair_coeff=[" ".join([str(cc) for cc in kwargs.values()]) + f" {cutoff}"],
            cutoff=cutoff,
        )


CustomPotential.__doc__ += general_doc
