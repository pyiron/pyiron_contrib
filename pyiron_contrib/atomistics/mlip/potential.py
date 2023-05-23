# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Convenience class to handle fitted MTPs.
"""

from pyiron_base import DataContainer
from pyiron_contrib.atomistics.mlip.parser import potential as parse_potential

import numpy as np

__author__ = "Marvin Poul"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Marvin Poul"
__email__ = "poul@mpie.de"
__status__ = "development"
__date__ = "Aug 18, 2021"


class MtpPotential:
    """
    Representation of a fitted MTP.

    Potentials may be loaded from a file when initializing

    >>> pot = MtpPotential("some/path")

    or later with :method:`.load()`

    >>> pot = MtpPotential()
    >>> pot.load("another/path")

    This class exports the fitted radial basis and the moment coefficients

    >>> pot.moment_coefficients
    array([...])
    >>> pot.radial_basis[0]
    Chebyshev([...], domain=(...), window=(...))

    """

    def __init__(self, filename=None, table_name="potential"):
        """
        Create a new potential.

        If `filename` is not given, :method:`.load()` must be called before the class is usable.

        Args:
            filename (str, optional): from where to load the potential
            table_name (str, optional): default group name when saving to HDF, default 'potential'
        """
        self.__hdf_version__ = "0.0.1"
        self._store = DataContainer(table_name=table_name)
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        with open(filename) as f:
            self._store.clear()
            self._store.update(parse_potential(f.read()), wrap=True)

    def write(self, filename):
        """
        Write potential to a text file readable by mlp.

        Args:
            filename (str): file path
        """

        def format_array(a):
            return "{" + ", ".join(map(str, a)) + "}"

        with open(filename, "w") as f:
            f.write("MTP\n")
            f.write(f"version = {self._store.version}\n")
            f.write(f"potential_name = {self._store.potential_name}\n")
            f.write(f"scaling = {self._store.scaling:.15e}\n")
            f.write(f"species_count = {self._store.species_count}\n")
            f.write(f"potential_tag = {self._store.get('potential_tag', '')}\n")
            f.write(f"radial_basis_type = RB{self._store.radial.basis_type}\n")
            f.write(f"\tmin_dist = {self._store.radial.info.min_dist:.15e}\n")
            f.write(f"\tmax_dist = {self._store.radial.info.max_dist:.15e}\n")
            f.write(f"\tradial_basis_size = {self._store.radial.info.basis_size}\n")
            f.write(f"\tradial_funcs_count = {self._store.radial.info.funcs_count}\n")
            f.write(f"\tradial_coeffs\n")
            for types, funcs in self._store.radial.funcs.items():
                f.write(f"\t\t{types}\n")
                for coeffs in funcs:
                    f.write("\t\t\t" + format_array(coeffs) + "\n")
            f.write(f"alpha_moments_count = {self._store.alpha_moments_count}\n")
            f.write(
                f"alpha_index_basic_count = {self._store.alpha_index_basic_count}\n"
            )
            f.write(
                f"alpha_index_basic = {format_array(map(format_array, self._store.alpha_index_basic))}\n"
            )
            f.write(
                f"alpha_index_times_count = {self._store.alpha_index_times_count}\n"
            )
            f.write(
                f"alpha_index_times = {format_array(map(format_array, self._store.alpha_index_times))}\n"
            )
            f.write(f"alpha_scalar_moments = {self._store.alpha_scalar_moments}\n")
            f.write(
                f"alpha_moment_mapping = {format_array(self._store.alpha_moment_mapping)}\n"
            )
            f.write(f"species_coeffs = {format_array(self._store.species_coeffs)}\n")
            f.write(f"moment_coeffs = {format_array(self._store.moment_coeffs)}\n")

    def _type_to_hdf(self, hdf):
        """
        Internal helper function to save type and version in hdf root

        Args:
            hdf (ProjectHDFio): HDF5 group object
        """
        hdf["NAME"] = self.__class__.__name__
        hdf["TYPE"] = str(type(self))
        hdf["HDF_VERSION"] = self.__hdf_version__

    def to_hdf(self, hdf, group_name=None):
        if group_name is not None:
            hdf = hdf.open(group_name)
        # force DataContainer to write into hdf group directly
        self._store.to_hdf(hdf=hdf, group_name=None)
        # then overwrite type information with ours, so we get reinstantiated later
        self._type_to_hdf(hdf)
        if group_name is not None:
            hdf.close()

    def from_hdf(self, hdf, group_name=None):
        self._store.from_hdf(hdf=hdf, group_name=group_name)

    @property
    def radial_basis(self):
        """
        list of numpy polynomials: radial basis functions used to compute the moment tensors
        """
        basis_type = self._store.radial.basis_type
        rmin = self._store.radial.info.min_dist
        rmax = self._store.radial.info.max_dist
        scaling = self._store.scaling
        if basis_type == "Chebyshev":
            return {
                types: [
                    scaling * np.polynomial.Chebyshev(coeffs, domain=(rmin, rmax))
                    for coeffs in funcs
                ]
                for types, funcs in self._store.radial.funcs.items()
            }

        else:
            raise NotImplementedError(f"unknown basis type {basis_type}")

    @property
    def moment_coefficients(self):
        """
        array of floats: expansion coefficients for moment tensor contractions
        """
        return self._store.moment_coeffs
