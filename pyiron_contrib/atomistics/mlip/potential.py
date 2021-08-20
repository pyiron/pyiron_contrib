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
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - " \
                "Computational Materials Design (CM) Department"
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
        self._store = DataContainer(table_name=table_name)
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        with open(filename) as f:
            self._store.clear()
            self._store.update(parse_potential(f.read()), wrap=True)

    def to_hdf(self, hdf, group_name=None):
        self._store.to_hdf(hdf=hdf, group_name=group_name)

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
        if basis_type == 'Chebyshev':
            return {types: [scaling * np.polynomial.Chebyshev(coeffs, domain=(rmin, rmax))
                                for coeffs in funcs]
                                    for types, funcs in self._store.radial.funcs.items()}

        else:
            raise NotImplementedError(f"unknown basis type {basis_type}")

    @property
    def moment_coefficients(self):
        """
        array of floats: expansion coefficients for moment tensor contractions
        """
        return self._store.moment_coeffs
