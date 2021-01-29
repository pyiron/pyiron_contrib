# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Extends the :class:`.StructureContainer` to store energies and forces with the structures.
"""

import os.path
import pandas as pd

from pyiron_atomistics import pyiron_to_ase
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_base import GenericJob

class TrainingContainer(GenericJob):
    """
    Stores ASE structres with energies and forces.
    """

    def __init__(self, job_name, project):
        super().__init__(job_name, project)
        self._table = pd.DataFrame({
            "name": [],
            "atoms": [],
            "energy": [],
            "forces": [],
            "number_of_atoms": []
        })

    def include_structure(self, structure, energy, forces=None, name=None):
        """
        Add new structure to structure list and save energy and forces with it.

        For consistency with the rest of pyiron, energy should be in units of eV and forces in eV/A, but no conversion
        is performed.

        Args:
            structure_or_job (:class:`~.Atoms`, :class:`ase.Atoms`): if :class:`~.Atoms` convert to :class:`ase.Atoms`
            energy (float): energy of the whole structure
            forces (Nx3 array of float, optional): per atom forces, where N is the number of atoms in the structure
            name (str, optional): name describing the structure
        """
        if isinstance(structure, Atoms):
            structure = pyiron_to_ase(structure)
        self._table = self._table.append(
                {"name": name, "atoms": structure, "energy": energy, "forces": forces,
                 "number_of_atoms": len(structure)},
                ignore_index=True)

    def to_pandas(self):
        """
        Export list of structure to pandas table for external fitting codes.

        The table contains the following columns:
            - 'ase_atoms': the structure as a :class:`.Atoms` object
            - 'energy': the energy of the full structure
            - 'forces': the per atom forces as a :class:`numpy.ndarray`, shape Nx3
            - 'number_of_atoms': the number of atoms in the structure, N

        Returns:
            :class:`pandas.DataFrame`: collected structures
        """
        return self._table

    def write_input(self):
        pass

    def collect_output(self):
        pass

    def run_static(self):
        self.status.finished = True

    def run_if_interactive(self):
        self.to_hdf()
        self.status.finished = True

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self._table.to_hdf(self.project_hdf5.file_name, self.name + "/output/structure_table")

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        self._table = pd.read_hdf(self.project_hdf5.file_name, self.name + "/output/structure_table")
