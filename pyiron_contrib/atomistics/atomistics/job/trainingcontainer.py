# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Store structures together with energies and forces for potential fitting applications.

Basic usage:

>>> pr = Project("training")
>>> container = pr.create.job.TrainingContainer("small_structures")

Let's make a structure and invent some forces

>>> structure = pr.create.structure.ase_bulk("Fe")
>>> forces = numpy.array([-1, 1, -1])
>>> container.include_structure(structure, energy=-1.234, forces=forces, name="Fe_bcc")

If you have a lot of precomputed structures you may also add them in bulk from a pandas DataFrame

>>> df = pandas.DataFrame({ "name": "Fe_bcc", "atoms": structure, "energy": -1.234, "forces": forces })
>>> container.include_dataset(df)

You can retrieve the full database with :method:`~.TrainingContainer.to_pandas()` like this

>>> container.to_pandas()
name    atoms   energy  forces  number_of_atoms
Fe_bcc  ...
"""

from warnings import catch_warnings

import numpy as np
import pandas as pd
from pyiron_contrib.atomistics.atomistics.job.structurestorage import StructureStorage
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure
from pyiron_base import GenericJob


class TrainingContainer(GenericJob, HasStructure):
    """
    Stores ASE structures with energies and forces.
    """

    def __init__(self, project, job_name):
        super().__init__(project=project, job_name=job_name)
        self.__name__ = "TrainingContainer"
        self.__hdf_version__ = "0.2.0"
        self._container = StructureStorage()
        self._container.add_array("energy", dtype=np.float64, per="chunk")
        self._container.add_array("forces", shape=(3,), dtype=np.float64, per="element")
        # save stress in voigt notation
        self._container.add_array("stress", shape=(6,), dtype=np.float64, per="chunk")
        self._table_cache = None

    @property
    def _table(self):
        if self._table_cache is None or len(self._table_cache) != len(self._container):
            self._table_cache = pd.DataFrame({
                "name":             [self._container.get_array("identifier", i)
                                        for i in range(len(self._container))],
                "atoms":            [self._container.get_structure(i)
                                        for i in range(len(self._container))],
                "energy":           [self._container.get_array("energy", i)
                                        for i in range(len(self._container))],
                "forces":           [self._container.get_array("forces", i)
                                        for i in range(len(self._container))],
                "stress":           [self._container.get_array("stress", i)
                                        for i in range(len(self._container))],
            })
            self._table_cache["number_of_atoms"] = [len(s) for s in self._table_cache.atoms]
        return self._table_cache

    def include_job(self, job, iteration_step=-1):
        """
        Add structure, energy and forces from job.

        Args:
            job (:class:`.AtomisticGenericJob`): job to take structure from
            iteration_step (int, optional): if job has multiple steps, this selects which to add
        """
        energy = job.output.energy_pot[iteration_step]
        ff = job.output.forces
        if ff is not None:
            forces = ff[iteration_step]
        else:
            forces = None
        # HACK: VASP work-around, current contents of pressures are meaningless, correct values are in
        # output/generic/stresses
        pp = job["output/generic/stresses"]
        if pp is None:
            pp = job.output.pressures
        if pp is not None:
            stress = pp[iteration_step]
        else:
            stress = None
        if stress is not None:
            stress = np.asarray(stress)
            if stress.shape == (3, 3):
                stress = np.array([stress[0, 0], stress[1 ,1], stress[2, 2],
                                   stress[1, 2], stress[0, 2], stress[0, 1]])
        self.include_structure(job.get_structure(iteration_step=iteration_step),
                               energy=energy, forces=forces, stress=stress,
                               name=job.name)

    def include_structure(self, structure, energy, forces=None, stress=None, name=None):
        """
        Add new structure to structure list and save energy and forces with it.

        For consistency with the rest of pyiron, energy should be in units of eV and forces in eV/A, but no conversion
        is performed.

        Args:
            structure_or_job (:class:`~.Atoms`): structure to add
            energy (float): energy of the whole structure
            forces (Nx3 array of float, optional): per atom forces, where N is the number of atoms in the structure
            stress (3x3 array of float, optional): per structure stresses
            name (str, optional): name describing the structure
        """
        data = {"energy": energy}
        if forces is not None:
            data["forces"] = forces
        if stress is not None:
            data["stress"] = stress
        self._container.add_structure(structure, name, **data)
        if self._table_cache:
            self._table = self._table.append(
                    {"name": name, "atoms": structure, "energy": energy, "forces": forces, "stress": stress,
                     "number_of_atoms": len(structure)},
                    ignore_index=True)

    def include_dataset(self, dataset):
        """
        Add a pandas DataFrame to the saved structures.

        The dataframe should have the following columns:
            - name: human readable name of the structure
            - atoms(:class:`ase.Atoms`): the atomic structure
            - energy(float): energy of the whole structure
            - forces (Nx3 array of float): per atom forces, where N is the number of atoms in the structure
        """
        self._table_cache = self._table.append(dataset, ignore_index=True)
        # in case given dataset has more columns than the necessary ones, swallow/ignore them in *_
        for name, atoms, energy, forces, *_ in dataset.itertuples(index=False):
            self._container.add_structure(atoms, name, energy=energy, forces=forces)

    def _get_structure(self, frame=-1, wrap_atoms=True):
        return self._container.get_structure(frame=frame, wrap_atoms=wrap_atoms)

    def _number_of_structures(self):
        return self._container.number_of_structures

    def get_elements(self):
        """
        Return a list of chemical elements in the training set.

        Returns:
            :class:`list`: list of unique elements in the training set as strings of their standard abbreviations
        """
        return self._container.get_elements()

    def to_pandas(self):
        """
        Export list of structure to pandas table for external fitting codes.

        The table contains the following columns:
            - 'name': human-readable name of the structure
            - 'ase_atoms': the structure as a :class:`.Atoms` object
            - 'energy': the energy of the full structure
            - 'forces': the per atom forces as a :class:`numpy.ndarray`, shape Nx3
            - 'number_of_atoms': the number of atoms in the structure, N

        Returns:
            :class:`pandas.DataFrame`: collected structures
        """
        return self._table

    def to_list(self, filter_function=None):
        """
        Returns the data as lists of pyiron structures, energies, forces, and the number of atoms

        Args:
            filter_function (function): Function applied to the dataset (which is a pandas DataFrame) to filter it

        Returns:
            tuple: list of structures, energies, forces, and the number of atoms
        """
        if filter_function is None:
            data_table = self._table
        else:
            data_table = filter_function(self._table)
        structure_list = data_table.atoms.to_list()
        energy_list = data_table.energy.to_list()
        force_list = data_table.forces.to_list()
        num_atoms_list = data_table.number_of_atoms.to_list()
        return structure_list, energy_list, force_list, num_atoms_list

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
        self._container.to_hdf(self.project_hdf5, "structures")

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        hdf_version = self.project_hdf5.get("HDF_VERSION", "0.1.0")
        if hdf_version == "0.1.0":
            table = pd.read_hdf(self.project_hdf5.file_name, self.name + "/output/structure_table")
            self.include_dataset(table)
        else:
            self._container = StructureStorage()
            self._container.from_hdf(self.project_hdf5, "structures")
