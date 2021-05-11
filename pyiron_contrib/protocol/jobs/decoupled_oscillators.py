# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import numpy as np
from abc import ABC

from pyiron_atomistics.atomistics.job.interactive import GenericInteractive
from pyiron_base.master.flexible import FlexibleMaster
from pyiron_base.generic.datacontainer import DataContainer
from pyiron_base.generic.hdfio import ProjectHDFio
from pyiron_atomistics.lammps.lammps import LammpsInteractive
from pyiron_atomistics.vasp.interactive import VaspInteractive
from pyiron_atomistics.sphinx.interactive import SphinxInteractive
from pyiron_atomistics.atomistics.structure.atoms import Atoms

__author__ = "Raynol Dsouza"
__copyright__ = "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Raynol Dsouza"
__email__ = "dsouza@mpie.de"
__status__ = "development"
__date__ = "06 May, 2021"


class DecoupledOscillators(GenericInteractive, FlexibleMaster, ABC):
    def __init__(self, project, job_name):
        super(DecoupledOscillators, self).__init__(project, job_name)
        self.__version__ = "0.0.1"
        self.__name__ = "DecoupledOscillators"
        self.input = DataContainer(table_name="decoupled_input")
        self.output = DataContainer(table_name="decoupled_output")
        self.input.structure = None
        self.input.ref_job_id = None
        self.input.save_components = False
        self.interactive_cache = {
            "forces": [],
            "energy_pot": [],
            "base_forces": [],
            "base_energy_pot": [],
            "harmonic_forces": [],
            "harmonic_energy_pot": []
        }
        self._fast_mode = True
        self._python_only_job = True
        self._base_structure = None
        self._base_atom_ids = None
        self._forces = None
        self._ref_job = None

    @property
    def structure(self):
        if self.input.structure is None:
            struct = self.input.ref_job.structure.copy()
        else:
            struct = self.input.structure.copy()
        return struct

    @structure.setter
    def structure(self, value):
        self.input.structure = value

    @property
    def positions(self):
        pos = self.input.positions
        return pos

    @positions.setter
    def positions(self, value):
        self.input.positions = value

    def _collect_ref_job_id(self):
        self.input.ref_job_id = self.input.ref_job.job_id
        self._ref_job = self.input.ref_job

    def _check_inputs(self):
        """
        Check if all necessary inputs are provided.
        """
        # check if input structure is of the Atoms class, and set the base structure
        if self.input.structure is not None:
            assert isinstance(self.input.structure, Atoms), \
                '<job>.input.structure should be an instance of the Atoms class'
            self._base_structure = self.input.structure.copy()
        # otherwise revert to the structure of the reference job
        elif self.input.structure is None:
            self.input.structure = self._ref_job.structure.copy()
            self._base_structure = self._ref_job.structure.copy()

        # check if input positions are provided
        assert self.input.positions is not None, '<job>.input.positions need to be provided'

        # check if oscillator_id_list is a list of integers
        if isinstance(self.input.oscillators_id_list, list):
            if not all(isinstance(element, int) for element in self.input.oscillators_id_list):
                raise ValueError('oscillator ids should be integers')
        else:
            raise ValueError('<job>.input.oscillators_id_list should be a list of integers')

        # check if spring_constants_list is a list of integers/floats
        if isinstance(self.input.spring_constants_list, list):
            if not all(isinstance(element, (int, float)) for element in self.input.spring_constants_list):
                raise ValueError('spring constants should be integers or floats')
        else:
            raise ValueError('<job>.input.spring_constants_list should be a list of integers or floats')

        # check if the length of oscillators_id_list and spring_constants_list is the same
        assert len(self.input.oscillators_id_list) == len(self.input.spring_constants_list), \
            '<job>.input.oscillators_id_list and <job>.input.spring_constants_list should have the same length'

    def _set_base_structure(self):
        """
        Create a base structure with vacancies at the oscillator atom ids.
        """

        # remove atoms that are oscillators from the base structure
        for i, atom_id in enumerate(self.input.oscillators_id_list):
            new_atom_id = atom_id - i
            self._base_structure.pop(new_atom_id)

        # collect indices of atoms that are NOT harmonic oscillators
        self._base_atom_ids = np.delete(np.arange(len(self._ref_job.structure)).astype(int),
                                        self.input.oscillators_id_list)

    def _create_base_job(self):
        """
        Create the base interpreter (Lammps/Vasp/Sphinx) job with the vacancy structure and save it.
        """

        # check if ref_job is an instance of Lammps/Vasp/Sphinx interactive
        assert isinstance(self._ref_job, (LammpsInteractive, VaspInteractive, SphinxInteractive)), \
            'ref_job should be one of Lammps/Vasp/SphinxInteractive'

        # copy the reference job to create the base job
        self.append(self._ref_job.copy_to(project=self.project, new_job_name='base_job', input_only=True))
        self[0].structure = self._base_structure

        # set interactive open
        self[0].interactive_open()
        self[0].interactive_initialize_interface()

        # change the flush and write frequencies, if fast_mode is enabled
        if self._fast_mode:
            self[0].interactive_flush_frequency = 10**10
            self[0].interactive_write_frequency = 10**10

        # save the job and set status to running
        self[0].save()
        self.status.running = True

    def _calc_static_base_job(self):
        """
        Run calc_static on the base structure using the interpreter (Lammps/Vasp/Sphinx).
        Returns:
            forces
            energy_pot
        """
        self[0].interactive_positions_setter(self.input.positions[self._base_atom_ids])
        self[0].run()
        return self[0].interactive_forces_getter(), self[0].interactive_energy_pot_getter()

    def _calc_harmonic(self):
        """
        Calculate the harmonic forces and energy_pot for the oscillators.
        Returns:
            forces
            energy_pot
        """
        reference_positions = self.input.structure.positions
        dr = self.input.structure.find_mic(self.input.positions - reference_positions)
        spring_constants = np.expand_dims(self.input.spring_constants_list, axis=-1)
        harmonic_forces = -np.array(spring_constants) * dr[self.input.oscillators_id_list]
        harmonic_energy_pot = 0
        for i, m in enumerate(self.input.oscillators_id_list):
            harmonic_energy_pot += -0.5 * np.dot(dr[m], harmonic_forces[i].T)
        return harmonic_forces, harmonic_energy_pot

    def validate_ready_to_run(self):
        """
        A pre check before running the main job. Also initializes the base job.
        """
        self._collect_ref_job_id()
        self._check_inputs()
        self.interactive_open()
        self.status.running = True
        self._set_base_structure()
        self._create_base_job()
        self._forces = np.zeros(self.input.positions.shape)

    def run_if_interactive(self):
        """
        The main run function.
        """
        self.status.running = True
        self._forces[self._base_atom_ids], base_energy_pot = self._calc_static_base_job()
        self._forces[self.input.oscillators_id_list], harmonic_energy_pot = self._calc_harmonic()
        energy_pot = base_energy_pot + harmonic_energy_pot
        self.interactive_cache["forces"].append(self._forces)
        self.interactive_cache["energy_pot"].append(energy_pot)
        if self.input.save_components:
            self.interactive_cache["base_forces"].append(self._forces[self._base_atom_ids])
            self.interactive_cache["base_energy_pot"].append(base_energy_pot)
            self.interactive_cache["harmonic_forces"].append(self._forces[self.input.oscillators_id_list])
            self.interactive_cache["harmonic_energy_pot"].append(harmonic_energy_pot)

    def run_static(self):
        """
        In case run_static is called.
        """
        self.run_if_interactive()
        self.interactive_close()

    def interactive_forces_getter(self):
        return self.output.forces

    def interactive_energy_pot_getter(self):
        return self.output.energy_pot

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the DecoupledOscillator object in the HDF5 File

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(DecoupledOscillators, self).to_hdf(hdf=hdf, group_name=group_name)
        # ref job probably still needs explicit care? It seems to work for now... -R
        with self.project_hdf5.open("decoupled_input") as hdf5_input:
            self.input.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the DecoupledOscillator object in the HDF5 File

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(DecoupledOscillators, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("decoupled_input") as hdf5_input:
            self.input = hdf5_input

    def interactive_close(self):
        self[0].interactive_close()  # close the base job
        self.to_hdf()   # run to_hdf to re-save input
        with self.project_hdf5.open("decoupled_output") as hdf5_output:
            self.output.to_hdf(hdf5_output)  # save output
        # assign forces and energy_pot to output list
        self.output.forces = np.array(self.interactive_cache["forces"])
        self.output.energy_pot = np.array(self.interactive_cache["energy_pot"])
        if self.input.save_components:
            self.output.base_forces = np.array(self.interactive_cache["base_forces"])
            self.output.base_energy_pot = np.array(self.interactive_cache["base_energy_pot"])
            self.output.harmonic_forces = np.array(self.interactive_cache["harmonic_forces"])
            self.output.harmonic_energy_pot = np.array(self.interactive_cache["harmonic_energy_pot"])
        self.project.db.item_update(self._runtime(), self._job_id)  # update
        self.status.finished = True  # set job status to finish
