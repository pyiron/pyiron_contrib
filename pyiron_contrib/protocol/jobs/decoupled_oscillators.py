# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import numpy as np
from os.path import split

from pyiron_atomistics.atomistics.job.interactive import GenericInteractive
from pyiron_atomistics import Project
from pyiron_base.master.generic import GenericMaster
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
__date__ = "25 May, 2021"


class _DecoupledOscillatorsInput(DataContainer):
    def __init__(self, init=None, table_name='decoupled_input'):
        super().__init__(init=init, table_name=table_name)
        self.oscillators_id_list = None
        self.spring_constants_list = None
        self.save_debug_data = False
        self._ref_job = None
        self._structure = None
        self._positions = None
        self._ref_job_full_path = None
        self._base_job_name = None
        self._base_atom_ids = None

    @property
    def structure(self) -> Atoms:
        return self._structure

    @structure.setter
    def structure(self, atoms: Atoms):
        if not isinstance(atoms, Atoms):
            raise TypeError(f"<job>.input.structure must be of type Atoms but got {type(atoms)}")
        self._structure = atoms

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, pos):
        self._positions = pos

    @property
    def ref_job_full_path(self):
        return self._ref_job_full_path

    @ref_job_full_path.setter
    def ref_job_full_path(self, path):
        self._ref_job_full_path = path


class DecoupledOscillators(GenericInteractive, GenericMaster):
    def __init__(self, project, job_name):
        super(DecoupledOscillators, self).__init__(project, job_name)
        self.__version__ = "0.0.1"
        self.__name__ = "DecoupledOscillators"
        self.input = _DecoupledOscillatorsInput()
        self.output = DataContainer(table_name='decoupled_output')
        self.interactive_cache = {
            'forces': [],
            'energy_pot': [],
            'base_forces': [],
            'base_energy_pot': [],
            'harmonic_forces': [],
            'harmonic_energy_pot': []
        }
        self._base_job = None
        self._forces = None
        self._fast_lammps_mode = True

    @property
    def structure(self):
        return self.input.structure.copy()

    @structure.setter
    def structure(self, atoms):
        self.input.structure = atoms

    @property
    def positions(self):
        return self.input.positions

    @positions.setter
    def positions(self, pos):
        self.input.positions = pos

    def write_input(self):
        """
        GenericJob complains if this method is not implemented.
        """
        pass

    def _preliminary_check(self):
        """
        Check if the necessary inputs are provided.
        """
        # check if input structure is of the Atoms class, and set the base structure
        if self.input.structure is None:
            raise ValueError("<job>.input.structure is a necessary input")

        # check if oscillator_id_list is a list of integers
        if isinstance(self.input.oscillators_id_list, list):
            if not all(isinstance(element, int) for element in self.input.oscillators_id_list):
                raise ValueError("oscillator ids should be integers")
        else:
            raise ValueError("<job>.input.oscillators_id_list should be a list of integers")

        # check if spring_constants_list is a list of integers/floats
        if isinstance(self.input.spring_constants_list, list):
            if not all(isinstance(element, (int, float)) for element in self.input.spring_constants_list):
                raise ValueError("spring constants should be integers or floats, and not array "
                                 "data types like int64, float64 etc.")
        else:
            raise ValueError("<job>.input.spring_constants_list should be a list of integers or floats")

        # check if the length of oscillators_id_list and spring_constants_list is the same
        assert len(self.input.oscillators_id_list) == len(self.input.spring_constants_list), \
            "<job>.input.oscillators_id_list and <job>.input.spring_constants_list should have the same length"

    def _load_ref_job(self):
        """
        Load the reference job from its path.
        """
        if self.input.ref_job_full_path is None:
            raise ValueError("<job>.input.ref_job_full_path is a necessary input")
        else:
            ref_job_path, ref_job_name = split(self.input.ref_job_full_path)
            pr = Project(ref_job_path)
            self.input._ref_job = pr.load(ref_job_name)
            if not isinstance(self.input._ref_job, (LammpsInteractive, VaspInteractive, SphinxInteractive)):
                raise TypeError(f"Got reference type {type(self.input._ref_job)}, which is not a recognized "
                                f"interactive job")

    def _set_initial_positions(self):
        """
        Check if input positions are provided, otherwise set the positions as structure positions
        """
        if self.input.positions is None:
            self.input.positions = self.input._ref_job.structure.positions

    def _set_base_structure(self):
        """
        Create a base structure with vacancies at the oscillator atom ids.
        """
        # set it to the input structure
        self._base_structure = self.input.structure.copy()
        # remove atoms that are oscillators from the base structure
        for i, atom_id in enumerate(np.sort(self.input.oscillators_id_list)):
            new_atom_id = atom_id - i
            self._base_structure.pop(int(new_atom_id))

        # collect indices of atoms that are NOT harmonic oscillators
        self.input._base_atom_ids = np.delete(np.arange(len(self.input.structure)).astype(int),
                                              self.input.oscillators_id_list)

    @property
    def _base_name(self):
        """
        Returns the name of the child job of this class based off of the name of the job.
        """
        return self.job_name + "__base"

    def _create_base_job(self):
        """
        Create the base interpreter (Lammps/Vasp/Sphinx) job with the vacancy structure and save it.
        """
        # copy the reference job to create the base job
        self._base_job = self.input._ref_job.copy_to(
            project=self.project,
            new_job_name=self._base_name,
            input_only=True,
            new_database_entry=False,
            delete_existing_job=True
        )
        # set all the parameters
        self._base_job.structure = self._base_structure
        if self._fast_lammps_mode:
            self._base_job.interactive_flush_frequency = 10**10
            self._base_job.interactive_write_frequency = 10**10
        self._base_job.interactive_open()
        self._base_job.save()
        self._base_job.status.running = True
        self.input._base_job_name = self._base_job.name

    def _calc_static_base_job(self):
        """
        Run calc_static on the base structure using the interpreter (Lammps/Vasp/Sphinx).
        Returns:
            forces
            energy_pot
        """
        if isinstance(self._base_job, LammpsInteractive) and self._fast_lammps_mode:
            # this is the code that runs lammps super fast
            self._base_job.interactive_initialize_interface()
            self._base_job.interactive_positions_setter(self.input.positions[self.input._base_atom_ids])
            self._base_job._interactive_lib_command(self._base_job._interactive_run_command)
        else:
            self._base_job.structure.positions = self.input.positions[self.input._base_atom_ids]
            self._base_job.calc_static()
            self._base_job.run()
        return self._base_job.interactive_forces_getter(), self._base_job.interactive_energy_pot_getter()

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
        harmonic_forces = -np.array(spring_constants) * dr
        harmonic_energy_pot = 0
        for m in self.input.oscillators_id_list:
            harmonic_energy_pot += -0.5 * np.dot(dr[m], harmonic_forces[m])
        return harmonic_forces[self.input.oscillators_id_list], harmonic_energy_pot

    def validate_ready_to_run(self):
        """
        Perform these before running the job.
        """
        self._preliminary_check()
        self._load_ref_job()
        self._set_initial_positions()
        self._set_base_structure()
        self._create_base_job()

    def run_if_interactive(self):
        """
        The main run function.
        """
        if not self.status.running:
            self.status.running = True
        forces = np.zeros(self.input.positions.shape)
        forces[self.input._base_atom_ids], base_energy_pot = self._calc_static_base_job()  # get base forces and ens
        forces[self.input.oscillators_id_list], harmonic_energy_pot = self._calc_harmonic()  # get harm forces and ens
        energy_pot = base_energy_pot + harmonic_energy_pot
        self.interactive_cache['forces'].append(forces)
        self.interactive_cache['energy_pot'].append(energy_pot)
        if self.input.save_debug_data:
            self.interactive_cache['base_forces'].append(forces[self.input._base_atom_ids])
            self.interactive_cache['base_energy_pot'].append(base_energy_pot)
            self.interactive_cache['harmonic_forces'].append(forces[self.input.oscillators_id_list])
            self.interactive_cache['harmonic_energy_pot'].append(harmonic_energy_pot)

    def run_static(self):
        """
        If the job is not run interactively, then run it for one step (interactively) and close.
        """
        run_mode = self.server.run_mode.mode  # save the existing run mode
        self.interactive_open()  # change the run mode to interactive
        self.run_if_interactive()  # run interactively
        self.interactive_close()  # close interactive
        self.server.run_mode = run_mode  # set the current run mode to the initial run mode

    def interactive_forces_getter(self):
        return self.interactive_cache['forces'][-1]

    def interactive_energy_pot_getter(self):
        return self.interactive_cache['energy_pot'][-1]

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the DecoupledOscillator object in the HDF5 File
        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(DecoupledOscillators, self).to_hdf(hdf=hdf, group_name=group_name)
        # I do not want to save the reference job, so I set it to None
        if self.input._ref_job is not None:
            self.input._ref_job = None
        self.input.to_hdf(self.project_hdf5)
        self.output.to_hdf(self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the DecoupledOscillator object from the HDF5 File
        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(DecoupledOscillators, self).from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(self.project_hdf5)
        self.output.from_hdf(self.project_hdf5)
        # if the base job is not saved for some reason, load it back
        if self.input._base_job_name is not None:
            self._base_job = self.project.load(self.input._base_job_name)

    def interactive_close(self):
        # close the base_job
        # if it is already loaded and not an hdf object
        if not isinstance(self._base_job, ProjectHDFio):
            self._base_job.interactive_close()
            self._base_job.status.finished = True
        else:
            # if it is an hdf object, check its job status
            for job in self.project.iter_jobs(status="running"):
                if "__base" in job.job_name:
                    base_job = self.project.load(job.job_name)  # load and close
                    base_job.interactive_close()
                    base_job.status.finished = True
        # assign forces and energy_pot to output list
        self.output.forces = np.array(self.interactive_cache['forces'])
        self.output.energy_pot = np.array(self.interactive_cache['energy_pot'])
        if self.input.save_debug_data:
            self.output.base_forces = np.array(self.interactive_cache['base_forces'])
            self.output.base_energy_pot = np.array(self.interactive_cache['base_energy_pot'])
            self.output.harmonic_forces = np.array(self.interactive_cache['harmonic_forces'])
            self.output.harmonic_energy_pot = np.array(self.interactive_cache['harmonic_energy_pot'])
        self.to_hdf(self.project_hdf5)   # run to_hdf to re-save input
        self.output.to_hdf(self.project_hdf5)  # save output
        self.project.db.item_update(self._runtime(), self._job_id)  # update
        self.status.finished = True  # set job status to finish
