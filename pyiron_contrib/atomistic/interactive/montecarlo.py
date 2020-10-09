# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import random
import scipy.constants as sci_const
from pyiron_base import GenericParameters
from pyiron.lammps.interactive import LammpsInteractive
from pyiron.atomistics.job.interactivewrapper import InteractiveWrapper
from pyiron.atomistics.structure.atoms import Atoms

__author__ = "Jan Janssen"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - " \
                "Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Jul 25, 2018"


class MonteCarloMaster(InteractiveWrapper):
    """
    Args:
        project (pyiron.project.Project instance):  Specifies the project path among other attributes
        job_name (str): Name of the job

    Attributes:
        input (pyiron.objects.hamilton.md.lammps.Input instance): Instance which handles the input
    """

    def __init__(self, project, job_name):
        super(MonteCarloMaster, self).__init__(project, job_name)
        self.__name__ = "MonteCarloMaster"
        self.input = MonteCarloInput()
        self._structure_sublattice = None
        self._structure_lattice = None

    @property
    def structure_sublattice(self):
        """

        Returns:

        """
        return self._structure_sublattice

    @structure_sublattice.setter
    def structure_sublattice(self, basis):
        """

        Args:
            basis:

        Returns:

        """
        self._structure_sublattice = basis

    @property
    def structure_lattice(self):
        """

        Returns:

        """
        return self._structure_lattice

    @structure_lattice.setter
    def structure_lattice(self, basis):
        """

        Args:
            basis:

        Returns:

        """
        self._structure_lattice = basis

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        super(MonteCarloMaster, self).set_input_to_read_only()
        self.input.read_only = True

    def validate_ready_to_run(self):
        """

        Returns:

        """
        super(MonteCarloMaster, self).validate_ready_to_run()
        if not self.structure_sublattice:
            raise ValueError('This job does not contain a valid intersitial structure: {}'.format(self.job_name))

    def db_entry(self):
        """
        Generate the initial database entry

        Returns:
            (dict): db_dict
        """
        db_dict = super(MonteCarloMaster, self).db_entry()
        if self.structure_lattice and self.structure_sublattice:
            interstitial_selected_lst = self._initial_config(self.input['number_of_interstitials'],
                                                             self.structure_sublattice)
            structure = self.structure_lattice + self.structure_sublattice[interstitial_selected_lst]
            parent_structure = structure.get_parent_basis()
            db_dict["ChemicalFormula"] = parent_structure.get_chemical_formula()
        return db_dict

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the ExampleJob object in the HDF5 File

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(MonteCarloMaster, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.structure_sublattice.to_hdf(hdf5_input, group_name="structure_sublattice")
            self.structure_lattice.to_hdf(hdf5_input, group_name="structure_lattice")

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the ExampleJob object in the HDF5 File

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(MonteCarloMaster, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.structure_sublattice = Atoms().from_hdf(hdf5_input, group_name="structure_sublattice")
            self.structure_lattice = Atoms().from_hdf(hdf5_input, group_name="structure_lattice")

    def write_input(self):
        pass

    def _write_run_wrapper(self, debug=False):
        pass

    def run_static(self):
        for temperature in self.input['temperatures']:
            for round_nr in range(int(self.input['run_rounds'])):
                self.status.running = True
                interstitial_selected_lst = self._initial_config(self.input['number_of_interstitials'],
                                                                 self.structure_sublattice)
                energy_accepted_lst, energy_total_lst = self._calc_monte_carlo(interstitial_selected_lst,
                                                                               temperature)
                if self.ref_job.server.run_mode.interactive:
                    self.ref_job.interactive_close()
                self.status.collect = True
                self._store_output_in_hdf5(temperature, round_nr, energy_accepted_lst, energy_total_lst)
        self._finish_job()

    def _calc_monte_carlo(self, interstitial_selected_lst, temperature):
        """

        Args:
            interstitial_selected_lst:
            temperature:

        Returns:

        """
        step_nr, try_nr = 0, 0
        e_prev = None
        energy_total_lst, energy_accepted_lst = [], []
        while step_nr < int(self.input['run_steps']) and \
                try_nr < int(self.input['run_steps']) * int(self.input['run_try_factor']):
            prev_positions = interstitial_selected_lst[:]

            interstitial_selected_lst = self._index_switch(interstitial_selected_lst)
            structure = self.structure_lattice + self.structure_sublattice[interstitial_selected_lst]
            e_step = self._calc_energy_for_structure(structure)

            if self._monte_carlo_accept_step(temperature, e_step, e_prev):
                e_prev = e_step
                energy_accepted_lst.append(e_step)
                step_nr += 1
            else:
                interstitial_selected_lst = prev_positions[:]
            energy_total_lst.append(e_step)
            try_nr += 1
        return energy_accepted_lst, energy_total_lst

    def _index_switch(self, interstitial_selected_lst):
        """

        Args:
            interstitial_selected_lst:

        Returns:

        """
        interstitial_not_selected_lst = self._not_selected(interstitial_selected_lst=interstitial_selected_lst,
                                                           sub_lattice_structure=self.structure_sublattice)
        random_new_element = random.choice(interstitial_not_selected_lst)
        random_current_element = random.choice(interstitial_selected_lst)
        del interstitial_selected_lst[interstitial_selected_lst.index(random_current_element)]
        interstitial_selected_lst.append(random_new_element)
        return interstitial_selected_lst

    def _calc_energy_for_structure(self, structure):
        """
        Calculate the energy using the ref_job

        Args:
            structure:

        Returns:
            float
        """
        if isinstance(self.ref_job, LammpsInteractive) and len(self._job_name_lst) == 0 and self.ref_job.server.run_mode.interactive:
            self.ref_job.interactive_positions_setter(structure.positions)
            self.ref_job._interactive_lib_command(self.ref_job._interactive_run_command)
            return self.ref_job.interactive_energy_tot_getter()
        else:
            self.ref_job.structure = structure
            self.ref_job_initialize()
            if self.ref_job.server.run_mode.interactive:
                self.ref_job.run()
            else:
                self.ref_job.run(run_again=True)
            return self.ref_job.output.energy_tot[-1]

    def _store_output_in_hdf5(self, temperature, round_nr, energy_accepted_lst, energy_total_lst):
        """
        Store accepted energies and total energies inside the HDF5 file

        Args:
            temperature:
            round_nr:
            energy_accepted_lst:
            energy_total_lst:
        """
        with self.project_hdf5.open("output") as h5:
            h5['energy_accepted_' + str(int(temperature)) + '_' + str(int(round_nr))] = \
                np.array(energy_accepted_lst)
            h5['energy_total_' + str(int(temperature)) + '_' + str(int(round_nr))] = np.array(energy_total_lst)

    @staticmethod
    def _monte_carlo_accept_step(temperature, e_step, e_prev):
        """
        Monte carlo step - decide whether a new step is accepted or not.

        Args:
            temperature:
            e_step:
            e_prev:

        Returns:
            boolean
        """
        if e_prev:
            energy_change = e_step - e_prev
            exponent = -energy_change / (sci_const.Boltzmann / sci_const.e * temperature)
            if exponent > 0:  # np.exp(709) = 8.2184074615549724e+307
                probability = 1.0  # np.exp(710) = inf  ('RuntimeWarning: overflow encountered in exp')
            else:
                probability = np.exp(exponent)
            return random.random() <= probability or energy_change < 0
        else:
            return True

    @staticmethod
    def _not_selected(interstitial_selected_lst, sub_lattice_structure):
        """

        Args:
            interstitial_selected_lst:
            sub_lattice_structure:

        Returns:

        """
        interstitial_all_lst = list(range(len(sub_lattice_structure)))
        return list(set(interstitial_all_lst) - set(interstitial_selected_lst))

    @staticmethod
    def _initial_config(number_of_interstitials, sublattice):
        """

        Args:
            number_of_interstitials:
            sublattice:

        Returns:

        """
        interstitial_index = []
        while len(interstitial_index) < number_of_interstitials:
            interstitial_index.append(random.randint(a=0, b=len(sublattice) - 1))
            interstitial_index = list(set(interstitial_index))
        return interstitial_index


class MonteCarloInput(GenericParameters):
    """
    Input class for the ExampleJob based on the GenericParameters class.

    Args:
        input_file_name (str): Name of the input file - optional
    """
    def __init__(self, input_file_name=None):
        super(MonteCarloInput, self).__init__(input_file_name=input_file_name,
                                              table_name="montecarlo_inp",
                                              separator_char="=",
                                              comment_char="#")

    def load_default(self):
        """
        Loading the default settings for the input file.
        """
        input_str = '''\
number_of_interstitials = 32
run_rounds = 1
run_steps = 100
run_try_factor = 10
temperatures = [200]
'''
        self.load_string(input_str)
