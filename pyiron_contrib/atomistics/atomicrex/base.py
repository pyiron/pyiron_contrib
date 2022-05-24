# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""Pyiron interface to atomicrex"""
import numpy as np
import pandas as pd

from pyiron_base import state, GenericJob, Executable, FlattenedStorage

from pyiron_contrib.atomistics.atomicrex.general_input import (
    GeneralARInput,
    AlgorithmFactory,
)
from pyiron_contrib.atomistics.atomicrex.structure_list import ARStructureContainer
from pyiron_contrib.atomistics.atomicrex.potential_factory import ARPotFactory
from pyiron_contrib.atomistics.atomicrex.output import Output
from pyiron_contrib.atomistics.atomicrex.function_factory import FunctionFactory
from pyiron_contrib.atomistics.ml.potentialfit import PotentialFit
from pyiron_contrib.atomistics.atomistics.job.trainingcontainer import (
    TrainingContainer,
    TrainingStorage,
)


class AtomicrexBase(GenericJob, PotentialFit):
    __version__ = "0.1.0"
    __hdf_version__ = "0.1.0"
    """Class to set up and run atomicrex jobs"""

    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        # self._executable_activate(enforce=True)
        state.publications.add(self.publication)
        self.input = GeneralARInput()
        self.potential = None
        self.structures = ARStructureContainer()
        self.output = Output()
        self.factories = Factories()
        self._compress_by_default = True

    def plot_final_potential(self):
        """
        Plot the fitted potential.
        Returns the matplotlib objects to change the look of the plot.

        Returns:
            [matplotlib figure, axis]: [description]
        """
        return self.potential.plot_final_potential(self)

    def to_hdf(self, hdf=None, group_name=None):
        """Internal function to store the job in hdf5 format"""
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(hdf=self.project_hdf5)
        self.potential.to_hdf(hdf=self.project_hdf5)
        self.structures.to_hdf(hdf=self.project_hdf5)
        self.output.to_hdf(hdf=self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        """Internal function to reload the job object from hdf5"""
        super().from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(hdf=self.project_hdf5)
        self.potential = self.project_hdf5["potential"].to_object()
        self.structures.from_hdf(hdf=self.project_hdf5)
        self.output.from_hdf(hdf=self.project_hdf5)

    @property
    def publication(self):
        return {
            "atomicrex": {
                "atomicrex": {
                    "title": "Atomicrex-a general purpose tool for the construction of atomic interaction models",
                    "journal": "Modelling and Simulation in Materials Science and Engineering",
                    "volume": "25",
                    "number": "5",
                    "year": "2017",
                    "issn": "0965-0393",
                    "doi": "10.1088/1361-651X/aa6ecf",
                    "url": "https://doi.org/10.1088%2F1361-651x%2Faa6ecf",
                    "author": [
                        "Alexander Stukowski",
                        "Erik Fransson",
                        "Markus Mock",
                        "Paul Erhart",
                    ],
                }
            }
        }

    def compress(self, files_to_compress=None):
        """
        Compress the output files of a job object.

        Args:
            files_to_compress (list): A list of files to compress (optional)
        """
        if files_to_compress is None:
            if self.potential.export_file is None:
                files_to_compress = self.list_files()
            else:
                files_to_compress = [
                    f for f in self.list_files() if f != self.potential.export_file
                ]
        super().compress(files_to_compress=files_to_compress)

    def collect_output(self, cwd=None):
        """Internal function that parses the output of an atomicrex job

        Args:
            cwd (str, optional): Working directory. Defaults to None.
        """
        # self.input.from_hdf(self._hdf5)
        if cwd is None:
            cwd = self.working_directory
        if self.input.__version__ == "0.1.0":
            filepath = f"{cwd}/atomicrex.out"
        else:
            filepath = f"{cwd}/error.out"

        finished_triggered = False
        params_triggered = False
        dependent_dofs_triggered = False
        structures_triggered = False

        # Allocate numpy arrays for iterations and residual
        # I assume this is better than appending to a list if many iterations are done
        if self.input.fit_algorithm.name == "BFGS":
            residuals = np.zeros(self.input.fit_algorithm.max_iter + 1)
        else:
            residuals = np.zeros(self.input.fit_algorithm.max_iter)

        # Since every step is written out in atomicrex arange can be used.
        # Needs to be adapted when atomicrex output is changed to write only every xth step.
        # Unsinged 32 bit int should be enough or this will overflow anyway in most cases.
        iterations = np.arange(start=1, stop=len(residuals) + 1, dtype=np.uintc)
        iter_index = 0

        with open(filepath, "r") as f:
            final_parameter_lines = []
            final_property_lines = []
            depdendent_dof_lines = []

            for l in f:
                if l.startswith("ERROR"):
                    self.status.aborted = True
                    self.output.error = l

                elif not finished_triggered:
                    if l.startswith("Iterations"):
                        l = l.split()
                        finished_triggered = True
                        self.output.residual = residuals[0:iter_index]
                        self.output.iterations = iterations[0:iter_index]
                    else:
                        l = l.split()
                        try:
                            if l[1] == "iter=":
                                residuals[iter_index] = float(l[-1])
                                iter_index += 1
                        except IndexError:
                            continue

                else:  # if finished_triggered
                    if params_triggered:
                        if not l.startswith("---"):
                            final_parameter_lines.append(l)
                        else:
                            # Collecting lines with final parameters finished, hand over to the potential class
                            self.potential._parse_final_parameters(
                                final_parameter_lines
                            )
                            params_triggered = False

                    elif structures_triggered:
                        if not l.startswith("---"):
                            final_property_lines.append(l)
                        else:
                            # Collecting structure information finished, hand over structures class
                            self.structures._parse_final_properties(
                                final_property_lines
                            )
                            structures_triggered = False

                    elif dependent_dofs_triggered:
                        if not l.startswith("---"):
                            depdendent_dof_lines.append(l)
                        else:
                            self.potential._parse_final_parameters(depdendent_dof_lines)
                            dependent_dofs_triggered = False

                    elif l.startswith("Potential parameters"):
                        # Get the number of dofs
                        n_fit_dofs = int(l.split("=")[1][:-3])
                        params_triggered = True

                    elif l.startswith("Computing"):
                        structures_triggered = True

                    elif l.startswith("Dependent DOFs:"):
                        dependent_dofs_triggered = True
        self.to_hdf()

    def convergence_check(self):
        """
        Internal function, TODO
        find something to reasonably judge convegence
        """
        return True

    def write_input(self, directory=None):
        """Internal function that writes input files

        Args:
            directory ([string], optional): Working directory. Defaults to None.
        """
        if directory is None:
            directory = self.working_directory
        self.input._write_xml_file(directory=directory, job=self)
        self.potential.write_xml_file(directory=directory)
        self.structures.write_xml_file(directory=directory)

    def _executable_activate(self, enforce=False):
        """
        Internal function that sets up and Executable() object
        and finds executables available in pyiron resources/atomicrex/bin

        Args:
            enforce (bool, optional): [description]. Defaults to False.
        """
        if self._executable is None or enforce:
            if len(self.__module__.split(".")) > 1:
                self._executable = Executable(
                    codename=self.__name__,
                    module=self.__module__.split(".")[-2],
                    path_binary_codes=state.settings.resource_paths,
                )
            else:
                self._executable = Executable(
                    codename=self.__name__,
                    path_binary_codes=state.settings.resource_paths,
                )

    # Leftover of the potentials workshop.
    # Maybe this property will be used in unified interface
    # to several fitting codes in the future
    # instead of the potential_as_pd_df function
    @property
    def lammps_potential(self):
        return self.potential_as_pd_df()

    def potential_as_pd_df(self):
        """
        Return the fitted potential as a pandas dataframe,
        which can be used for lammps calculations.
        """
        return self.potential._potential_as_pd_df(job=self)

    #### PotentialFit methods
    def _add_training_data(self, container: TrainingContainer) -> None:
        self.structures.add_training_data(container)

    def _get_training_data(self) -> TrainingStorage:
        return self.structures.get_training_data()

    def _get_predicted_data(self) -> FlattenedStorage:
        return self.structures.get_predicted_data()

    def get_lammps_potential(self) -> pd.DataFrame:
        """
        Return a pyiron compatible dataframe that defines a potential to be used with a Lammps job (or subclass
        thereof).

        Returns:
            DataFrame: contains potential information to be used with a Lammps job.
        """
        return self.potential_as_pd_df()


class Factories:
    """
    Provides conventient acces to other factory classes.
    Functionality to set up an atomicrex job can be found here.
    """

    def __init__(self):
        self.potentials = ARPotFactory()
        self.functions = FunctionFactory()
        self.algorithms = AlgorithmFactory()
