# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_base import JobStatus, GenericJob, GenericParameters
from pyiron_atomistics.atomistics.job.interactivewrapper import (
    InteractiveWrapper,
    ReferenceJobOutput,
)
from pyiron_mpie.interactive.pipe_forward import pipe_forwarding

__author__ = "Osamu Waseda"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Osamu Waseda"
__email__ = "waseda@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2018"


class Intermixer(InteractiveWrapper):
    """
    Args:
        project (pyiron.project.Project instance):  Specifies the project path among other attributes
        job_name (str): Name of the job

    Attributes:
        input (pyiron.objects.hamilton.md.lammps.Input instance): Instance which handles the input
    """

    def __init__(self, project, job_name):
        super(Intermixer, self).__init__(project, job_name)
        self.__name__ = "Intermixer"
        self.input = Input()
        self._ref_job_all = []
        self.output = IntermixingOutput(job=self)
        self.server.run_mode.interactive = True
        self.interactive_cache = {}
        self.weights = None

    @property
    def _n_jobs(self):
        return len(self._ref_job_all) + len(self)

    @property
    def ref_job(self):
        if len(self._ref_job_all) > 0:
            return self._ref_job_all[0]
        try:
            if isinstance(self[0], GenericJob):
                self._ref_job_all.append(self[0])
                return self._ref_job_all[0]
            else:
                return None
        except IndexError:
            return None

    @ref_job.setter
    def ref_job(self, ref_job: list):
        if not isinstance(ref_job, list):
            raise TypeError("ref_job must be a list of jobs")
        self._ref_job_all.extend(ref_job)
        for job in self._ref_job_all:
            job.structure.positions = self._ref_job_all[0].structure.positions
            job.structure.cell = self._ref_job_all[0].structure.cell
            self.append(job)

    @property
    def ref_job_all(self):
        return self._ref_job_all

    @property
    def structure(self):
        if self._n_jobs > 0:
            return self.ref_job.structure
        else:
            return None

    @structure.setter
    def structure(self, basis):
        for job in self._ref_job_all:
            job.structure.positions = basis.positions
            job.structure.cell = basis.cell

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        self.input.read_only = True

    def write_input(self):
        pass

    def _write_run_wrapper(self, debug=False):
        pass

    def interactive_initialize_interface(self):
        for key in self.ref_job.interactive_cache.keys():
            self.interactive_cache[key] = []
        self.output = IntermixingOutput(job=self)

    def ref_job_initialize(self):
        self._ref_job_all = []
        while len(self._job_name_lst) > 0:
            self._ref_job_all.append(self.pop(-1))
            self._ref_job_all[-1].structure.positions = self._ref_job_all[
                0
            ].structure.positions
            self._ref_job_all[-1].structure.cell = self._ref_job_all[0].structure.cell
            if self._job_id is not None and self._ref_job_all[-1]._master_id is None:
                self._ref_job_all[-1]._job_id = self.job_id

    def run_if_interactive(self):
        if not self.status.initialized:
            self.ref_job_initialize()
        self.status.running = True
        for i in range(self._n_jobs):
            if (
                self.ref_job_all[i].server.run_mode.interactive
                or self.ref_job_all[i].server.run_mode.interactive_non_modal
            ):
                self.ref_job_all[i].run()
            else:
                self.ref_job_all[i].run(run_again=True)
        for i in range(self._n_jobs):
            if self.ref_job_all[i].server.run_mode.interactive_non_modal:
                self.ref_job_all[i].interactive_fetch()

    def interactive_store_in_cache(self, key, value):
        for i in range(self._n_jobs):
            self.ref_job_all[i].interactive_cache[key] = value

    def interactive_close(self):
        self.status.collect = True
        for i in range(self._n_jobs):
            if (
                self.ref_job_all[i].server.run_mode.interactive
                or self.ref_job_all[i].server.run_mode.interactive_non_modal
            ):
                self.ref_job_all[i].interactive_close()
        self.project.db.item_update(self._runtime(), self.job_id)
        self.status.finished = True

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the ParallelMaster from an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(Intermixer, self).from_hdf(hdf=hdf, group_name=group_name)
        self.interactive_initialize_interface()


class Input(GenericParameters):
    """
    class to control the generic input for a Sphinx calculation.

    Args:
        input_file_name (str): name of the input file
        table_name (str): name of the GenericParameters table
    """

    def __init__(self, input_file_name=None, table_name="input"):
        super(Input, self).__init__(
            input_file_name=input_file_name,
            table_name=table_name,
            comment_char="//",
            separator_char="=",
            end_value_char=";",
        )

    def load_default(self):
        """
        Loads the default file content
        """
        file_content = "beta = 0\n"
        self.load_string(file_content)


class IntermixingOutput(ReferenceJobOutput):
    def __init__(self, job):
        super(IntermixingOutput, self).__init__(job=job)

    @property
    def n_jobs(self):
        return self._job._n_jobs

    @property
    def energy_pot(self):
        return np.average(
            [self._job._ref_job_all[i].output.energy_pot for i in range(self.n_jobs)],
            axis=0,
            weights=self.boltzmann_weight * self.weights,
        )

    @property
    def energy_tot(self):
        return np.average(
            [self._job._ref_job_all[i].output.energy_tot for i in range(self.n_jobs)],
            axis=0,
            weights=self.boltzmann_weight * self.weights,
        )

    @property
    def forces(self):
        return np.average(
            [self._job._ref_job_all[i].output.forces for i in range(self.n_jobs)],
            axis=0,
            weights=self.boltzmann_weight * self.weights,
        )

    @property
    def pressures(self):
        return np.average(
            [self._job._ref_job_all[i].output.pressures for i in range(self.n_jobs)],
            axis=0,
            weights=self.boltzmann_weight * self.weights,
        )

    @property
    def temperatures(self):
        return np.average(
            [self._job._ref_job_all[i].output.temperatures for i in range(self.n_jobs)],
            axis=0,
            weights=self.boltzmann_weight * self.weights,
        )

    @property
    def weights(self):
        if self._job.weights is None or len(self._job.weights) != self.n_jobs:
            return np.ones(self.n_jobs)
        return np.array(self._job.weights)[::-1]

    @property
    def boltzmann_weight(self):
        if self._job.input["beta"] > 0:
            mean_E = np.mean(
                [
                    self._job._ref_job_all[i].output.energy_pot
                    for i in range(self.n_jobs)
                ],
                axis=0,
            )
            return np.exp(
                -np.array(
                    [
                        self._job._ref_job_all[i].output.energy_pot - mean_E
                        for i in range(self.n_jobs)
                    ]
                )
                / len(self._job.structure)
                * self._job.input["beta"]
            )
        else:
            return np.ones(self.n_jobs)
