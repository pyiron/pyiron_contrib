# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_base import JobStatus, GenericJob, GenericParameters
from pyiron.atomistics.job.interactivewrapper import InteractiveWrapper, ReferenceJobOutput

__author__ = "Jan Janssen"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - " \
                "Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2018"


class Mixer(InteractiveWrapper):
    """
    Args:
        project (pyiron.project.Project instance):  Specifies the project path among other attributes
        job_name (str): Name of the job

    Attributes:
        input (pyiron.objects.hamilton.md.lammps.Input instance): Instance which handles the input
    """

    def __init__(self, project, job_name):
        super(Mixer, self).__init__(project, job_name)
        self.__name__ = "Mixer"
        self.input = Input()
        self._ref_job_0 = None
        self._ref_job_1 = None
        self.output = MixingOutput(job=self)
        self.interactive_cache = {}
        self.server.run_mode.interactive = True

    @property
    def ref_job(self):
        return self.ref_job_0

    @ref_job.setter
    def ref_job(self, ref_job):
        self.ref_job_0 = ref_job

    @property
    def ref_job_0(self):
        """
        Get the reference job template from which all jobs within the ParallelMaster are generated.

        Returns:
            GenericJob: reference job
        """
        if self._ref_job_0 is not None:
            return self._ref_job_0
        try:
            if isinstance(self[0], GenericJob):
                self._ref_job_0 = self[0]
                self._ref_job_0._job_id = None
                self._ref_job_0._status = JobStatus(db=self.project.db)
                return self._ref_job_0
            else:
                return None
        except IndexError:
            return None

    @ref_job_0.setter
    def ref_job_0(self, ref_job):
        """
        Set the reference job template from which all jobs within the ParallelMaster are generated.

        Args:
            ref_job (GenericJob): reference job
        """
        self.append(ref_job)

    @property
    def ref_job_1(self):
        """
        Get the reference job template from which all jobs within the ParallelMaster are generated.

        Returns:
            GenericJob: reference job
        """
        if self._ref_job_1 is not None:
            return self._ref_job_1
        try:
            if isinstance(self[1], GenericJob):
                self._ref_job_1 = self[1]
                self._ref_job_1._job_id = None
                self._ref_job_1._status = JobStatus(db=self.project.db)
                return self._ref_job_1
            else:
                return None
        except IndexError:
            return None

    @ref_job_1.setter
    def ref_job_1(self, ref_job):
        """
        Set the reference job template from which all jobs within the ParallelMaster are generated.

        Args:
            ref_job (GenericJob): reference job
        """
        if self.ref_job_0 is None:
            raise ValueError('Please assign ref_job_0 before ref_job_1.')
        ref_job.structure = self._ref_job_0.structure
        self.append(ref_job)

    @property
    def structure(self):
        if self.ref_job is not None:
            return self._ref_job_0.structure
        else:
            return None

    @structure.setter
    def structure(self, basis):
        if self.ref_job_0 is not None and self.ref_job_1 is not None:
            self._ref_job_0.structure = basis
            self._ref_job_1.structure = basis
        else:
            raise ValueError('A structure can only be set after both ref_jobs (ref_job_0 and ref_job_1) have been' 
                             'assinged.')

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
        for key in list(set(list(self.ref_job_0.interactive_cache.keys()) +
                            list(self.ref_job_1.interactive_cache.keys()))):
            self.interactive_cache[key] = []
        self.output = MixingOutput(job=self)

    def run_if_interactive(self):
        self.status.running = True
        if self.ref_job_0.server.run_mode.interactive:
            self.ref_job_0.run()
        else:
            self.ref_job_0.run(run_again=True)
        if self.ref_job_1.server.run_mode.interactive:
            self.ref_job_1.run()
        else:
            self.ref_job_1.run(run_again=True)

    def interactive_store_in_cache(self, key, value):
        self.ref_job_0.interactive_cache[key] = value
        self.ref_job_1.interactive_cache[key] = value

    def interactive_close(self):
        self.status.collect = True
        if self.ref_job_0.server.run_mode.interactive:
            self.ref_job_0.interactive_close()
        if self.ref_job_1.server.run_mode.interactive:
            self.ref_job_1.interactive_close()
        self.project.db.item_update(self._runtime(), self.job_id)
        self.status.finished = True

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the ParallelMaster from an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(Mixer, self).from_hdf(hdf=hdf, group_name=group_name)
        self.interactive_initialize_interface()


class Input(GenericParameters):
    """
    class to control the generic input for a Sphinx calculation.

    Args:
        input_file_name (str): name of the input file
        table_name (str): name of the GenericParameters table
    """

    def __init__(self, input_file_name=None, table_name="input"):
        super(Input, self).__init__(input_file_name=input_file_name, table_name=table_name, comment_char="//",
                                    separator_char="=", end_value_char=';')

    def load_default(self):
        """
        Loads the default file content
        """
        file_content = ('lambda = 0.5\n')
        self.load_string(file_content)


class MixingOutput(ReferenceJobOutput):
    def __init__(self, job):
        super(MixingOutput, self).__init__(job=job)
        self._lambda = float(job.input['lambda'])

    @property
    def energy_pot(self):
        return self._lambda * np.array(self._job.ref_job_0.output.energy_pot) + \
               (1-self._lambda) * np.array(self._job.ref_job_1.output.energy_pot)

    @property
    def energy_tot(self):
        return self._lambda * np.array(self._job.ref_job_0.output.energy_tot) + \
               (1 - self._lambda) * np.array(self._job.ref_job_1.output.energy_tot)

    @property
    def forces(self):
        return self._lambda * np.array(self._job.ref_job_0.output.forces) + \
               (1 - self._lambda) * np.array(self._job.ref_job_1.output.forces)

    @property
    def pressures(self):
        return self._lambda * np.array(self._job.ref_job_0.output.pressures) + \
               (1 - self._lambda) * np.array(self._job.ref_job_1.output.pressures)

    @property
    def temperatures(self):
        return self._lambda * np.array(self._job.ref_job_0.output.temperatures) + \
               (1 - self._lambda) * np.array(self._job.ref_job_1.output.temperatures)

