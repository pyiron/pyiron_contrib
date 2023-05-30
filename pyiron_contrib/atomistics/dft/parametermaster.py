# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import pandas
from pyiron.atomistics.master.parallel import AtomisticParallelMaster
from pyiron_base import JobGenerator

__author__ = "Jan Janssen"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "production"
__date__ = "Sep 1, 2017"


class ParameterJobGenerator(JobGenerator):
    """
    Job Generator for the ParameterMaster
    """

    @property
    def parameter_list(self):
        """
        List of parameters - a combination of energy cut off and kpoint mesh pairs

        Returns:
            list: of pairs of energy cut off and kpoint mesh
        """
        return [
            [encut, kpoint_mesh]
            for encut, kpoint_mesh in zip(
                self._job.iteration_frame.ENCUT, self._job.iteration_frame.KPOINT_MESH
            )
        ]

    @staticmethod
    def job_name(parameter):
        """
        Generate job name for a given job based on energy cut off and kpoint mesh
        Args:
            parameter (list): pair of energy cut off and kpoint mesh

        Returns:
            str: job name
        """
        return (
            "job_encut_"
            + str(parameter[0]).replace(".", "_")
            + "_kpoints_"
            + str(parameter[1][0])
            + "_"
            + str(parameter[1][1])
            + "_"
            + str(parameter[1][2])
        )

    def modify_job(self, job, parameter):
        """
        Modify a given job by setting the energy cut off and the kpoint mesh

        Args:
            job (pyiron.dft.job.generic.GenericDFTJob): DFT quantum engine
            parameter (list): pair of energy cut off and kpoint mesh

        Returns:
            pyiron.dft.job.generic.GenericDFTJob: updated job object
        """
        job.set_encut(parameter[0])
        job.set_kpoints(parameter[1])
        return job


class ParameterMaster(AtomisticParallelMaster):
    def __init__(self, project, job_name):
        """
        Parameter master to calculate energies and forces for a given list of energy cut off and kpoint mesh pairs.
        The parameter master is typically used to validate the convergence of a DFT calculation

        Args:
            project (ProjectHDFio): ProjectHDFio instance which points to the HDF5 file the job is stored in
            job_name (str): name of the job, which has to be unique within the project
        """
        super(ParameterMaster, self).__init__(project, job_name)
        self.__name__ = "ParameterMaster"
        self.__version__ = "0.0.1"
        self._job_generator = ParameterJobGenerator(self)
        self.iteration_frame = pandas.DataFrame({"ENCUT": [], "KPOINT_MESH": []})

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the ParameterMaster in an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(ParameterMaster, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            hdf5_input["dataframe"] = self.iteration_frame.to_dict(orient="list")

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the ParameterMaster from an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(ParameterMaster, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.iteration_frame = pandas.DataFrame(hdf5_input["dataframe"])

    def collect_output(self):
        """
        The ParameterMaster does not store any specitic output, it is recommended to use a pyiron table instead.
        """
        pass
