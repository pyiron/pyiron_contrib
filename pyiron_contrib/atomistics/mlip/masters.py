# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_atomistics.atomistics.master.parallel import ParallelMaster
from pyiron_base import JobGenerator

__author__ = "Jan Janssen"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2018"


def random_displacement(basis, displacement=0.05):
    basis_copy = basis.copy()
    random_vec = (np.random.random([len(basis_copy), 3]) - 0.5) * displacement
    basis_copy.positions += random_vec
    return basis_copy


class DisplacementJobGenerator(JobGenerator):
    @property
    def parameter_list(self):
        """

        Returns:
            (list)
        """
        return list(range(self._job.input["num_points"]))

    @staticmethod
    def job_name(parameter):
        return "rsdis_job_" + str(parameter)

    def modify_job(self, job, parameter):
        job.structure = random_displacement(
            basis=job.structure, displacement=float(self._job.input["displacement"])
        )
        return job


class RandomDisMaster(ParallelMaster):
    """

    Args:
        project:
        job_name:
    """

    def __init__(self, project, job_name):
        super(RandomDisMaster, self).__init__(project, job_name)
        self.__name__ = "RandomDisMaster"
        self.__version__ = "0.0.1"
        self.input["num_points"] = (100, "number of points")
        self.input["displacement"] = (0.05, "displacement")
        self._job_generator = DisplacementJobGenerator(self)

    def collect_output(self):
        """

        Returns:

        """
        pass


class RandomMDJobGenerator(JobGenerator):
    @property
    def parameter_list(self):
        """

        Returns:
            (list)
        """
        return list(
            enumerate([int(job_id) for job_id in self._job.input["job_ids"].split()])
        )

    @staticmethod
    def job_name(parameter):
        return "md_job_" + str(parameter[1]) + "_" + str(parameter[0])

    def modify_job(self, job, parameter):
        job.structure = self._job.project.load(parameter[1]).structure
        job.server.accept_crash = True
        return job


class RandomMDMaster(ParallelMaster):
    """

    Args:
        project:
        job_name:
    """

    def __init__(self, project, job_name):
        super(RandomMDMaster, self).__init__(project, job_name)
        self.__name__ = "RandomMDMaster"
        self.__version__ = "0.0.1"
        self.input["num_points"] = (100, "number of points")
        self._job_generator = RandomMDJobGenerator(self)

    @property
    def structure_job_id_lst(self):
        return [int(job_id) for job_id in self.input["job_ids"].split()]

    @structure_job_id_lst.setter
    def structure_job_id_lst(self, job_id_lst):
        self.input["job_ids"] = " ".join([str(job_id) for job_id in job_id_lst])

    def run_static(self):
        if "job_ids" not in self.input._dataset["Parameter"]:
            self.input["job_ids"] = " ".join(
                [
                    str(job_id)
                    for job_id in np.random.choice(
                        self.structure_job_id_lst, self.input["num_points"]
                    )
                ]
            )
            self.to_hdf()
        super(RandomMDMaster, self).run_static()

    def collect_output(self):
        """

        Returns:

        """
        pass
