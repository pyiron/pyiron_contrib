# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import shutil
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob
from pyiron_contrib.atomistics.mlip.mlip import write_cfg, read_cgfs
from pyiron_base import GenericParameters

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


class MlipJob(AtomisticGenericJob):
    def __init__(self, project, job_name):
        super(MlipJob, self).__init__(project, job_name)
        self.__name__ = "MlipJob"
        self.__version__ = None
        self._executable_activate()
        self.input = MlipParameter()

    @property
    def potential(self):
        return self.input["mlip:load-from"]

    @potential.setter
    def potential(self, potential_filename):
        self.input["mlip:load-from"] = potential_filename

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        super(MlipJob, self).set_input_to_read_only()
        self.input.read_only = True

    def to_hdf(self, hdf=None, group_name=None):
        super(MlipJob, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None, group_name=None):
        super(MlipJob, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)

    def write_input(self):
        write_cfg(
            file_name=os.path.join(self.working_directory, "structure.cfg"),
            indices_lst=[self.structure.indices],
            position_lst=[self.structure.positions],
            cell_lst=[self.structure.cell],
            forces_lst=None,
            energy_lst=None,
            track_lst=None,
            stress_lst=None,
        )
        shutil.copyfile(
            self.input["mlip:load-from"],
            os.path.join(self.working_directory, "potential.mtp"),
        )
        self.input["mlip:load-from"] = "potential.mtp"
        self.input.write_file(file_name="mlip.ini", cwd=self.working_directory)

    def collect_output(self):
        file_name = os.path.join(self.working_directory, "structurebyMTP.cfg")
        (
            cell,
            positions,
            forces,
            stress,
            energy,
            indicies,
            grades,
            jobids,
            timesteps,
        ) = read_cgfs(file_name=file_name)
        with self.project_hdf5.open("output") as hdf5_output:
            hdf5_output["forces"] = forces
            hdf5_output["energy_tot"] = energy
            hdf5_output["pressures"] = stress
            hdf5_output["cells"] = cell
            hdf5_output["positions"] = positions
            hdf5_output["indicies"] = indicies


class MlipParameter(GenericParameters):
    def __init__(self, separator_char=" ", comment_char="#", table_name="mlip_inp"):
        super(MlipParameter, self).__init__(
            separator_char=separator_char,
            comment_char=comment_char,
            table_name=table_name,
        )
        self._bool_dict = {True: "TRUE", False: "FALSE"}

    def load_default(self, file_content=None):
        if file_content is None:
            file_content = """\
abinitio void
mlip mtpr
mlip:load-from auto
calculate-efs TRUE
write-cfgs structurebyMTP.cfg
"""
        self.load_string(file_content)
