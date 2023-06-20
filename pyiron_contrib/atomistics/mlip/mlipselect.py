# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
from pyiron_contrib.atomistics.mlip.mlip import Mlip, read_cgfs

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


class MlipSelect(Mlip):
    def __init__(self, project, job_name):
        super(MlipSelect, self).__init__(project, job_name)
        self.__version__ = None
        self.__name__ = "MlipSelect"
        self._executable_activate()
        del self.input["min_dist"]
        del self.input["max_dist"]
        del self.input["iteration"]
        del self.input["energy-weight"]
        del self.input["force-weight"]
        del self.input["stress-weight"]
        self._command_line._delete_line(0)
        self._command_line._delete_line(0)

    def validate_ready_to_run(self):
        if len(self.restart_file_list) == 0:
            raise ValueError()

    def write_input(self):
        species_count = self._write_test_set(
            file_name="testing.cfg", cwd=self.working_directory
        )
        self._copy_potential(
            species_count, file_name="start.mtp", cwd=self.working_directory
        )
        self._command_line[0] = self._command_line[0].replace(
            "Trained.mtp_", "start.mtp"
        )
        self._command_line.write_file(file_name="mlip.sh", cwd=self.working_directory)

    def collect_output(self):
        file_name = os.path.join(self.working_directory, "diff.cfg")
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
