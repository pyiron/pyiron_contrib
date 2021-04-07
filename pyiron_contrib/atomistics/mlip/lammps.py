# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
from pyiron.lammps.base import Input
from pyiron.lammps.interactive import LammpsInteractive
from pyiron_contrib.atomistic.mlip.mlip import read_cgfs
from pyiron_base import GenericParameters

__author__ = "Jan Janssen"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - " \
                "Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2018"


class LammpsMlip(LammpsInteractive):
    def __init__(self, project, job_name):
        super(LammpsMlip, self).__init__(project, job_name)
        self.input = MlipInput()
        self.__name__ = "LammpsMlip"
        self.__version__ = None  # Reset the version number to the executable is set automatically
        self._executable = None
        self._executable_activate()

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        super(LammpsMlip, self).set_input_to_read_only()
        self.input.mlip.read_only = True

    def write_input(self):
        super(LammpsMlip, self).write_input()
        if self.input.mlip['mlip:load-from'] == 'auto':
            self.input.mlip['mlip:load-from'] = os.path.basename(self.potential['Filename'][0][0])
        self.input.mlip.write_file(file_name="mlip.ini", cwd=self.working_directory)

    def enable_active_learning(self):
        self.input.mlip.load_string("""\
abinitio void
mlip mtpr
mlip:load-from Trained.mtp_
calculate-efs TRUE
fit FALSE
select TRUE
select:site-en-weight 0.0
select:energy-weight 1.0
select:force-weight 0.0
select:stress-weight 0.0
select:threshold-init 1e-5
select:threshold 2.0
select:threshold-swap 1.000001
select:threshold-break 5.0
select:save-selected selected.cfg
select:save-state selection.mvs
select:load-state state.mvs
select:efs-ignored FALSE
select:log selection.log
write-cfgs:skip 0
log lotf.log""")

    def collect_output(self):
        super(LammpsMlip, self).collect_output()
        if 'select:save-selected' in self.input.mlip._dataset['Parameter']:
            file_name = os.path.join(self.working_directory, self.input.mlip['select:save-selected'])
            if os.path.exists(file_name):
                cell, positions, forces, stress, energy, indicies, grades, jobids, timesteps = read_cgfs(file_name=file_name)
                with self.project_hdf5.open("output/mlip") as hdf5_output:
                    hdf5_output['forces'] = forces
                    hdf5_output['energy_tot'] = energy
                    hdf5_output['pressures'] = stress
                    hdf5_output['cells'] = cell
                    hdf5_output['positions'] = positions
                    hdf5_output['indicies'] = indicies


class MlipInput(Input):
    def __init__(self):
        self.mlip = MlipParameter()
        super(MlipInput, self).__init__()

    def to_hdf(self, hdf5):
        """
        
        Args:
            hdf5:
        Returns:
        """
        with hdf5.open("input") as hdf5_input:
            self.mlip.to_hdf(hdf5_input)
        super(MlipInput, self).to_hdf(hdf5)

    def from_hdf(self, hdf5):
        """
        
        Args:
            hdf5:
        Returns:
        """
        with hdf5.open("input") as hdf5_input:
            self.mlip.from_hdf(hdf5_input)
        super(MlipInput, self).from_hdf(hdf5)


class MlipParameter(GenericParameters):
    def __init__(self, separator_char=' ', comment_char='#', table_name="mlip_inp"):
        super(MlipParameter, self).__init__(separator_char=separator_char, comment_char=comment_char, table_name=table_name)
        
    def load_default(self, file_content=None):
        if file_content is None:
            file_content = '''\
abinitio void
mlip mtpr
mlip:load-from auto
calculate-efs TRUE
fit FALSE
select FALSE
'''
        self.load_string(file_content)

