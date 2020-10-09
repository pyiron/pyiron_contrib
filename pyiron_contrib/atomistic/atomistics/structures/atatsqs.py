# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
import os 
import numpy as np
from pyiron.atomistics.structure.atoms import Atoms
from pyiron_base import GenericJob, GenericParameters

__author__ = "Jan Janssen"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - " \
                "Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


class ATATsqs(GenericJob):
    def __init__(self, project, job_name):
        super(ATATsqs, self).__init__(project, job_name)
        self.__version__ = "0.1"
        self.__name__ = "ATATsqs"
        self.structure_file = RndStr()
        self.command_line_options = CommandLine()
        self._structure = None

    @property 
    def structure(self):
        if self._structure is None: 
            with self.project_hdf5.open('output') as hdf5_output:
                if 'structure' in hdf5_output.list_groups(): 
                    self._structure = Atoms().from_hdf(hdf5_output)
                else: 
                    return None
        return self._structure

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        self.command_line_options.read_only = True
        self.structure_file.read_only = True

    def write_input(self):
        self.structure_file.write_file(file_name='rndstr.in', cwd=self.working_directory)
        self.command_line_options.write_file(file_name='sqs.sh', cwd=self.working_directory)

    def collect_output(self):
        number_of_pointd, diameter, correlations_of_the_best_sqs, target_disordered_state_correlation, difference = self.best_correlation(file_name='bestcorr.out', cwd=self.working_directory)
        basis = self.best_sqs(file_name='bestsqs.out', cwd=self.working_directory)
        with self.project_hdf5.open('output') as hdf5_output:
            hdf5_output['number_of_pointd'] = number_of_pointd
            hdf5_output['diameter'] = diameter
            hdf5_output['correlations_of_the_best_sqs'] = correlations_of_the_best_sqs
            hdf5_output['target_disordered_state_correlation'] = target_disordered_state_correlation
            hdf5_output['difference'] = difference
            basis.to_hdf(hdf5_output)

    def collect_logfiles(self):
        pass
    
    def best_correlation(self, file_name='bestcorr.out', cwd=None):
        if cwd is not None: 
            file_name = os.path.join(cwd, file_name)
        with open(file_name) as f:
            lines = f.readlines()
        number_of_pointd, diameter, correlations_of_the_best_sqs, target_disordered_state_correlation, difference = list(zip(*[[float(f) if ind != 0 else int(f) for ind, f in enumerate(l.split())] for l in lines if len(l.split()) > 2]))
        return number_of_pointd, diameter, correlations_of_the_best_sqs, target_disordered_state_correlation, difference 

    def best_sqs(self, file_name='bestsqs.out', cwd=None):
        if cwd is not None:
            file_name = os.path.join(cwd, file_name)
        with open(file_name) as f:
            lines = f.readlines()
        cell = np.array([[float(f) for f in l.split()] for l in lines[0:3]])
        super_cell = np.array([[float(f) for f in l.split()] for l in lines[3:6]])
        x_pos, y_pos, z_pos, elements = zip(*[[float(f) if ind != 3 else f 
                                               for ind, f in enumerate(l.split())] 
                                              for l in lines[6:]])
        positions = list(zip(x_pos, y_pos, z_pos))
        return Atoms(elements=np.array(elements), 
                     positions=np.dot(np.array(positions), cell), 
                     cell=np.dot(super_cell, cell))


class RndStr(GenericParameters):
    def __init__(self, input_file_name=None, **qwargs):
        super(RndStr, self).__init__(input_file_name=input_file_name,
                                     table_name="randstr",
                                     comment_char="#",
                                     val_only=True)

    def load_default(self, file_content=None):
        if file_content is None:
            file_content = '''\
3.8 3.8 3.8 90 90 90
0   0.5 0.5
0.5 0   0.5
0.5 0.5 0
0 0 0 Cu=0.5,Au=0.5
'''
        self.load_string(file_content)


class CommandLine(GenericParameters):
    def __init__(self, input_file_name=None, **qwargs):
        super(CommandLine, self).__init__(input_file_name=input_file_name,
                                          table_name="cmd",
                                          comment_char="#",
                                          val_only=True)

    def load_default(self, file_content=None):
        if file_content is None:
            file_content = '''\
corrdump -l=rndstr.in -ro -noe -nop -clus -2=5.0 -3=5.0
getclus
mcsqs -n 16
'''
        self.load_string(file_content)