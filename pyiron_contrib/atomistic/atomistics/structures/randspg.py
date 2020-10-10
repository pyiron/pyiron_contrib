# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import os
import posixpath

from pyiron.atomistics.structure.atoms import Atoms
from pyiron_base import GenericParameters, GenericJob
from pyiron.vasp.structure import read_atoms

__author__ = "Jan Janssen"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - " \
                "Computational Materials Design (CM) Department"
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


class RandSpg(GenericJob):
    """
    RandSpg is a program that generates random crystals with specific space groups.
    The user inputs a specific composition and space group to be generated. The
    algorithm then proceeds by finding all combinations of Wyckoff positions that
    satisfy the composition. It then randomly selects a combination of Wyckoff
    positions for the space group, generates random coordinates for variables
    in the Wyckoff positions, and places atoms in those sites. It ensures that any
    constraints the user placed on the system (lattice constraints including
    min/max volume, minimum interatomic distances, specific Wyckoff positions for
    specific atoms, etc.) are satisfied.
    Code: https://github.com/xtalopt/randSpg
    Paper: https://www.sciencedirect.com/science/article/pii/S0010465516303848
    """
    def __init__(self, project, job_name):
        super(RandSpg, self).__init__(project, job_name)
        self.__version__ = "0.1"
        self.__name__ = "RandSpg"
        self._lst_of_struct = []
        self.input = ExampleInput()
        self._executable_activate()

    @property
    def list_of_structures(self):
        return self._lst_of_struct

    def list_structures(self):
        if self.status.finished:
            return self._lst_of_struct
        else:
            return []

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        self.input.read_only = True

    # define routines that create all necessary input files
    def write_input(self):
        """
        Call routines that generate the codespecifc input files
        """
        self.input.write_file(file_name="randspg.in", cwd=self.working_directory)

    # define routines that collect all output files
    def collect_output(self):
        """
        Parse the output files of the example job and store the results in the HDF5 File.
        """
        self.collect_output_log()

    def collect_output_log(self, dir_name="randSpgOut"):
        """
        general purpose routine to extract output from logfile

        Args:
            file_name (str): output.log - optional
        """
        self._lst_of_struct = [[file_name.replace('-', '_'),
                                read_atoms(filename=posixpath.join(self.working_directory, dir_name, file_name))]
                               for file_name in os.listdir(posixpath.join(self.working_directory, dir_name))]
        for structure_name, structure in self._lst_of_struct:
            with self.project_hdf5.open("output/structures/" + structure_name) as h5:
                structure.to_hdf(h5)

    def collect_logfiles(self):
        pass

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the ExampleJob object in the HDF5 File

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(RandSpg, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the ExampleJob object in the HDF5 File

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(RandSpg, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)
        self._lst_of_struct = []
        with self.project_hdf5.open("output/structures") as hdf5_output:
            structure_names = hdf5_output.list_groups()
        for group in structure_names:
            with self.project_hdf5.open("output/structures/" + group) as hdf5_output:
                self._lst_of_struct.append([group, Atoms().from_hdf(hdf5_output)])


class ExampleInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(ExampleInput, self).__init__(input_file_name=input_file_name,
                                           table_name="randspg_in",
                                           comment_char="#",
                                           separator_char='=')

    def load_default(self):
        """
        Loading the default settings for the input file.
        """
        input_str = '''\
#Inputfile
composition = Mg4Al2
spacegroups = 1-8, 10, 25, 28, 30-32
latticeMins =  3.0,  3.0,  3.0,  60.0,  60.0,  60.0
latticeMaxes = 10.0, 10.0, 10.0, 120.0, 120.0, 120.0
minVolume = 450
maxVolume = 500
numOfEachSpgToGenerate = 3
setMinRadii = 0.3
scalingFactor = 0.5
maxAttempts = 100
outputDir = randSpgOut
verbosity = r
'''
        self.load_string(input_str)
