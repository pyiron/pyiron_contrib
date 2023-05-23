# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import os
import posixpath
import shutil

from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure
from pyiron_base import GenericParameters, GenericJob, deprecate
from pyiron_atomistics.vasp.structure import read_atoms

__author__ = "Jan Janssen"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


class RandSpg(GenericJob, HasStructure):
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
        self._structure_storage = StructureStorage()
        self.input = ExampleInput()
        self._executable_activate()
        self._compress_by_default = True

    @property
    def list_of_structures(self):
        return list(
            (
                self._structure_storage.get_array("identifier", i),
                self._structure_storage.get_structure(i),
            )
            for i in range(len(self._structure_storage))
        )

    @deprecate("Use get_structure()/iter_structures()/list_of_structures instead!")
    def list_structures(self):
        if self.status.finished:
            return self.list_of_structures
        else:
            return []

    def _number_of_structures(self):
        return self._structure_storage._number_of_structures()

    def _translate_frame(self, frame):
        return self._structure_storage._translate_frame(frame)

    def _get_structure(self, frame, wrap_atoms=True):
        return self._structure_storage._get_structure(
            frame=frame, wrap_atoms=wrap_atoms
        )

    def validate_ready_to_run(self):
        if shutil.which("randSpg") is None:
            raise ValueError(
                "randSpg binary not installed; install with `conda install -c conda-forge randspg"
            )

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
        self._structure_storage = StructureStorage()  # reset saved structures
        dir_path = posixpath.join(self.working_directory, dir_name)
        for file_name in os.listdir(dir_path):
            self._structure_storage.add_structure(
                read_atoms(filename=posixpath.join(dir_path, file_name)),
                identifier=file_name.replace("-", "_"),
            )
        with self.project_hdf5.open("output") as hdf5_output:
            self._structure_storage.to_hdf(hdf5_output)

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
        if self.status.finished:
            with self.project_hdf5.open("output") as hdf5_output:
                self._structure_storage.from_hdf(hdf5_output)


class ExampleInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(ExampleInput, self).__init__(
            input_file_name=input_file_name,
            table_name="randspg_in",
            comment_char="#",
            separator_char="=",
        )

    def load_default(self):
        """
        Loading the default settings for the input file.
        """
        input_str = """\
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
"""
        self.load_string(input_str)

    def write_file(self, file_name, cwd=None):
        """
        Write GenericParameters to input file

        Args:
            file_name (str): name of the file, either absolute (then cwd must be None) or relative
            cwd (str): path name (default: None)
        """
        if cwd is not None:
            file_name = posixpath.join(cwd, file_name)

        with open(file_name, "w") as f:
            for line in self.get_string_lst():
                f.write(line.replace("(", "").replace(")", ""))
