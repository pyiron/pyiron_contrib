"""
Pyiron interface to atomicrex
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import posixpath
import os
import subprocess

from pyiron_base import GenericJob, Settings, PyironFactory, Executable

from pyiron_contrib.atomistic.atomicrex.general_input import GeneralARInput, AlgorithmFactory
from pyiron_contrib.atomistic.atomicrex.structure_list import ARStructureList
from pyiron_contrib.atomistic.atomicrex.potential_factory import ARPotFactory
from pyiron_contrib.atomistic.atomicrex.output import Output
from pyiron_contrib.atomistic.atomicrex.function_factory import FunctionFactory



s = Settings()

## Class defined for future addition of other codes
## Not sure which functionality (if any) can be extracted yet, but a similar pattern is followed in other pyiron modules
class PotentialFittingBase(GenericJob):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)


class Atomicrex(PotentialFittingBase):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)

        self.__name__ = "atomicrex"
        self.__version__ = (
            None
        )
        self._executable_activate(enforce=True)

        s.publication_add(self.publication)
        self.input = GeneralARInput()
        self.potential = None
        self.structures = ARStructureList()
        ## temprorary set working directory manually before full pyiron integration
        self._working_directory = None
        self.output = Output()
        self.factories = Factories()
        

    def to_hdf(self, hdf=None, group_name=None):
        if hdf == None:
            hdf = self.project_hdf5
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(hdf=hdf)
        self.potential.to_hdf(hdf=hdf)
        self.structures.to_hdf(hdf=hdf)
        return

    def from_hdf(self, hdf=None, group_name=None):
        if hdf == None:
            hdf = self.project_hdf5
        self.input.from_hdf()
        self.potential.from_hdf()
        self.structures.from_hdf()
        return


    @property
    def publication(self):
        return {
            "atomicrex": {
                "atomicrex": {
                    "title": "Atomicrex-a general purpose tool for the construction of atomic interaction models",
                    "journal": "Modelling and Simulation in Materials Science and Engineering",
                    "volume": "25",
                    "number": "5",
                    "year": "2017",
                    "issn": "0965-0393",
                    "doi": "10.1088/1361-651X/aa6ecf",
                    "url": "https://doi.org/10.1088%2F1361-651x%2Faa6ecf",
                    "author": ["Alexander Stukowski", "Erik Fransson", "Markus Mock", "Paul Erhart"],
                }
            }
        }
    
    def collect_output(self, cwd=None):
        #self.input.from_hdf(self._hdf5)
        if cwd is None:
            cwd = self.working_directory

        filepath = f"{cwd}/atomicrex.out"
        with open(filepath) as f:
            lines = f.readlines()
        l_index_params = None
        l_index_complete = None
        for i, line in enumerate(lines):
            if line.startswith("ERROR"):
                self.status.aborted=True
                self.output.error = line
                return
            if line.startswith("Potential parameters"):
                l_index_params = i
            if line.startswith("Fitting process complete."):
                l_index_complete = i
                struct_lines = self.output._get_structure_lines(l_index_complete, lines)
                parameter_lines = self.output._get_parameter_lines(l_index_params, lines)
                self.structures._parse_final_properties(struct_lines)
                self.potential._parse_final_parameters(parameter_lines)
                break


    def convergence_check(self):
        return
    
    def write_input(self, directory=None):
        if directory is None:
            directory = self.working_directory
        self.input._write_xml_file(directory = directory)
        self.potential.write_xml_file(directory = directory)
        self.structures.write_xml_file(directory = directory)
    
    def _executable_activate(self, enforce=False):
        if self._executable is None or enforce:
            if len(self.__module__.split(".")) > 1:
                self._executable = Executable(
                    codename=self.__name__,
                    module=self.__module__.split(".")[-2],
                    path_binary_codes=s.resource_paths,
                )
            else:
                self._executable = Executable(
                    codename=self.__name__, path_binary_codes=s.resource_paths
                )


    """
    def run(self):
        cwd = self.working_directory
        self._create_working_directory()
        self.write_input(directory=cwd)
        with open(f"{cwd}/atomicrex.out", "w") as f:
            subprocess.run(["atomicrex", "main.xml"], stdout=f, stderr=f, text=True, cwd=cwd)
    """

class Factories:
    def __init__(self):
        self.potentials = ARPotFactory()
        self.functions = FunctionFactory()
        self.algorithms = AlgorithmFactory()
