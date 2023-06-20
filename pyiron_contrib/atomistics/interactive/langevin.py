# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from ase import units
from ase.md.langevin import Langevin
from pyiron_base import GenericParameters
from pyiron.gpaw.pyiron_ase import AseAdapter
from pyiron.atomistics.job.interactivewrapper import InteractiveWrapper

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


class LangevinAse(InteractiveWrapper):
    """
    Reference implementation of a thermostat implemented in ASE. While we have multiple implementations of the
    Langevin thermostat already, this one is using the ASE interface and the Langevin thermostat implemented in ASE.
    It is therefore a perfect example how other thermostats implemented in ASE can be used inside pyiron.

    Args:
        project (ProjectHDFio): points to the HDF5 file the job is stored in
        job_name (str): name of the job, which has to be unique within the project

    Attributes:
        input (pyiron.objects.hamilton.md.lammps.Input): handle the input
    """

    def __init__(self, project, job_name):
        super(LangevinAse, self).__init__(project, job_name)
        self.__name__ = "LangevinAse"
        self.input = Input()
        self._fast_mode = False

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        super(LangevinAse, self).set_input_to_read_only()
        self.input.read_only = True

    def write_input(self):
        """
        No input is written when using the ASE interface
        """
        pass

    def _write_run_wrapper(self, debug=False):
        """
        No wrapper is written for the ASE interface

        Args:
            debug (bool): the debug flag is ignored
        """
        pass

    def run_static(self):
        """
        Static run function, which uses the ASEAdapter to connect the pyiron interactive reference job with the
        Langevin thermostat implemented in ASE, by setting the ASEAdapter as a replacement of the ASE atoms object.
        """
        self.status.running = True
        self.ref_job_initialize()
        aseadapter = AseAdapter(self.ref_job, self._fast_mode)
        langevin = Langevin(
            atoms=aseadapter,
            timestep=self.input["time_step"] * units.fs,
            temperature=self.input["temperature"] * units.kB,
            friction=self.input["friction"],
            fixcm=True,
        )
        langevin.run(self.input["ionic_steps"])
        self.status.collect = True
        aseadapter.interactive_close()
        self._finish_job()


class Input(GenericParameters):
    """
    class to control the generic input for a Sphinx calculation.

    Args:
        input_file_name (str): name of the input file
        table_name (str): name of the GenericParameters table
    """

    def __init__(self, input_file_name=None, table_name="input"):
        super(Input, self).__init__(
            input_file_name=input_file_name,
            table_name=table_name,
            comment_char="#",
            separator_char="=",
            end_value_char=";",
        )

    def load_default(self):
        """
        Loads the default file content
        """
        file_content = (
            "ionic_steps = 100\n"
            "temperature = 1500\n"
            "time_step = 1\n"
            "friction = 0.002\n"
            "fix_center_of_mass = True\n"
        )
        self.load_string(file_content)
