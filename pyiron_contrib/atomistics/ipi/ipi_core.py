# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.lammps.interactive import LammpsInteractive
from pyiron_atomistics.lammps.base import Input
from pyiron_base.storage.datacontainer import DataContainer

import numpy as np
import subprocess
from shutil import copy
from os.path import isdir

__author__ = "Raynol Dsouza"
__copyright__ = (
    "Copyright 2022, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "0.0"
__maintainer__ = "Raynol Dsouza"
__email__ = "dsouza@mpie.de"
__status__ = "development"
__date__ = "Jan 18, 2023"


class IPiCore(LammpsInteractive):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input = Input()
        self.custom_input = DataContainer(table_name="pimd_input")
        self.custom_output = DataContainer(table_name="pimd_output")
        self._templates_directory = None

    @property
    def templates_directory(self):
        return self._templates_directory

    @templates_directory.setter
    def templates_directory(self, templates_directory):
        if not isinstance(templates_directory, str):
            raise TypeError("templates_directory must be a str!")
        self._templates_directory = templates_directory

    def calc_npt_md(self):
        pass

    def write_potential(self):
        self.input.potential.write_file(
            file_name="potential.inp", cwd=self.working_directory
        )
        self.input.potential.copy_pot_files(self.working_directory)

    def write_init_xyz(self):
        filepath = self.working_directory + "/init.xyz"
        self.structure.write(filename=filepath, format="xyz")
        cell = self.structure.cell.array.diagonal()
        angle = self.structure.cell.angles()
        with open(filepath, "r") as file:
            data = file.readlines()
            data[1] = (
                "# CELL(abcABC): "
                + str(cell[0])
                + " "
                + str(cell[1])
                + " "
                + str(cell[2])
                + " "
                + str(angle[0])
                + " "
                + str(angle[1])
                + " "
                + str(angle[2])
                + " positions{angstrom} cell{angstrom}\n"
            )
        with open(filepath, "w") as file:
            file.writelines(data)

    def write_data_lmp(self):
        filepath = self.working_directory + "/data.lmp"
        self.structure.write(filename=filepath, format="lammps-data")

    def write_input_lmp(self):
        filepath = self.working_directory + "/input.lmp"
        mass = self.structure.get_masses()[0]
        data = (
            "# LAMMPS input file\n\n"
            + "atom_style \t atomic\n"
            + "units \t metal\n"
            + "dimension \t 3\n"
            + "boundary \t p p p\n"
            + "\n"
            + "read_data \t data.lmp\n"
            + "mass \t 1 "
            + str(mass)
            + "\n\n"
            + "include potential.inp\n\n"
            + "fix \t 1 all ipi "
            + self.job_name
            + " "
            + str(self.custom_input.port)
            + " unix\n"
            + "run \t 5000000"
        )
        with open(filepath, "w") as file:
            file.writelines(data)

    def write_ipi_xml(self):
        pass

    def write_shell_scripts(self):
        copy(
            self._templates_directory + "/run_ipi.sh",
            self.working_directory + "/run_ipi.sh",
        )
        copy(
            self._templates_directory + "/run_rdf.sh",
            self.working_directory + "/run_rdf.sh",
        )

    def write_template_file(self):
        pass

    def write_input(self):
        if not isdir(self.working_directory):
            self.project_hdf5.create_working_directory()
        super(IPiCore, self).write_input()
        self.write_potential()
        self.write_init_xyz()
        self.write_data_lmp()
        self.write_input_lmp()
        self.write_shell_scripts()
        self.write_template_file()
        self.write_ipi_xml()

    def run_static(self):
        subprocess.check_call(
            [
                self.working_directory + "/./run_ipi.sh",
                self.working_directory,
                str(self.server.cores),
            ]
        )
        self.collect_output()
        self.to_hdf()
        self.compress()

    def collect_props(self):
        f = open(self.working_directory + "/ipi_out.out", "r")
        lines = f.readlines()
        time = []
        temperature = []
        energy_kin = []
        energy_pot = []
        volume = []
        pressure = []
        for x in lines:
            if not x.startswith("#"):
                time.append(x.split()[1])
                temperature.append(x.split()[2])
                energy_kin.append(x.split()[3])
                energy_pot.append(x.split()[4])
                volume.append(x.split()[5])
                pressure.append(x.split()[6])
        f.close()
        self.custom_output.time = np.array([float(i) for i in time])
        self.custom_output.temperature = np.array([float(i) for i in temperature])
        self.custom_output.energy_kin = np.array([float(i) for i in energy_kin])
        self.custom_output.energy_pot = np.array([float(i) for i in energy_pot])
        self.custom_output.volume = np.array([float(i) for i in volume])
        self.custom_output.pressure = np.array([float(i) for i in pressure])
        self.custom_output.energy_tot = (
            self.custom_output.energy_pot + self.custom_output.energy_kin
        )

    def collect_cells(self):
        f = open(self.working_directory + "/ipi_out.pos_0.xyz")
        lines = f.readlines()
        abc = []
        ABC = []
        for x in lines:
            if x.startswith("#"):
                split_line = x.split()
                abc.append([float(i) for i in split_line[2:5]])
                ABC.append([float(i) for i in split_line[5:8]])
        f.close()
        self.custom_output.cell_abc = np.array(abc)
        self.custom_output.cell_ABC = np.array(ABC)

    def collect_output(self):
        self.collect_props()
        self.collect_cells()

    def collect_rdf(self):
        f = open(self.working_directory + "/ipi_out.AlAl.rdf.dat", "r")
        lines = f.readlines()
        rdf_r = []
        rdf_g_r = []
        for x in lines:
            rdf_r.append(x.split()[0])
            rdf_g_r.append(x.split()[1])
        f.close()
        return np.array([float(i) for i in rdf_r]), np.array(
            [float(i) for i in rdf_g_r]
        )

    def get_rdf(self, r_min=2.0, r_max=5.0, bins=100, thermalize=50):
        self.decompress()
        rdf_list = [
            self.working_directory + "/./run_rdf.sh",
            self.working_directory,
            str(self.custom_input.temperature),
            self.structure.get_chemical_symbols()[0],
            self.structure.get_chemical_symbols()[0],
            str(bins),
            str(r_min),
            str(r_max),
            str(thermalize),
        ]
        subprocess.check_call(rdf_list)
        rdf_r, rdf_g_r = self.collect_rdf()
        self.compress()
        return rdf_r, rdf_g_r

    def to_hdf(self, hdf=None, group_name=None):
        super(IPiCore, self).to_hdf(hdf=hdf, group_name=group_name)
        self._structure_to_hdf()
        self.custom_input.templates_directory = self._templates_directory
        self.custom_input.to_hdf(self._hdf5)
        self.custom_output.to_hdf(self._hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super(IPiCore, self).from_hdf(hdf=hdf, group_name=group_name)
        self._structure_from_hdf()
        self.custom_input.from_hdf(self._hdf5)
        self._templates_directory = self.custom_input.templates_directory
        self.custom_output.from_hdf(self._hdf5)
