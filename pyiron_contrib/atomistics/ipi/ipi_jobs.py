# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
import warnings

from pyiron_contrib.atomistics.ipi.ipi_core import IPiCore

from xml.etree import ElementTree
import random
from shutil import copy

__author__ = "Raynol Dsouza"
__copyright__ = "Copyright 2022, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Raynol Dsouza"
__email__ = "dsouza@mpie.de"
__status__ = "development"
__date__ = "Jan 18, 2023"

class PiMD(IPiCore):

    def __init__(self, project, job_name):
        super(PiMD, self).__init__(project, job_name)

    def calc_npt_md(self, temperature=300., pressure=101325e-9, n_beads=4, timestep=1.,
                    temperature_damping_timescale=100., pressure_damping_timescale=1000.,
                    n_ionic_steps=100, n_print=1, seed=None, port=31415, constrain_ids=None, *args, **kwargs):
        if not seed:
            random.seed(self.job_name)
            seed = random.randint(1, 99999)
        self.custom_input.temperature = temperature
        self.custom_input.pressure = pressure
        self.custom_input.n_beads = n_beads
        self.custom_input.timestep = timestep
        self.custom_input.temperature_damping_timescale = temperature_damping_timescale
        self.custom_input.pressure_damping_timescale = pressure_damping_timescale
        self.custom_input.n_ionic_steps = n_ionic_steps
        self.custom_input.n_print = n_print
        self.custom_input.seed = seed
        self.custom_input.port = port
        self.custom_input.constrain_ids = constrain_ids

    def write_template_file(self):
        copy(self._templates_directory + '/pimd_template.xml', self.working_directory + '/pimd_template.xml')

    def ipi_write_helper(self, xml_filename):
        if not isinstance(xml_filename, str):
            raise "template must be an xml filename string!"
        tree = ElementTree.parse(self.working_directory + '/' + xml_filename)
        root = tree.getroot()
        filepath = self.working_directory + '/ipi_input.xml'
        for i in range(0, 3):
            root[0][i].attrib['stride'] = str(self.custom_input.n_print)
        root[0][3].attrib['stride'] = str(self.custom_input.n_ionic_steps)
        root[1].text = str(self.custom_input.n_ionic_steps)
        root[2][0].text = str(self.custom_input.seed)
        root[3][0].text = self.job_name
        root[4][0][0].text = 'init.xyz'
        root[4][0][1].text = str(self.custom_input.temperature)
        root[4][2][0][0][0].text = str(self.custom_input.pressure_damping_timescale)
        root[4][2][0][2].text = str(self.custom_input.timestep)
        constrain_ids = self.custom_input.constrain_ids.copy()
        if constrain_ids is not None:
            if not isinstance(constrain_ids, list):
                constrain_ids = constrain_ids.tolist()
            root[4][2][2].text = str(constrain_ids)
        root[4][3][0].text = str(self.custom_input.temperature)
        root[4][3][1].text = str(self.custom_input.pressure)
        tree.write(filepath)
        return tree, root, filepath

    def write_ipi_xml(self):
        tree, root, filepath = self.ipi_write_helper('pimd_template.xml')
        root[4][0].attrib['nbeads'] = str(self.custom_input.n_beads)
        root[4][2][0][1][0].text = str(self.custom_input.temperature_damping_timescale)
        tree.write(filepath)

class GleMD(PiMD):

    def __init__(self, project, job_name):
        super(GleMD, self).__init__(project, job_name)

    def calc_npt_md(self, A=None, C=None, n_beads=1, *args, **kwargs):
        super(GleMD, self).calc_npt_md(A=None, C=None, n_beads=1, *args, **kwargs)
        if self.custom_input.n_beads != 1:
            warnings.warn("For GLE, the number of n_beads == 1. Setting n_beads to 1.")
        self.custom_input.A = A
        self.custom_input.C = C

    def write_template_file(self):
        copy(self._templates_directory + '/gle_template.xml', self.working_directory + '/gle_template.xml')

    def write_ipi_xml(self):
        tree, root, filepath = self.ipi_write_helper('gle_template.xml')
        root[4][0].attrib['nbeads'] = str(1)
        root[4][2][0][1][0].text = str(self.custom_input.A)
        root[4][2][0][1][1].text = str(self.custom_input.C)
        tree.write(filepath)

class PigletMD(PiMD):

    def __init__(self, project, job_name):
        super(PigletMD, self).__init__(project, job_name)

    def calc_npt_md(self, A=None, C=None, *args, **kwargs):
        super(PigletMD, self).calc_npt_md(A=None, C=None, *args, **kwargs)
        self.custom_input.A = A
        self.custom_input.C = C

    def write_template_file(self):
        copy(self._templates_directory + '/piglet_template.xml', self.working_directory + '/piglet_template.xml')

    def write_ipi_xml(self):
        tree, root, filepath = self.ipi_write_helper('piglet_template.xml')
        root[4][0].attrib['nbeads'] = str(self.custom_input.n_beads)
        for i in range(2):
            root[4][2][0][1][i].attrib['shape'] = str((self.custom_input.n_beads,9,9))
        root[4][2][0][1][0].text = str(self.custom_input.A)
        root[4][2][0][1][1].text = str(self.custom_input.C)
        tree.write(filepath)
