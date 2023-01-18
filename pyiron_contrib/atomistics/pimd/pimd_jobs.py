# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_contrib.atomistics.pimd.pimd_core import PIMDCore

import xml.etree.ElementTree as ET

__author__ = "Raynol Dsouza"
__copyright__ = "Copyright 2022, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Raynol Dsouza"
__email__ = "dsouza@mpie.de"
__status__ = "development"
__date__ = "Jan 18, 2023"


class Piglet(PIMDCore):
    
    def __init__(self, project, job_name):
        super(Piglet, self).__init__(project, job_name)
        
    def calc_npt_md(self, temperature=300., pressure=101325e-9, n_beads=4, timestep=1., damping_timescale=100., 
                    n_ionic_steps=100, n_print=1, seed=32345, port=31415, A=None, C=None):
        self.custom_input.temperature = temperature
        self.custom_input.pressure = pressure
        self.custom_input.n_beads = n_beads
        self.custom_input.timestep = timestep
        self.custom_input.damping_timescale = damping_timescale
        self.custom_input.n_ionic_steps = n_ionic_steps
        self.custom_input.n_print = n_print
        self.custom_input.seed = seed
        self.custom_input.port = port
        self.custom_input.A = A
        self.custom_input.C = C

    def write_ipi_xml(self):
        tree = ET.parse(self._templates_directory+ '/piglet_template.xml')
        root = tree.getroot()
        filepath = self.working_directory + '/ipi_input.xml'
        for i in range(4):
            root[0][i].attrib['stride'] = str(self.custom_input.n_print)
        root[1].text = str(self.custom_input.n_ionic_steps)
        root[2][0].text = str(self.custom_input.seed)
        root[3][0].text = self.job_name
        root[4][0].attrib['nbeads'] = str(self.custom_input.n_beads)
        root[4][0][0].text = 'init.xyz'
        root[4][0][1].text = str(self.custom_input.temperature)
        root[4][2][0][0][0].text = str(self.custom_input.damping_timescale)
        for i in range(2):
            root[4][2][0][1][i].attrib['shape'] = str((self.custom_input.n_beads,9,9))
        root[4][2][0][1][0].text = str(self.custom_input.A)
        root[4][2][0][1][1].text = str(self.custom_input.C)
        root[4][2][0][2].text = str(self.custom_input.timestep)
        root[4][3][0].text = str(self.custom_input.temperature)
        root[4][3][1].text = str(self.custom_input.pressure)
        tree.write(filepath)
