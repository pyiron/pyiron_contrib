# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import unittest
from pyiron import Project
import pyiron_contrib
print(pyiron_contrib.__file__)


class TestTrainingContainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.execution_path, "test_training"))

    @classmethod
    def tearDownClass(cls):
        cls.execution_path = os.path.dirname(os.path.abspath(__file__))
        project = Project(os.path.join(cls.execution_path, "test_training"))
        project.remove_jobs_silently(recursive=True)
        project.remove(enable=True)

    def setUp(self):
        self.container = self.project.create.job.TrainingContainer("test")
        basis_1 = self.project.create_ase_bulk("Al")
        force = [[0.0, 0.0, 0.0]]
        energy = 0.0
        self.container.include_structure(basis_1, energy=energy, forces=force, name="unitcell")
        basis_2 = self.project.create_ase_bulk("Al").repeat([2, 1, 1])
        force = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]
        energy = 0.01
        self.container.include_structure(basis_2, energy=energy, forces=force, name="repeated")
        self.container.run()

    def tearDown(self):
        del self.container

    def test_include_structure(self):
        structure_list, energy_list, forces_list, num_atoms = self.container.to_list()
        self.assertEqual(len(structure_list), 2)
        self.assertEqual(structure_list[0], basis_1)
        self.assertEqual(structure_list[1], basis_2)
        self.assertEqual(num_atoms, [1, 2])
        self.assertEqual(energy_list, [0.0, 0.01])
        structure_list, energy_list, _, _ = self.container.to_list(filter_function=
                                                                         lambda df: df[df.number_of_atoms > 1])
        self.assertEqual(len(structure_list), 1)
        self.assertEqual(energy_list[0], 0.01)

    def test_elements(self):
        self.assertEqual(self.container.get_elements(), ["Al"])

    def test_get_structure(self):
        self.assertEqual(len(self.container.get_structure(iteration_step=0)), 1,
                         "get_structure() returned wrong structure.")
        self.assertEqual(len(self.container.get_structure(iteration_step=1)), 2,
                         "get_structure() returned wrong structure.")
