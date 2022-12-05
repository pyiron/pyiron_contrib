# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import unittest
from pyiron_atomistics._tests import TestWithCleanProject
import pyiron_contrib


class TestTrainingContainer(TestWithCleanProject):

    def setUp(self):
        self.container = self.project.create.job.TrainingContainer("test")
        self.basis_1 = self.project.create.structure.ase.bulk("Al")
        force = [[0.0, 0.0, 0.0]]
        energy = 0.0
        self.container.include_structure(self.basis_1, energy=energy, forces=force, name="unitcell")
        self.basis_2 = self.project.create.structure.ase.bulk("Al").repeat([2, 1, 1])
        force = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]
        energy = 0.01
        self.container.include_structure(self.basis_2, energy=energy, forces=force, name="repeated")
        self.container.run()

    def tearDown(self):
        del self.container

    def test_include_structure(self):
        structure_list, energy_list, forces_list, num_atoms = self.container.to_list()
        self.assertEqual(len(structure_list), 2)
        self.assertEqual(structure_list[0], self.basis_1)
        self.assertEqual(structure_list[1], self.basis_2)
        self.assertEqual(num_atoms, [1, 2])
        self.assertEqual(energy_list, [0.0, 0.01])
        structure_list, energy_list, _, _ = self.container.to_list(
            filter_function=lambda df: df[df.number_of_atoms > 1]
        )
        self.assertEqual(len(structure_list), 1)
        self.assertEqual(energy_list[0], 0.01)

    def test_elements(self):
        self.assertEqual(self.container.get_elements(), ["Al"])

    def test_get_structure(self):
        self.assertEqual(len(self.container.get_structure(frame=0)), 1,
                         "get_structure() returned wrong structure.")
        self.assertEqual(len(self.container.get_structure(frame=1)), 2,
                         "get_structure() returned wrong structure.")

    def test_hdf(self):
        """Container read from HDF should match container written to HDF."""

        container_from_hdf = self.project.load("test")

        self.assertEqual(len(self.container.to_pandas()), len(container_from_hdf.to_pandas()),
                         "Container has different number of structures after reading/writing.")

        for i in range(len(self.container.to_pandas())):
            self.assertEqual(self.container.get_structure(i), container_from_hdf.get_structure(i),
                             f"{i}th structure not the same after reading/writing.")
            self.assertTrue((self.container._container.get_array("energy", i) \
                                == container_from_hdf._container.get_array("energy", i)).all(),
                            "Energy not the same after reading/writing.")
            self.assertTrue((self.container._container.get_array("forces", i) \
                                == container_from_hdf._container.get_array("forces", i)).all(),
                            "Energy not the same after reading/writing.")

        self.assertTrue(self.container.to_pandas().equals(container_from_hdf.to_pandas()),
                        "Conversion to pandas not the same after reading/writing.")
