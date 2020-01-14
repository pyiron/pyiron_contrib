# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from pyiron.atomistics.structure.atoms import Atoms
from pyiron_contrib.protocol.utils.comparers import Comparer


class TestComparer(unittest.TestCase):
    """
    Test value differs.
    Test length/shape differs.
    Test type differs.
    """

    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    @staticmethod
    def create_arrays():
        a = np.random.rand(5, 5)
        b = np.random.rand(5, 5)
        c = np.random.rand(6, 6)
        return a, b, c

    @staticmethod
    def create_atoms():
        a = Atoms(
            symbols=["Fe", "Cu", "Ni", "Al"],
            positions=np.random.random((4, 3)),
            cell=np.eye(3),
        )
        b = a.copy()
        b.positions = np.random.random(b.positions.shape)
        c = a.copy()
        c[0] = "Cu"
        d = a + b
        return a, b, c, d

    def test_primitives(self):
        self.assertTrue(Comparer(4) == 4)
        self.assertFalse(Comparer(4) == 5)

        self.assertTrue(Comparer(4.4) == 4.4)
        self.assertFalse(Comparer(4.4) == 5.5)

        self.assertTrue(Comparer('a') == 'a')
        self.assertFalse(Comparer('a') == 'b')

        # self.assertTrue(Comparer(4) == 4.)  # Raises assertion error
        self.assertFalse(Comparer(4) == 'a')

        self.assertTrue(Comparer(4) == Comparer(4))
        # self.assertTrue(Comparer(4) == Comparer(4.))
        self.assertFalse(Comparer(4) == Comparer(4.4))
        self.assertFalse(Comparer(4) == Comparer('a'))

    def test_array(self):
        a, b, c = self.create_arrays()

        self.assertTrue(Comparer(a) == a)

        self.assertFalse(Comparer(a) == b)
        self.assertFalse(Comparer(a) == c)
        self.assertFalse(Comparer(a) == 'a')

    def test_atoms(self):
        a, b, c, d = self.create_atoms()

        self.assertTrue(Comparer(a) == a)

        self.assertFalse(Comparer(a) == b)
        self.assertFalse(Comparer(a) == c)
        self.assertFalse(Comparer(a) == d)
        self.assertFalse(Comparer(a) == 'a')

    def test_lists(self):
        array_a, array_b, _ = self.create_arrays()
        atoms_a, atoms_b, atoms_c, atoms_d = self.create_atoms()

        a = [1, 'a', array_a, atoms_a]
        b = [[atoms_a, atoms_b], [atoms_c, atoms_d]]

        self.assertTrue(Comparer(a) == a)
        self.assertTrue(Comparer(b) == b)

        self.assertFalse(Comparer(a) == [2, 'a', array_a, atoms_a])
        self.assertFalse(Comparer(a) == [1, 'b', array_a, atoms_a])
        self.assertFalse(Comparer(a) == [1, 'a', array_b, atoms_b])
        self.assertFalse(Comparer(a) == [1, 'a', array_a, atoms_a, 1])

        self.assertFalse(Comparer(b) == [[atoms_a, atoms_b], [atoms_a, atoms_b]])
        self.assertFalse(Comparer(b) == [[atoms_a, atoms_b], [atoms_c, atoms_d], [atoms_a, atoms_b]])
        self.assertFalse(Comparer(b) == [b])
        self.assertFalse(Comparer(b) == [[1, 2], ['a', 'b']])
