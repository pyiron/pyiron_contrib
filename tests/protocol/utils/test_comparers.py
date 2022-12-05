# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from pyiron_atomistics.atomistics.structure.atoms import Atoms
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
        self.assertEqual(Comparer(4), 4)
        self.assertNotEqual(Comparer(4), 5)

        self.assertEqual(Comparer(4.4), 4.4)
        self.assertNotEqual(Comparer(4.4), 5.5)

        self.assertEqual(Comparer('a'), 'a')
        self.assertNotEqual(Comparer('a'), 'b')

        self.assertEqual(Comparer(4), 4.)
        self.assertNotEqual(Comparer(4), 'a')

        # Check that comparing comparers works too
        self.assertEqual(Comparer(4), Comparer(4))
        self.assertEqual(Comparer(4), Comparer(4.))
        self.assertNotEqual(Comparer(4), Comparer(4.4))
        self.assertNotEqual(Comparer(4), Comparer('a'))

    def test_array(self):
        a, b, c = self.create_arrays()

        self.assertEqual(Comparer(a), a)

        self.assertNotEqual(Comparer(a), b)
        self.assertNotEqual(Comparer(a), c)
        self.assertNotEqual(Comparer(a), 'a')

    def test_atoms(self):
        a, b, c, d = self.create_atoms()

        self.assertEqual(Comparer(a), a)

        self.assertNotEqual(Comparer(a), b)
        self.assertNotEqual(Comparer(a), c)
        self.assertNotEqual(Comparer(a), d)
        self.assertNotEqual(Comparer(a), 'a')

    def test_lists(self):
        array_a, array_b, _ = self.create_arrays()
        atoms_a, atoms_b, atoms_c, atoms_d = self.create_atoms()

        a = [1, 'a', array_a, atoms_a]
        b = [[atoms_a, atoms_b], [atoms_c, atoms_d]]

        self.assertEqual(Comparer(a), a)
        self.assertEqual(Comparer(b), b)

        self.assertNotEqual(Comparer(a), [2, 'a', array_a, atoms_a])
        self.assertNotEqual(Comparer(a), [1, 'b', array_a, atoms_a])
        self.assertNotEqual(Comparer(a), [1, 'a', array_b, atoms_b])
        self.assertNotEqual(Comparer(a), [1, 'a', array_a, atoms_a, 1])

        self.assertNotEqual(Comparer(b), [[atoms_a, atoms_b], [atoms_a, atoms_b]])
        self.assertNotEqual(Comparer(b), [[atoms_a, atoms_b], [atoms_c, atoms_d], [atoms_a, atoms_b]])
        self.assertNotEqual(Comparer(b), [b])
        self.assertNotEqual(Comparer(b), [[1, 2], ['a', 'b']])
