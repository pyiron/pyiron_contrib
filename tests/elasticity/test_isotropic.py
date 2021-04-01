# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
import numpy as np
from pyiron_contrib.atomistic.elasticity.point_defect.isotropic import displacement_field
from pyiron_contrib.atomistic.elasticity.point_defect.isotropic import strain_field


class TestIsotropic(unittest.TestCase):

    def test_displacement_strain(self):
        r = 2*np.random.random(3)-1
        dipole_tensor = np.random.random(3)*np.eye(3)
        poissons_ratio = np.random.random()
        shear_modulus = np.random.random()
        epsilon = strain_field(
            r,
            dipole_tensor=dipole_tensor,
            poissons_ratio=poissons_ratio,
            shear_modulus=shear_modulus
        )
        dr = 1.0e-4
        rp = r[None,:]+np.eye(3)*dr
        rd = r[None,:]-np.eye(3)*dr
        up = displacement_field(
            rp,
            dipole_tensor=dipole_tensor,
            poissons_ratio=poissons_ratio,
            shear_modulus=shear_modulus
        )
        ud = displacement_field(
            rd,
            dipole_tensor=dipole_tensor,
            poissons_ratio=poissons_ratio,
            shear_modulus=shear_modulus
        )
        dudx = (up-ud)/(2*dr)
        dudx = 0.5*(dudx+dudx.T)
        self.assertLess(np.linalg.norm(epsilon-dudx, ord=2)/np.linalg.norm(epsilon, ord=2), 1.0e-4)


if __name__ == "__main__":
    unittest.main()

