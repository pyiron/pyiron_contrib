# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import warnings

"""
A job class for FEM linear elasticity with [fenics](https://fenicsproject.org/pub/tutorial/html/._ftut1008.html).
"""

from pyiron_contrib.continuum.fenics.job.generic import Fenics


class FenicsLinearElastic(Fenics):
    """
    Solves a linear elastic problem in three dimensions using bulk and shear moduli to describe the elasticity.
    Determines the displacement given the body load, traction, and boundary conditions.

    The variational equation solved is the integral of:
    `inner(sigma(u), epsilon(v)) * dx == dot(f, v) * dx + dot(T, v) * ds`

    Parameters:
        f (Constant/Expression): The body force term. (Default is Constant((0, 0, 0)).)
        T (Constant/Expression): The traction conditions. (Default is Constant((0, 0, 0)).)

    Input:
        bulk_modulus (float): Material elastic parameter. (Default is 76, the experimental value for Al in GPa.)
        shear_modulus (float): Material elastic parameter. (Default is 26, the experimental value for Al in GPa.)

    Output
        von_Mises (list): The von Mises stress from the mesh-evaluated solution at each step.
    """

    def __init__(self, project, job_name):
        """Create a new Fenics type job for linear elastic problems"""
        super().__init__(project=project, job_name=job_name)
        self.input.bulk_modulus = 76
        self.input.shear_modulus = 26

        self.output.von_Mises = []

        self.V_class = self.fenics.VectorFunctionSpace

        self.f = self.Constant((0, 0, 0))
        self.T = self.Constant((0, 0, 0))

    def epsilon(self, u):
        return self.fenics.sym(self.nabla_grad(u))

    def sigma(self, u):
        lambda_ = self.input.bulk_modulus - (2 * self.input.shear_modulus / 3)
        return lambda_ * self.nabla_div(u) * self.Identity(u.geometric_dimension()) \
               + 2 * self.input.shear_modulus * self.epsilon(u)

    @property
    def LHS(self):
        return self._lhs

    @LHS.setter
    def LHS(self, _):
        raise NotImplementedError

    @property
    def RHS(self):
        return self._rhs

    @RHS.setter
    def RHS(self, _):
        raise NotImplementedError

    def validate_ready_to_run(self):
        self._lhs = self.inner(self.sigma(self.u), self.epsilon(self.v)) * self.dx
        self._rhs = self.dot(self.f, self.v) * self.dx + self.dot(self.T, self.v) * self.ds
        super().validate_ready_to_run()

    def von_Mises(self, u):
        s = self.sigma(u) - (1. / 3) * self.tr(self.sigma(u)) * self.Identity(u.geometric_dimension())
        return self.fenics.project(self.sqrt(3. / 2 * self.inner(s, s)), self.fenics.FunctionSpace(self.mesh, 'P', 1))

    def _append_to_output(self):
        super()._append_to_output()
        self.output.von_Mises.append(self.von_Mises(self.solution).compute_vertex_values(self.mesh))
