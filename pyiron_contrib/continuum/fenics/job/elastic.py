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
    Solves a linear elastic problem in three dimensions using Lame's parameters to describe the elasticity.

    The exact variational equation solved is the integral of:
        `inner(sigma(u), epsilon(v)) * dx == dot(f, v) * dx + dot(T, v) * ds`

    Parameters:
        f (Constant/Expression): The body force term. (Default is Constant((0, 0, 0)).)
        T (Constant/Expression): The traction conditions. (Default is Constant((0, 0, 0)).)

    Input:
        lambda_ (float): The Lame lambda parameter. (Default is 1.25.)
        mu (float): The Lame mu parameters. (Default is 10.)
    """

    def __init__(self, project, job_name):
        """Create a new Fenics type job for linear elastic problems"""
        super().__init__(project=project, job_name=job_name)
        self.input.lambda_ = 1.25
        self.input.mu = 10.

        self.V_class = self.fenics.VectorFunctionSpace

        self.f = self.Constant((0, 0, 0))
        self.T = self.Constant((0, 0, 0))

    def epsilon(self, u):
        return self.fenics.sym(self.nabla_grad(u))

    def sigma(self, u):
        return self.input.lambda_ * self.nabla_div(u) * self.Identity(u.geometric_dimension()) \
               + 2 * self.input.mu * self.epsilon(u)

    def validate_ready_to_run(self):
        self.LHS = self.inner(self.sigma(self.u), self.epsilon(self.v)) * self.dx
        self.RHS = self.dot(self.f, self.v) * self.dx + self.dot(self.T, self.v) * self.ds
        super().validate_ready_to_run()

    def von_Mises(self, u):
        s = self.sigma(u) - (1. / 3) * self.tr(self.sigma(u)) * self.Identity(u.geometric_dimension())
        return self.fenics.project(self.sqrt(3. / 2 * self.inner(s, s)), self.fenics.FunctionSpace(self.mesh, 'P', 1))
