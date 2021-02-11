# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Compare the mesh-node-evaluated solutions of our fenics wrapper to the raw tutorial code from the fenics documentation:
https://fenicsproject.org/pub/tutorial/html/._ftut1000.html

Fenics documentation scripts are all stored in the `tutorials` module and each method returns the numeric solution.
Plotting and other stuff has simply been commented out. The naming scheme is according to the website pages and is
accessed on 2020-12-16

The test code can be copied and pasted into the relevant tutorial notebook to maintain synchronization at minimal (but
not zero) cost in developer time.
"""

import unittest
from pyiron_base import Project
import pyiron_contrib
import numpy as np
from .tutorials import page_5, page_6, page_7, page_8, page_9


class TestFenicsTutorials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pr = Project('fenics_tutorial_integration')

    @classmethod
    def tearDownClass(cls):
        cls.pr.remove_jobs_silently(recursive=True)
        cls.pr.remove(enable=True)

    def test_page_5(self):
        job = self.pr.create.job.Fenics('poisson', delete_existing_job=True)
        job.domain = job.create.domain.unit_mesh.square(8, 8)

        u_D = job.Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)  # String expressions must have valid C++ syntax
        job.BC = job.create.bc.dirichlet(u_D)

        job.LHS = job.dot(job.grad_u, job.grad_v) * job.dx
        job.RHS = job.Constant(-6.0) * job.v * job.dx

        job.run()

        self.assertTrue(np.all(np.isclose(job.output.solution[-1], page_5.poisson())))

    def test_page_6(self):
        job = self.pr.create.job.Fenics('membrane', delete_existing_job=True)
        job.input.mesh_resolution = 64
        job.input.element_order = 2

        job.domain = job.create.domain.circle((0, 0), 1)
        job.BC = job.create.bc.dirichlet(job.Constant(0))

        p = job.Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))', degree=1, beta=8, R0=0.6)
        job.LHS = job.dot(job.grad_u, job.grad_v) * job.dx
        job.RHS = p * job.v * job.dx

        job.run()

        self.assertTrue(np.all(np.isclose(job.output.solution[-1], page_6.membrane())))

    def test_page_7_heat(self):
        job = self.pr.create.job.Fenics('heat', delete_existing_job=True)

        job.input.n_steps = 10
        job.input.dt = 2.0 / job.input.n_steps

        job.domain = job.create.domain.unit_mesh.square(8, 8)

        u_D = job.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=3, beta=1.2, t=0)
        job.BC = job.create.bc.dirichlet(u_D)

        u_n = job.interpolate_function(u_D)

        f = job.Constant(u_D.beta - 2 - 2 * u_D.alpha)
        job.F = job.u * job.v * job.dx + job.input.dt * job.dot(job.grad_u, job.grad_v) * job.dx \
                - (u_n + job.input.dt * f) * job.v * job.dx

        job.time_dependent_expressions.append(u_D)
        job.assigned_u = u_n

        job.run()

        self.assertTrue(np.all(np.isclose(job.output.solution[-1], page_7.heat_equation())))

    def test_page_7_gaussian(self):
        job = self.pr.create.job.Fenics('gauss', delete_existing_job=True)
        job.input.n_steps = 50
        job.input.dt = 2.0 / job.input.n_steps

        job.domain = job.create.domain.regular_mesh.rectangle((-2, -2), (2, 2), 30, 30)
        job.BC = job.create.bc.dirichlet(job.Constant(0))

        u_0 = job.Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', degree=2, a=5)
        u_n = job.interpolate_function(u_0)

        f = job.Constant(0)
        job.F = job.u * job.v * job.dx + job.input.dt * job.dot(job.grad_u, job.grad_v) * job.dx \
                - (u_n + job.input.dt * f) * job.v * job.dx

        job.assigned_u = u_n

        job.run()

        self.assertTrue(np.all(np.isclose(job.output.solution[-1], page_7.gaussian_evolution())))

    def test_page_8(self):
        job = self.pr.create.job.Fenics('poisson_nonlinear', delete_existing_job=True)

        def q(u):
            """Return nonlinear coefficient"""
            return 1 + u ** 2

        x, y = job.sympy.symbols('x[0], x[1]')
        u = 1 + x + 2 * y
        f = - job.sympy.diff(q(u) * job.sympy.diff(u, x), x) - job.sympy.diff(q(u) * job.sympy.diff(u, y), y)
        f = job.sympy.simplify(f)
        u_code = job.sympy.printing.ccode(u)
        f_code = job.sympy.printing.ccode(f)

        job.domain = job.create.domain.unit_mesh.square(8, 8)

        u_D = job.Expression(u_code, degree=1)
        job.BC = job.create.bc.dirichlet(u_D)

        f = job.Expression(f_code, degree=1)
        job.LHS = q(job.solution) * job.dot(job.grad_solution, job.grad_v) * job.dx - f * job.v * job.dx
        job.RHS = 0

        job.run()

        self.assertTrue(np.all(np.isclose(job.output.solution[-1], page_8.poisson_nonlinear())))

    def test_page_9(self):
        # Scaled variables
        L = 1
        W = 0.2
        mu = 1
        rho = 1
        delta = W / L
        gamma = 0.4 * delta ** 2
        lambda_ = 1.25

        job = self.pr.create.job.FenicsLinearElastic('linear_elasticity', delete_existing_job=True)
        job.input.bulk_modulus = lambda_ + (2 * mu / 3)
        job.input.shear_modulus = mu
        job.domain = job.create.domain.regular_mesh.box((0, 0, 0), (L, W, W), 10, 3, 3)

        def clamped_boundary(x, on_boundary):
            return on_boundary and x[0] < 1e-14

        job.BC = job.create.bc.dirichlet(job.Constant((0, 0, 0)), bc_fnc=clamped_boundary)

        job.f = job.Constant((0, 0, -rho * gamma))
        job.T = job.Constant((0, 0, 0))

        job.run()

        self.assertTrue(np.all(np.isclose(
            job.solution_to_original_shape(job.output.solution[-1]),
            page_9.linear_elasticity()
        )))
