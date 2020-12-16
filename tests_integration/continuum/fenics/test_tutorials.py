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
from .tutorials import page_5, page_6, page_7


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
        job.input.element_type = 'P'
        job.input.element_order = 1
        job.domain = job.create.domain.unit_square(8, 8)

        u_D = job.Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)  # String expressions must have valid C++ syntax
        job.BC = job.create.bc.dirichlet(u_D)

        job.LHS = job.dot(job.grad_u, job.grad_v) * job.dx
        job.RHS = job.Constant(-6.0) * job.v * job.dx

        job.run()

        self.assertTrue(np.all(np.isclose(job.output.u[-1], page_5.poisson())))

    def test_page_6(self):
        job = self.pr.create.job.Fenics('membrane', delete_existing_job=True)
        job.input.mesh_resolution = 64
        job.input.element_type = 'P'
        job.input.element_order = 2

        job.domain = job.create.domain.circle((0, 0), 1)
        job.BC = job.create.bc.dirichlet(job.Constant(0))

        p = job.Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))', degree=1, beta=12, R0=0.3)
        job.LHS = job.dot(job.grad_u, job.grad_v) * job.dx
        job.RHS = p * job.v * job.dx

        job.run()

        self.assertTrue(np.all(np.isclose(job.output.u[-1], page_6.membrane())))

    def test_page_7_heat(self):
        job = self.pr.create.job.Fenics('heat', delete_existing_job=True)
        total_time = 2.0  # final time
        num_steps = 10  # number of time steps
        dt = total_time / num_steps  # time step size

        job.input.element_types = 'P'
        job.input.element_order = 1
        job.input.n_steps = 10
        job.input.dt = 0.2

        job.domain = job.create.domain.unit_square(8, 8)

        u_D = job.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=3, beta=1.2, t=0)
        job.BC = job.create.bc.dirichlet(u_D)

        u_n = job.interpolate_function(u_D)

        f = job.Constant(u_D.beta - 2 - 2 * u_D.alpha)
        job.F = job.u * job.v * job.dx + dt * job.dot(job.grad_u, job.grad_v) * job.dx \
                - (u_n + job.input.dt * f) * job.v * job.dx

        job.time_dependent_expressions.append(u_D)
        job.assigned_u = u_n
        # Notebook tutorial code ends. Copy and paste this chunk into the appropriate tutorial cell.

        job.run()

        self.assertTrue(np.all(np.isclose(job.output.u[-1], page_7.heat_equation())))
