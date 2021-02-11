# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from fenics import *
from mshr import *


def membrane():
    domain = Circle(Point(0, 0), 1)
    mesh = generate_mesh(domain, 64)
    beta = 8
    R0 = 0.6
    p = Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))',
                   degree=1, beta=beta, R0=R0)
    # p.beta = 12
    # p.R0 = 0.3

    def does_not_appear_on_webpage(mesh):
        """But rather in downloaded .py file from the page."""
        V = FunctionSpace(mesh, 'P', 2)

        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, Constant(0.), boundary)

        return V, bc

    V, bc = does_not_appear_on_webpage(mesh)

    w = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(w), grad(v)) * dx
    L = p * v * dx

    w = Function(V)
    solve(a == L, w, bc)

    return w.compute_vertex_values(mesh)
