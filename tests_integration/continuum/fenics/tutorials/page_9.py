# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from fenics import *
from ufl import nabla_div

def linear_elasticity():
    # Scaled variables
    L = 1;
    W = 0.2
    mu = 1
    rho = 1
    delta = W / L
    gamma = 0.4 * delta ** 2
    beta = 1.25
    lambda_ = beta
    g = gamma

    # Create mesh and define function space
    mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 3, 3)
    V = VectorFunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    tol = 1E-14

    def clamped_boundary(x, on_boundary):
        return on_boundary and x[0] < tol

    bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

    # Define strain and stress

    def epsilon(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
        # return sym(nabla_grad(u))

    def sigma(u):
        return lambda_ * nabla_div(u) * Identity(d) + 2 * mu * epsilon(u)

    # Define variational problem
    u = TrialFunction(V)
    d = u.geometric_dimension()  # space dimension
    v = TestFunction(V)
    f = Constant((0, 0, -rho * g))
    T = Constant((0, 0, 0))
    a = inner(sigma(u), epsilon(v)) * dx
    L = dot(f, v) * dx + dot(T, v) * ds

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    return u.compute_vertex_values(mesh)