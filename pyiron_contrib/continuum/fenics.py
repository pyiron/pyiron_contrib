# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
A job class for performing finite element simulations using the [FEniCS](https://fenicsproject.org) code.
"""

import fenics as FEN
import mshr
from pyiron_base import GenericJob, InputList, PyironFactory
from os.path import join

__author__ = "Muhammad Hassani, Liam Huber"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Muhammad Hassani"
__email__ = "hassani@mpie.de"
__status__ = "development"
__date__ = "Dec 6, 2020"


class Fenics(GenericJob):
    """
    A job class for using the FEniCS library to solve a finite element method (FEM) problem.
    """

    def __init__(self, project, job_name):
        super(Fenics, self).__init__(project, job_name)
        self._python_only_job = True
        self.create = Creator(self)

        self.input = InputList(table_name='input')
        self.input.mesh_resolution = 2
        self.input.element_type = 'P'
        self.input.element_order = 1
        # TODO?: Make input sub-classes to catch invalid input?

        self.output = InputList(table_name='output')
        self.output.u = None

        self.domain = self.create.domain()  # the domain
        self.BC = None  # the boundary condition
        self.LHS = None  # the left hand side of the equation; FEniCS function
        self.RHS = None  # the right hand side of the equation; FEniCS function

        self._mesh = None  # the discretization mesh
        self._V = None  # finite element volume space
        self._u = None  # u is the unkown function
        self._v = None  # the test function
        self._vtk_filename = join(self.project_hdf5.path, 'output.pvd')

    def generate_mesh(self):
        self._mesh = mshr.generate_mesh(self.domain, self.input.mesh_resolution)
        self._V = FEN.FunctionSpace(self.mesh, self.input.element_type, self.input.element_order)
        # TODO: Allow changing what type of function space is used (VectorFunctionSpace, MultiMeshFunctionSpace...)
        # TODO: Allow having multiple sets of spaces and test/trial functions
        self._u = FEN.TrialFunction(self.V)
        self._v = FEN.TestFunction(self.V)

    def refresh(self):
        self.generate_mesh()

    @property
    def mesh(self):
        if self._mesh is None:
            self.refresh()
        return self._mesh

    @property
    def V(self):
        if self._V is None:
            self.refresh()
        return self._V

    @property
    def u(self):
        if self._u is None:
            self.refresh()
        return self._u

    @property
    def v(self):
        if self._v is None:
            self.refresh()
        return self._v
    # TODO: Do all this refreshing with a simple decorator instead of duplicate code

    def grad(self, arg):
        """
        Returns the gradient of the given argument.
        """
        return FEN.grad(arg)

    def Constant(self, value):
        """
        Wraps a value as a fenics constant.

        Args:
            value (float): The value to wrap.

        Returns:
            fenics.Constant: The wrapped value.
        """
        return FEN.Constant(value)

    def DirichletBC(self, expression, boundary):
        """
        This function defines Drichlet boundary condition based on the given expression on the boundary.

        Args:
            expression (string): The expression used to evaluate the value of the unknown on the boundary.
            boundary (fenics.DirichletBC): The spatial boundary, which the condition will be applied to.
        """
        return FEN.DirichletBC(self.V, expression, boundary)
    
    def dot(self, arg1, arg2):
        """
        Returns the dot product between the FEniCS objects.
        """
        return FEN.dot(arg1, arg2)

    @property
    def dx(self):
        """
        Returns the FEniCS dx object.
        """
        return FEN.dx

    def Expression(self, *args, **kwargs):
        return FEN.Expression(*args, **kwargs)

    def BC_default(self, x, on_boundary):
        """
        Returns the geometrical boundary.
        """
        return on_boundary

    def write_vtk(self):
        """
        Write the output to a .vtk file.
        """
        vtkfile = FEN.File(self._vtk_filename)
        vtkfile << self.u

    def validate_ready_to_run(self):
        if self.mesh is None:
            raise ValueError("No mesh is defined")
        if self.RHS is None:
            raise ValueError("The bilinear form (RHS) is not defined")
        if self.LHS is None:
            raise ValueError("The linear form (LHS) is not defined")
        if self.V is None:
            raise ValueError("The volume is not defined; no V defined")
        if self.BC is None:
            raise ValueError("The boundary condition(s) (BC) is not defined")

    def run_static(self):
        """
        Solve a PDE based on 'LHS=RHS' using u and v as trial and test function respectively. Here, u is the desired
        unknown and RHS is the known part.
        """
        self.status.running = True
        self._u = FEN.Function(self.V)
        FEN.solve(self.LHS == self.RHS, self.u, self.BC)
        self.status.collect = True
        self.run()

    def collect_output(self):
        self.output.u = self.u.compute_vertex_values(self.mesh)
        self.write_vtk()
        self.to_hdf()
        self.status.finished = True
    
    def plot_u(self):
        """
        Plots the unknown u.
        """
        FEN.plot(self.u)

    def plot_mesh(self):
        """
        Plots the mesh.
        """
        FEN.plot(self.mesh)

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(hdf=self.project_hdf5)
        self.output.to_hdf(hdf=self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(hdf=self.project_hdf5)
        self.output.from_hdf(hdf=self.project_hdf5)


class Creator:
    def __init__(self, job):
        self._job = job
        self._domain = DomainFactory()

    @property
    def domain(self):
        return self._domain


class DomainFactory(PyironFactory):

    def circle(self, center, radius):
        return mshr.Circle(FEN.Point(*center), radius)
    circle.__doc__ = mshr.Circle.__doc__

    def square(self, length, origin=None):
        if origin is None:
            x, y = 0, 0
        else:
            x, y = origin[0], origin[1]
        return mshr.Rectangle(FEN.Point(0 + x, 0 + y), FEN.Point(length + x, length + y))
    square.__doc__ = mshr.Rectangle.__doc__

    def __call__(self):
        return self.square(1.)
