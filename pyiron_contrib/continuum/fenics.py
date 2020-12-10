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
        self.input = InputList(table_name='input')
        self.output = InputList(table_name='output')
        self.LHS = None  # the left hand side of the equation; FEniCS function
        self.RHS = None  # the right hand side of the equation; FEniCS function
        self._vtk_filename = join(self.project_hdf5.path, 'output.pvd')
        self.mesh = None
        self.mesh = None
        self.BC = None
        self.V = None  # finite element volume space
        self.u = None  # u is the unkown function
        self.v = None  # the test function
        self.domain = None  # the domain
        self.create = Creator(self)

    def point(self, x, y):
        """
        Returns a spatial point as fenics object, based on the given coordinate.
        """
        return FEN.Point(x, y)

    def grad(self, arg):
        """
        Returns the gradient of the given argument.
        """
        return FEN.grad(arg)

    def Circle(self, center, rad):
        """
        Create a mesh on a circular domain with a radius equal to rad.
        """
        return mshr.Circle(center, rad)
    
    def dxProd(self, A):
        """
        Returns the product of A and the FEniCS library's dx object.
        """
        return A*FEN.dx

    def generate_mesh(self, typ, order, resolution):
        """
        Generate a mesh based on the resolution and the job's `domain` attribute and assigns it to the `mesh` attribute.
        Additionally, it creates and assigns as an attribute the finite element volume `V` with the given type and
        order, as well as the unknown trial function `u` and test function `v`.

        Args:
            type (str): Type of the elements making up the mesh.
            order (int): Order of the elements.
            resolution (int): Controls how fine/coarse the mesh is.
        """
        self.mesh = mshr.generate_mesh(self.domain, resolution)
        self.V = FEN.FunctionSpace(self.mesh, typ, order)
        self.u = FEN.TrialFunction(self.V)
        self.v = FEN.TestFunction(self.V)

    def FunctionSpace(self, typ, order):
        """
        Returns the volume function using the current mesh.

        Args:
            typ (string): The type of the element; e.g. 'p' refers to langrangian element. For further information see
                https://fenicsproject.org/pub/graphics/fenics-femtable-cards.pdf
            order (int): The order of the element. For further information see
                https://fenicsproject.org/pub/graphics/fenics-femtable-cards.pdf
        """
        return FEN.FunctionSpace(self.mesh, typ, order)

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

    def TrialFunction(self):
        """
        Returns a FEniCS trial function
        """
        return FEN.TrialFunction(self.V)
    
    def TestFunction(self):
        """
        Returns a FEniCS test function
        """
        return FEN.TestFunction(self.V)
    
    def dot(self, arg1, arg2):
        """
        Returns the dot product between the FEniCS objects.
        """
        return FEN.dot(arg1, arg2)
    
    def dx(self):
        """
        Returns the FEniCS dx object.
        """
        return FEN.dx

    def Expression(self, *args, **kwargs):
        return FEN.Expression(*args, **kwargs)

    def mesh_gen_default(self, intervals, typ='P', order=1):
        """
        Sets the mesh to a unit square (i.e. side length=1), updating the volume (`V`), and trial (`u`) and test (`v`)
        functions accordingly.

        Args:
            intervals (int): The number of squares on the mesh in each direction.
            typ (string): The type of the element; e.g. 'p' refers to langrangian element. (Default is 'P', i.e.
                Lagrangian.)For further information see
                https://fenicsproject.org/pub/graphics/fenics-femtable-cards.pdf.
            order (int): The order of the element. (Default is 1.) For further information see
                https://fenicsproject.org/pub/graphics/fenics-femtable-cards.pdf.
        """
        self.mesh = FEN.UnitSquareMesh(intervals, intervals)
        self.input['mesh'] = self.mesh
        self.V = FEN.FunctionSpace(self.mesh, typ, order)
        self.input['V'] = self.V
        self.u = FEN.TrialFunction(self.V)
        self.v = FEN.TestFunction(self.V)

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
        self.u = FEN.Function(self.V)
        FEN.solve(self.LHS == self.RHS, self.u, self.BC)
        self.collect_output()

    def collect_output(self):
        self.status.collect = True
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

    def square(self, length):
        return mshr.Rectangle(length, length)
    square.__doc__ = mshr.Rectangle.__doc__

    def __call__(self):
        return self.square(1.)
