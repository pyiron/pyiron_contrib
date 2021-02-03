# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
A job class for performing finite element simulations using the [FEniCS](https://fenicsproject.org) code.
"""

import fenics as FEN
import mshr
import sympy
from pyiron_base import GenericJob, InputList
from os.path import join
import warnings
import numpy as np
from dolfin.cpp.mesh import Mesh
from pyiron_contrib.continuum.fenics.factory import DomainFactory, BoundaryConditionFactory
from pyiron_contrib.continuum.fenics.plot import Plot
from matplotlib.docstring import copy as copy_docstring
from ufl import nabla_div as ufl_nabla_div

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
    The job provides an interface to the [FEniCS project](https://fenicsproject.org) PDE solver using the finite element
    method (FEM).

    The objective is to streamline and simplify regular usage and connect FEniCS calculations to the full job management
    and execution distribution capabilities of pyiron, without losing the power or flexibility of the underlying fenics
    library.

    Flexibility and power are currently maintained by directly exposing the underlying `fenics` and `mshr` libraries as
    attributes of the job for power users.

    Ease of use is underway, e.g. elements, trial and test functions, and the mesh are automatically populated based on
    the provided domain. Quality of life will be continuously improved as pyiron and fenics get to know each other.

    TODO: Integration with pyiron's job and data management is incomplete, as some input data types (domains and
          boundary conditions) are not yet compatible with HDF5 storage. This is a simple problem to describe, but might
          be a pain to solve with sufficient flexibility. We also need to consider storing more sophisticated output.

    TODO: Full power and flexibility still needs to be realized by allowing (a) variable function space types, (b)
          variable number of elements and trial/test functions, and (c) multiple solve types.
          (a) Is a simple input option, we just need to be smart about how to limit the choices to existing fenics
              classes.
          (b) Can probably be nicely realized by subclassing off the main job type to allow for two sets of functions --
              `V:(u,v), Q:(p,q)` -- and a variable number of functions -- `V[0]:(u[0], v[0]),...,V[n]:(u[n], v[n])` --
              which are automatically populated during mesh generation and which are accessible for building the
              equation.
          (c) Solution types just means linear system `solve(A, x, b, ...)`, linear variational problems
              `solve(a == L, u, ...)`, and nonlinear variational problems `solve(F == 0, u, ...)`. This is probably also
              going to be pretty easy to control through an input parameter with a few fixed options an a bit of
              modification to how the LHS and RHS of equations are provided, and what actually is called during `run`.
              Currently the linear variational problem is hardcoded.

    Attributes:
        input (InputList): The input parameters controlling the run.
        output (InputList): The output from the run, i.e. data that comes from `solve`ing the PDE.
        domain (?): The spatial domain on which to build the mesh or, in the case of special meshes, the mesh itself.
            To be provided prior to running the job.
        BC (?): The boundary conditions for the mesh. To be provided prior to running the job.
        LHS/RHS (?): The left-hand and right-hand sides of the equation to solve.
        time_dependent_expressions (list[Expression]): All expressions used in the domain, BC, LHS and RHS which have a
            `t` attribute that needs updating at each step. (Default is None, which initializes an empty list.)
        assigned_u (?): The term which will be assigned the solution at each timestep. (Default is None, don't assign
            anything.)
        mesh (?): The mesh. Generated automatically.
        u:
        v:
        solution:
        F:

    Input:
        mesh_resolution (int): How dense the mesh should be (larger values = denser mesh). (Default is 2, but not used
            if the domain is a special mesh, e.g. unit or regular.)
        element_type (str): What type of element should be used. (Default is 'P'.) TODO: Restrict choices.
        element_order (int): What order the elements have. (Default is 1.)  TODO: Better description.
        n_steps (int): How many steps to run for, where the `t` attribute of all time dependent expressions gets updated
            at each step. (Default is 1.)
        n_print (int): The period of steps for saving to output. (Default is 1.) Note: The final step will always be
            saved regardless of the mod value.
        dt (float): How much to increase the `t` attribute of  all time dependent expressions each step. (Default is 1.)
        solver_parameters (dict): kwargs for FEniCS solver.
            Cf. [FEniCS docs](https://fenicsproject.org/pub/tutorial/html/._ftut1017.html) for more details. (Default is
            an empty dictionary.)

    Output:
        u (list): The solved function values evaluated at the mesh points at each time step.

    Example:
        >>> job = pr.create.job.Fenics('fenics_job')
        >>> job.input.mesh_resolution = 64
        >>> job.input.element_type = 'P'
        >>> job.input.element_order = 2
        >>> job.domain = job.create.domain.circle((0, 0), 1)
        >>> job.BC = job.create.bc.dirichlet(job.Constant(0))
        >>> p = job.Expression('4*exp(-pow(beta, 2)*(pow(x[0], 2) + pow(x[1] - R0, 2)))', degree=1, beta=8, R0=0.6)
        >>> job.LHS = job.dot(job.grad_u, job.grad_v) * job.dx
        >>> job.RHS = p * job.v * job.dx
        >>> job.run()
        >>> job.plot_u()
    """

    def __init__(self, project, job_name):
        """Create a new Fenics type job"""
        super(Fenics, self).__init__(project, job_name)
        warnings.warn("Currently, the c++ dolfin functions used by fenics are not stored in the HDF5 file."
                      " This includes the domains, boundary condition, spatial functions."
                      " Therefore, it is not possible to reload the job properly, from HDF5 file."
                      " It would be safe to remove the Fenics jobs, after defining the project.")
        self._python_only_job = True
        self.create = Creator(self)
        self._plot = Plot(self)

        self.input = InputList(table_name='input')
        self.input.mesh_resolution = 2
        self.input.element_type = 'P'
        self.input.element_order = 1
        self.input.n_steps = 1
        self.input.n_print = 1
        self.input.dt = 1
        self.input.solver_parameters = {}
        # TODO?: Make input sub-classes to catch invalid input?

        self.output = InputList(table_name='output')
        self.output.solution = []

        # TODO: Figure out how to get these attributes into input/otherwise serializable
        self.domain = None  # the domain
        self.BC = None  # the boundary condition
        self._lhs = None  # the left hand side of the equation; FEniCS function
        self._rhs = None  # the right hand side of the equation; FEniCS function
        self.time_dependent_expressions = []  # Any expressions used with a `t` attribute to evolve
        # TODO: Make a class to force these to be Expressions and to update them?
        self.assigned_u = None
        self.V_class = FEN.FunctionSpace

        self._mesh = None  # the discretization mesh
        self._V = None  # finite element volume space
        self._u = None  # u is the unkown function
        self._v = None  # the test function
        self._solution = None
        self._vtk_filename = join(self.project_hdf5.path, 'output.pvd')

    # Wrap equations in setters so they can be easily protected in subclasses
    @property
    def LHS(self):
        return self._lhs

    @LHS.setter
    def LHS(self, new_lhs):
        self._lhs = new_lhs

    @property
    def RHS(self):
        return self._rhs

    @RHS.setter
    def RHS(self, new_rhs):
        self._rhs = new_rhs

    @property
    def plot(self):
        return self._plot

    def generate_mesh(self):
        if isinstance(self.domain, Mesh):
            self._mesh = self.domain  # Intent: Allow the domain to return a unit mesh
        else:
            self._mesh = mshr.generate_mesh(self.domain, self.input.mesh_resolution)

        self._V = self.V_class(self.mesh, self.input.element_type, self.input.element_order)
        # TODO: Allow changing what type of function space is used (VectorFunctionSpace, MultiMeshFunctionSpace...)
        # TODO: Allow having multiple sets of spaces and test/trial functions
        self._u = FEN.TrialFunction(self.V)
        self._v = FEN.TestFunction(self.V)
        self._solution = FEN.Function(self.V)

        if any([v is not None for v in [self.BC, self.LHS, self.RHS]]):
            warnings.warn("The mesh is being generated, but at least one of the boundary conditions or equation sides"
                          "is already defined -- please re-define these values since the mesh is updated")

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

    @property
    def solution(self):
        if self._solution is None:
            self.refresh()
        return self._solution

    @property
    def grad_u(self):
        return FEN.grad(self.u)

    @property
    def grad_v(self):
        return FEN.grad(self.v)

    @property
    def grad_solution(self):
        return FEN.grad(self.solution)

    @property
    def F(self):
        try:
            return self.LHS - self.RHS
        except TypeError:
            return self.LHS

    @F.setter
    def F(self, new_equation):
        self.LHS = FEN.lhs(new_equation)
        self.RHS = FEN.rhs(new_equation)

    def _write_vtk(self):
        """
        Write the output to a .vtk file.
        """
        vtkfile = FEN.File(self._vtk_filename)
        vtkfile << self.solution

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
        self._u = self.solution
        for step in np.arange(self.input.n_steps):
            for expr in self.time_dependent_expressions:
                expr.t += self.input.dt
            FEN.solve(self.LHS == self.RHS, self.u, self.BC, solver_parameters=self.input.solver_parameters)
            if step % self.input.n_print == 0 or step == self.input.n_print - 1:
                self._append_to_output()
            try:
                self.assigned_u.assign(self.solution)
            except AttributeError:
                pass
        self.status.collect = True
        self.run()

    def _append_to_output(self):
        """Evaluate the result at nodes and store in the output as a numpy array."""
        nodal_solution = self.solution.compute_vertex_values(self.mesh)
        nodes = self.mesh.coordinates()
        if len(nodal_solution) != len(nodes):
            nodal_solution = nodal_solution.reshape(nodes.T.shape).T
        self.output.solution.append(nodal_solution)

    @staticmethod
    def solution_to_original_shape(nodal_solution_frame):
        """Get a frame of the solution back to its original shape, even if it was reformed to match the mesh length."""
        if len(nodal_solution_frame.shape) == 1:
            return nodal_solution_frame
        else:
            return nodal_solution_frame.T.flatten()

    def collect_output(self):
        self._write_vtk()  # TODO: Get the output files so they're all tarballed after successful runs, like other codes
        self.to_hdf()
        self.status.finished = True

    @copy_docstring(FEN.project)
    def project_function(self, v, **kwargs):
        """
        Project v onto the job's element, V.

        Args:
            v (?): The function to project.
            **kwargs: Valid `fenics.project` kwargs (except `V`, which is provided automatically).

        Returns:
            (?): Projected function.
        """
        return FEN.project(v, V=self.V, **kwargs)

    @copy_docstring(FEN.interpolate)
    def interpolate_function(self, v):
        """
        Interpolate v on the job's element, V.

        Args:
            v (?): The function to interpolate.

        Returns:
            (?): Interpolated function.
        """
        return FEN.interpolate(v, V=self.V)

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(hdf=self.project_hdf5)
        self.output.to_hdf(hdf=self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(hdf=self.project_hdf5)
        self.output.from_hdf(hdf=self.project_hdf5)

    # Convenience bindings:
    @property
    def fenics(self):
        return FEN

    @property
    def mshr(self):
        return mshr

    @property
    def sympy(self):
        return sympy

    @copy_docstring(FEN.Constant)
    def Constant(self, value):
        return FEN.Constant(value)

    @copy_docstring(FEN.Expression)
    def Expression(self, *args, **kwargs):
        return FEN.Expression(*args, **kwargs)

    @copy_docstring(FEN.Identity)
    def Identity(self, dim):
        return FEN.Identity(dim)

    @property
    @copy_docstring(FEN.dx)
    def dx(self):
        return FEN.dx

    @property
    @copy_docstring(FEN.ds)
    def ds(self):
        return FEN.ds

    @copy_docstring(FEN.grad)
    def grad(self, arg):
        return FEN.grad(arg)

    @copy_docstring(FEN.nabla_grad)
    def nabla_grad(self, arg):
        return FEN.nabla_grad(arg)

    @copy_docstring(ufl_nabla_div)
    def nabla_div(self, f):
        return ufl_nabla_div(f)

    @copy_docstring(FEN.inner)
    def inner(self, a, b):
        return FEN.inner(a, b)

    @copy_docstring(FEN.dot)
    def dot(self, arg1, arg2):
        return FEN.dot(arg1, arg2)

    @copy_docstring(FEN.tr)
    def tr(self, A):
        return FEN.tr(A)

    @copy_docstring(FEN.sqrt)
    def sqrt(self, f):
        return FEN.sqrt(f)


class Creator:
    def __init__(self, job):
        self._job = job
        self._domain = DomainFactory()
        self._bc = BoundaryConditionFactory(job)

    @property
    def domain(self):
        return self._domain

    @property
    def bc(self):
        return self._bc
