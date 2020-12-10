from __future__ import print_function
import mshr as mshr
from pyiron_base import PythonTemplateJob


class Fenics(PythonTemplateJob):
    """
    A job class for using the FEniCS library to solve a finite element method (FEM) problem.
    """

    def __init__(self, project, job_name):
        super(Fenics, self).__init__(project, job_name)
        self.input['LHS'] = ''  # the left hand side of the equation; FEniCS function
        self.input['RHS'] = ''  # the right hand side of the equation; FEniCS function
        self._LHS = None
        self._RHS = None
        self.input['vtk_filename'] = project.name+'/output.pvd'
        self._vtk_filename = str(job_name)+'/output.pvd'
        self.input['mesh'] = None
        self._mesh = None
        self._BC = None
        self.input['BC'] = None
        self.input['V'] = None  # finite element volume space
        self._V = None
        self._u = None  # u is the unkown function
        self._v = None  # the test function
        self.input['u']
        self.input['v'] 
        self._domain = None  # the domain

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, dmn):
        self._domain = dmn

    @property
    def RHS(self):
        return self._RHS

    @RHS.setter
    def RHS(self, expression):
        self.input['RHS'] = expression
        self._RHS = expression

    @property
    def mesh(self):
        return self._mesh
    
    @mesh.setter
    def mesh(self, mesh):
        self.input['mesh'] = mesh
        self._mesh = mesh
        
    @property
    def V(self):
        return self._V
    
    @V.setter
    def V(self, vol):
        self.input['V'] = vol
        self._V = vol

    @property
    def u(self):
        return self._u
    
    @u.setter
    def u(self, exp):
        self.input['u'] = exp
        self._u = exp
    
    @property
    def v(self):
        return self._v
    
    @v.setter
    def v(self, exp):
        self.input['v'] = exp
        self._v = exp
    
    @property
    def BC(self):
        return self._BC
    
    @BC.setter
    def BC(self, boundary):
        self.input['BC'] = boundary
        self._BC = boundary

    @property
    def LHS(self):
        return self._LHS
    
    @LHS.setter
    def LHS(self, expression):
        self.input['LHS'] = expression
        self._LHS = expression

    @property
    def vtk_filename(self):
        return self.input['vtk_filename']
    
    @vtk_filename.setter
    def vtk_filename(self, filename):
        self.input['vtk_filename'] = filename
        self._vtk_filename = filename

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
        self._mesh = mshr.generate_mesh(self._domain, resolution)
        self.input['mesh'] = self._mesh
        self._V = FEN.FunctionSpace(self._mesh, typ, order)
        self.input['V'] = self._V
        self._u = FEN.TrialFunction(self._V)
        self._v = FEN.TestFunction(self._V)

    def FunctionSpace(self, typ, order):
        """
        Returns the volume function using the current mesh.

        Args:
            typ (string): The type of the element; e.g. 'p' refers to langrangian element. For further information see
                https://fenicsproject.org/pub/graphics/fenics-femtable-cards.pdf
            order (int): The order of the element. For further information see
                https://fenicsproject.org/pub/graphics/fenics-femtable-cards.pdf
        """
        return FEN.FunctionSpace(self._mesh, typ, order)

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
        return FEN.DirichletBC(self._V, expression, boundary)

    def TrialFunction(self):
        """
        Returns a FEniCS trial function
        """
        return FEN.TrialFunction(self._V)
    
    def TestFunction(self):
        """
        Returns a FEniCS test function
        """
        return FEN.TestFunction(self._V)
    
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
        self._mesh = FEN.UnitSquareMesh(intervals, intervals)
        self.input['mesh'] = self._mesh
        self._V = FEN.FunctionSpace(self._mesh, typ, order)
        self.input['V'] = self._V
        self._u = FEN.TrialFunction(self._V)
        self._v = FEN.TestFunction(self._V)

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
        vtkfile << self._u

    def run_static(self):
        """
        Solve a PDE based on 'LHS=RHS' using u and v as trial and test function respectively. Here, u is the desired
        unknown and RHS is the known part.
        """
        if self._mesh is None:
            print("Fatal error: no mesh is defined")
        if self._RHS is None:
            print("Fatal error: the bilinear form (RHS) is not defined")
        if self._LHS is None:
            print("Fatal error: the linear form (LHS) is not defined")
        if self._V is None:
            print("Fatal error: the volume is not defined; no V defined")
        if self._BC is None:
            print("Fatal error: the BC is not defined")
        self._u = FEN.Function(self._V)
        FEN.solve(self._LHS == self._RHS, self._u, self._BC)
        with self.project_hdf5.open("output/generic") as h5out:
            h5out["u"] = self._u.compute_vertex_values(self._mesh)
        self.write_vtk()
        self.status.finished = True
    
    def plot_u(self):
        """
        Plots the unknown u.
        """
        FEN.plot(self._u)

    def plot_mesh(self):
        """
        Plots the mesh.
        """
        FEN.plot(self._mesh)
