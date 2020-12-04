from __future__ import print_function
import matplotlib.pyplot as plt
import mshr as mshr
import fenics as FEN
from pyiron_base import PythonTemplateJob

class fenics(PythonTemplateJob):
    def __init__(self, project, job_name):
        super(fenics, self).__init__(project, job_name)
        self.input['LHS']= '' ## the left hand side of the equation; FEniCS function
        self.input['RHS'] = '' ## the right hand side of the equation; FEniCS function
        self._LHS=None 
        self._RHS=None
        self.input['vtk_filename']=project.name+'/output.pvd'
        self._vtk_filename = str(job_name)+'/output.pvd'
        self.input['mesh'] = None
        self._mesh = None
        self._BC = None
        self.input['BC']=None
        self.input['V']=None ## finite element volume space
        self._V = None
        self._u = None # u is the unkown function 
        self._v = None # the test function
        self.input['u']
        self.input['v'] 
        self._domain = None ## the domain


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
        return a spatial point as fenics object, based on the given coordinate
        """
        return FEN.Point(x,y)

    def grad(self, arg):
        """
        it return gradient of a given argument.
        """
        return FEN.grad(arg)

    def Circle(self,center, rad):
        """
        create a mesh on a circular domain with a radius equal to rad
        """
        return mshr.Circle(center, rad)
    
    def dxProd(self,A):
        """
        It returns the product of A and dx
        Here dx is an object from FEniCS library.
        """
        return A*FEN.dx


    def generate_mesh(self, typ, order, resolution):
        """
        This function generate the mesh based on the resolution and the job.domain
        Additionally, it creates the finite element volume "V" with the given type and order
        Moreover, the u (unknown function) and v (the test function) is initialized here.
        Arguments:
        type (str) and order (int) are the order and type of element
        resolution (int) is the resolution of the mesh 
        """
        self._mesh = mshr.generate_mesh(self._domain,resolution)
        self.input['mesh'] = self._mesh
        self._V = FEN.FunctionSpace(self._mesh, typ, order)
        self.input['V'] = self._V
        self._u = FEN.TrialFunction(self._V)
        self._v = FEN.TestFunction(self._V)
        

    def FunctionSpace(self, typ, order):
        """
        this function defines the volume function.
        typ (string), defines the type of the element; e.g. 'p' refers to langrangian element; 
                    for further information look at https://fenicsproject.org/pub/graphics/fenics-femtable-cards.pdf  
        order (int), defines the order of the element.
                    for further information look at https://fenicsproject.org/pub/graphics/fenics-femtable-cards.pdf  
        """
        return FEN.FunctionSpace(self._mesh, typ, order)

    def Constant(self, value):
        """
        returns the constant value as a function.
        Arges:
        value (float), is the value for the constant function.
        """
        return FEN.Constant(value)
    

    def DirichletBC(self, expression, boundary):
        """
        This function defines Drichlet boundary condition based on the given expression on the boundary
        Args:
        expression (string) is the expression used to evaluate the value of the unknown on the boundary
        boundary (boundary object from FEniCS module) is the spatial boundary, which the condition will be applied to
        """
        return FEN.DirichletBC(self._V, expression, boundary)

    def TrialFunction(self):
        """
        It returns a FEniCS trial function
        """
        return FEN.TrialFunction(self._V)
    
    def TestFunction(self):
        """
        It returns a FEniCS test function
        """
        return FEN.TestFunction(self._V)
    
    def dot(self, arg1, arg2):
        """
        It returns the dot product between the FEniCS objects. 
        """
        return FEN.dot(arg1, arg2)
    
    def dx(self):
        """
        It returns the FEniCS dx object.
        """
        return FEN.dx
    

    #def define_expression(expression,dictionary):


    def mesh_gen_default(self, intervals, typ='P', order=1):
        """
        creates a square with sides of 1, divided by the given intervals
        By default the type of the volume associated with the mesh
        is considered to be Lagrangian, with order 1.
        """
        self._mesh = FEN.UnitSquareMesh(intervals,intervals) 
        self.input['mesh'] = self._mesh
        self._V = FEN.FunctionSpace(self._mesh, typ, order)
        self.input['V'] = self._V
        self._u = FEN.TrialFunction(self._V)
        self._v = FEN.TestFunction(self._V)


    def BC_default(self,x,on_boundary):
        """
        return the geometrical boundary 
        """
        return on_boundary


    def write_vtk(self):
        """
        write the output to a .vtk file
        """
        vtkfile = FEN.File(self._vtk_filename)
        vtkfile << self._u

    def run_static(self):
        """
        solve a PDE based on 'LHS=RHS' using u and v as trial and test function respectively
        u is the desired unknown and RHS is the known part.
        """
        if self._mesh == None:
            print("Fatal error: no mesh is defined")
        if self._RHS == None:
            print("Fatal error: the bilinear form (RHS) is not defined")
        if self._LHS == None:
            print("Fatal error: the linear form (LHS) is not defined")
        if self._V == None:
            print("Fatal error: the volume is not defined; no V defined")
        if self._BC == None:
            print("Fatal error: the BC is not defined")
        self._u = FEN.Function(self._V)
        FEN.solve(self._LHS == self._RHS, self._u, self._BC)
        with self.project_hdf5.open("output/generic") as h5out: 
             h5out["u"] = self._u.compute_vertex_values(self._mesh)
        self.write_vtk()
        self.status.finished = True
    
    def plot_u(self):
        """
        plots the unknown u.
        """
        FEN.plot(self._u)
        #FEN.plot(self._mesh)
    def plot_mesh(self):
        """
        plots the mesh.
        """
        FEN.plot(self._mesh)
        #FEN.plot(self._mesh)