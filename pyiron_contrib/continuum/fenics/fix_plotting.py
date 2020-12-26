# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Aspect ratios of 'equal' on matplotlib 3d plots [doesn't work](https://github.com/matplotlib/matplotlib/issues/15382)
but the fenics plotting command tries to do this. I don't want to make a fork of or PR to fenics right now, so just
copy and modify the functionality
"""

import os
from dolfin.common.plotting import (
    _has_matplotlib,
    _all_plottable_types,
    _plot_x3dom,
    _matplotlib_plottable_types,
    mplot_mesh,
    mplot_dirichletbc,
    mplot_expression,
    mplot_function,
    mplot_meshfunction,
    _meshfunction_types
)
import dolfin.cpp as cpp
import ufl

__author__ = "Liam Huber"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Dec 26, 2020"


def plot(object, *args, **kwargs):
    """
    Plot given object.

    *Arguments*
        object
            a :py:class:`Mesh <dolfin.cpp.Mesh>`, a :py:class:`MeshFunction
            <dolfin.cpp.MeshFunction>`, a :py:class:`Function
            <dolfin.functions.function.Function>`, a :py:class:`Expression`
            <dolfin.cpp.Expression>, a :py:class:`DirichletBC`
            <dolfin.cpp.DirichletBC>, a :py:class:`FiniteElement
            <ufl.FiniteElement>`.

    *Examples of usage*
        In the simplest case, to plot only e.g. a mesh, simply use

        .. code-block:: python

            mesh = UnitSquare(4, 4)
            plot(mesh)

        Use the ``title`` argument to specify title of the plot

        .. code-block:: python

            plot(mesh, tite="Finite element mesh")

        It is also possible to plot an element

        .. code-block:: python

            element = FiniteElement("BDM", tetrahedron, 3)
            plot(element)

        Vector valued functions can be visualized with an alternative mode

        .. code-block:: python

            plot(u, mode = "glyphs")

        A more advanced example

        .. code-block:: python

            plot(u,
                 wireframe = True,              # use wireframe rendering
                 interactive = False,           # do not hold plot on screen
                 scalarbar = False,             # hide the color mapping bar
                 hardcopy_prefix = "myplot",    # default plotfile name
                 scale = 2.0,                   # scale the warping/glyphs
                 title = "Fancy plot",          # set your own title
                 )

    """

    # Return if plotting is disables
    if os.environ.get("DOLFIN_NOPLOT", "0") != "0":
        return

    # Return if Matplotlib is not available
    if not _has_matplotlib():
        cpp.log.info("Matplotlib is required to plot from Python.")
        return

    # Plot element
    if isinstance(object, ufl.FiniteElementBase):
        import ffc
        return ffc.plot(object, *args, **kwargs)

    # For dolfin.function.Function, extract cpp_object
    if hasattr(object, "cpp_object"):
        object = object.cpp_object()

    # Get mesh from explicit mesh kwarg, only positional arg, or via
    # object
    mesh = kwargs.pop('mesh', None)
    if isinstance(object, cpp.mesh.Mesh):
        if mesh is not None and mesh.id() != object.id():
            raise RuntimeError("Got different mesh in plot object and keyword argument")
        mesh = object
    if mesh is None:
        if isinstance(object, cpp.function.Function):
            mesh = object.function_space().mesh()
        elif hasattr(object, "mesh"):
            mesh = object.mesh()

    # Expressions do not carry their own mesh
    if isinstance(object, cpp.function.Expression) and mesh is None:
        raise RuntimeError("Expecting a mesh as keyword argument")

    backend = kwargs.pop("backend", "matplotlib")
    if backend not in ("matplotlib", "x3dom"):
        raise RuntimeError("Plotting backend %s not recognised" % backend)

    # Try to project if object is not a standard plottable type
    if not isinstance(object, _all_plottable_types):
        from dolfin.fem.projection import project
        try:
            cpp.log.info("Object cannot be plotted directly, projecting to "
                         "piecewise linears.")
            object = project(object, mesh=mesh)
            mesh = object.function_space().mesh()
            object = object._cpp_object
        except Exception as e:
            msg = "Don't know how to plot given object:\n  %s\n" \
                  "and projection failed:\n  %s" % (str(object), str(e))
            raise RuntimeError(msg)

    # Plot
    if backend == "matplotlib":
        return _plot_matplotlib(object, mesh, kwargs)
    elif backend == "x3dom":
        return _plot_x3dom(object, kwargs)
    else:
        assert False, "This code should not be reached."


def _plot_matplotlib(obj, mesh, kwargs):
    if not isinstance(obj, _matplotlib_plottable_types):
        print("Don't know how to plot type %s." % type(obj))
        return

    # Plotting is not working with all ufl cells
    if mesh.ufl_cell().cellname() not in ['interval', 'triangle', 'tetrahedron']:
        raise AttributeError(("Matplotlib plotting backend doesn't handle %s mesh.\n"
                              "Possible options are saving the output to XDMF file "
                              "or using 'x3dom' backend.") % mesh.ufl_cell().cellname())

    # Avoid importing pyplot until used
    try:
        import matplotlib.pyplot as plt
    except Exception:
        cpp.warning("matplotlib.pyplot not available, cannot plot.")
        return

    gdim = mesh.geometry().dim()
    if gdim == 3 or kwargs.get("mode") in ("warp",):
        # Importing this toolkit has side effects enabling 3d support
        from mpl_toolkits.mplot3d import axes3d  # noqa
        # Enabling the 3d toolbox requires some additional arguments
        ax = plt.gca(projection='3d')
    else:
        ax = plt.gca()
    if mesh.geometry().dim() < 3:
        ax.set_aspect('equal')

    title = kwargs.pop("title", None)
    if title is not None:
        ax.set_title(title)

    # Translate range_min/max kwargs supported by VTKPlotter
    vmin = kwargs.pop("range_min", None)
    vmax = kwargs.pop("range_max", None)
    if vmin and "vmin" not in kwargs:
        kwargs["vmin"] = vmin
    if vmax and "vmax" not in kwargs:
        kwargs["vmax"] = vmax

    # Drop unsupported kwargs and inform user
    _unsupported_kwargs = ["rescale", "wireframe"]
    for kw in _unsupported_kwargs:
        if kwargs.pop(kw, None):
            cpp.warning("Matplotlib backend does not support '%s' kwarg yet. "
                        "Ignoring it..." % kw)

    if isinstance(obj, cpp.function.Function):
        return mplot_function(ax, obj, **kwargs)
    elif isinstance(obj, cpp.function.Expression):
        return mplot_expression(ax, obj, mesh, **kwargs)
    elif isinstance(obj, cpp.mesh.Mesh):
        return mplot_mesh(ax, obj, **kwargs)
    elif isinstance(obj, cpp.fem.DirichletBC):
        return mplot_dirichletbc(ax, obj, **kwargs)
    elif isinstance(obj, _meshfunction_types):
        return mplot_meshfunction(ax, obj, **kwargs)
    else:
        raise AttributeError('Failed to plot %s' % type(obj))