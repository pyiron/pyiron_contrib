# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Store Fenics job plotting routines as a helper to declutter the main class.
"""

from matplotlib.docstring import copy as copy_docstring
from pyiron_contrib.continuum.fenics.fix_plotting import plot as modified_fenics_plot
import fenics as FEN
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.colors import LogNorm

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


class Plot:
    """A helper class to store Fenics job plotting methods."""

    def __init__(self, job):
        """Initialize the plot helper and store the parent job."""
        self._job = job

    @copy_docstring(modified_fenics_plot)
    def plot(self, object, *args, **kwargs):
        return modified_fenics_plot(object, *args, **kwargs)

    def __call__(self, object, *args, **kwargs):
        return self.plot(object, *args, **kwargs)

    def solution(self):
        FEN.plot(self._job.solution)

    def mesh(self):
        FEN.plot(self._job.mesh)

    @staticmethod
    def _nodes_to_2d(nodes, projection_axis):
        if len(nodes) == 3:
            axes = [0, 1, 2]
            axes.pop(projection_axis)
            return nodes[axes]
        elif len(nodes) != 2:
            raise ValueError("Expected nodes to be in 2- or 3-dimensions, but got node shape {}".format(nodes.T.shape))
        else:
            return nodes

    @staticmethod
    def _nodal_values_to_1d(nodal_values):
        if len(nodal_values.shape) > 1:
            if len(nodal_values.shape) != 2:
                raise ValueError(
                    "Expected nodal values to have shape (n,) or (n, m) but got {}".format(nodal_values.shape)
                )
            nodal_values = np.linalg.norm(nodal_values, axis=-1)
        return nodal_values

    def nodal2d(
            self,
            nodal_values,
            nodes=None,
            n_grid=1000,
            n_grid_x=None,
            n_grid_y=None,
            add_colorbar=True,
            lognorm=False,
            projection_axis=None
    ):
        """
        Plot a heatmap of nodal values (or their magnitude if vectors) interpolated onto a uniform grid.

        Based off of
        [matplotlib docs](https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/irregulardatagrid.html)

        Args:
            nodal_values (numpy.ndarray): A scalar array with the same length and order as the mesh points, e.g. a frame
                of scalar output. Shape should be `(n_nodes)` or `(n_nodes, n_dims)`. Note: higher dimensional values
            nodes (numpy.ndarray): xy- or xyz-positions of the nodes the values belong to. (Default is None, use mesh
                nodes.) Shape should be `(n_nodes, dim)`.
            n_grid (int): Number of points to use when interpolating the mesh values. (Default is 1000.)
            n_grid_x (int): Number of grid points to use when interpolating the mesh values in the x-direction.
                (Default is None, use n_grid value.)
            n_grid_y (int): Number of grid points to use when interpolating the mesh values in the y-direction.
                (Default is None, use n_grid value.)
            add_colorbar (bool): Whether or not to add the colorbar to the figure. (Default is True.)
            lognorm (bool): Normalize the colorscheme to a log scale. (Default is False.)
            projection_axis (int): The axis onto which to project mesh nodes and nodal values if the data is 3d.
                (Default is None, project z onto xy-plane.)

        Returns:
            (matplotlib.image.AxesImage): The imshow object.
            (matplotlib.figure.Figure): The parent figure.
            (matplotlib.axes._subplots.AxesSubplot): The subplots axis on which the plotting occurs.
        """
        n_grid_x = n_grid_x or n_grid
        n_grid_y = n_grid_y or n_grid

        nodes = self._job.mesh.coordinates() if nodes is None else nodes
        nodes = nodes.T

        projection_axis = -1 if projection_axis is None else projection_axis
        projection_axis = projection_axis % len(nodes)

        nodes = self._nodes_to_2d(nodes, projection_axis)
        nodal_values = self._nodal_values_to_1d(nodal_values)

        # Create grid values first.
        mesh_X, mesh_Y = nodes
        xi = np.linspace(np.amin(mesh_X), np.amax(mesh_X), n_grid_x)
        yi = np.linspace(np.amin(mesh_Y), np.amax(mesh_Y), n_grid_y)

        # Perform linear interpolation of the data (x,y)
        # on a grid defined by (xi,yi)
        triang = tri.Triangulation(mesh_X, mesh_Y)
        interpolator = tri.LinearTriInterpolator(triang, nodal_values)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = interpolator(Xi, Yi)

        fig, ax = plt.subplots()
        heat = ax.imshow(
            Zi[::-1],
            aspect='equal',
            cmap=plt.cm.viridis,
            extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
            norm=LogNorm() if lognorm else None
        )
        if add_colorbar:
            fig.colorbar(heat, shrink=0.5, aspect=10)
        return heat, fig, ax

    def nodal3d(self, nodal_values, nodes=None, add_colorbar=True):
        """
        3d scatter plot of nodal values.

        Args:
            nodal_values (numpy.ndarray): A scalar array with the same length and order as the mesh points, e.g. a frame
                of scalar output. Shape should be `(n_nodes)` or `(n_nodes, n_dims)`. Note: higher dimensional values
            nodes (numpy.ndarray): xyz-positions of the nodes the values belong to. (Default is None, use mesh nodes.)
                Shape should be `(n_nodes, 3)`.
            add_colorbar (bool): Whether or not to add the colorbar to the figure. (Default is True.)
            lognorm (bool): Normalize the colorscheme to a log scale. (Default is False.)
            projection_axis (int): The axis onto which to project mesh nodes and nodal values if the data is 3d.
                (Default is None, project z onto xy-plane.)

        Returns:
            (matplotlib.image.AxesImage): The scatter object.
            (matplotlib.figure.Figure): The parent figure.
            (matplotlib.axes._subplots.AxesSubplot): The subplots axis on which the plotting occurs.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        nodal_values = self._nodal_values_to_1d(nodal_values)
        nodes = self._job.mesh.coordinates() if nodes is None else nodes
        if nodes.shape[1] != 3:
            raise ValueError("Expected nodes to have shape (n, 3)  but got {}".format(nodes.shape))

        scattered = ax.scatter(*nodes.T, zdir='z', s=20, c=nodal_values, depthshade=True, cmap=plt.cm.viridis)
        if add_colorbar:
            fig.colorbar(scattered, shrink=0.5, aspect=10)
        return scattered, fig, ax

    def solution2d(
            self,
            frame=-1,
            n_grid=1000,
            n_grid_x=None,
            n_grid_y=None,
            add_colorbar=True,
            lognorm=False,
            projection_axis=None
    ):
        """
        Plot a heatmap of solution magnitudes interpolated onto a uniform grid.

        Args:
            frame (int): Which output frame to use. (Default is -1, most recent.)
            n_grid (int): Number of points to use when interpolating the mesh values. (Default is 1000.)
            n_grid_x (int): Number of grid points to use when interpolating the mesh values in the x-direction.
                (Default is None, use n_grid value.)
            n_grid_y (int): Number of grid points to use when interpolating the mesh values in the y-direction.
                (Default is None, use n_grid value.)
            add_colorbar (bool): Whether or not to add the colorbar to the figure. (Default is True.)
            lognorm (bool): Normalize the colorscheme to a log scale. (Default is False.)
            projection_axis (int): The axis onto which to project mesh nodes and nodal values if the data is 3d.
                (Default is None, project z onto xy-plane if needed.)

        Returns:
            (matplotlib.image.AxesImage): The imshow object.
            (matplotlib.figure.Figure): The parent figure.
            (matplotlib.axes._subplots.AxesSubplot): The subplots axis on which the plotting occurs.
        """
        return self.nodal2d(
            nodal_values=self._job.output.solution[frame],
            n_grid=n_grid,
            n_grid_x=n_grid_x,
            n_grid_y=n_grid_y,
            add_colorbar=add_colorbar,
            lognorm=lognorm,
            projection_axis=projection_axis
        )

    def solution3d(self, frame=-1, nodes=None, add_colorbar=True):
        """
        3d scatter plot of the solution magnitude.

        Args:
            frame (int): Which output frame to use. (Default is -1, most recent.)
            nodes (numpy.ndarray): xyz-positions of the nodes the values belong to. (Default is None, use mesh nodes.)
                Shape should be `(n_nodes, 3)`.
            add_colorbar (bool): Whether or not to add the colorbar to the figure. (Default is True.)

        Returns:
            (matplotlib.image.AxesImage): The scatter object.
            (matplotlib.figure.Figure): The parent figure.
            (matplotlib.axes._subplots.AxesSubplot): The subplots axis on which the plotting occurs.
        """
        return self.nodal3d(nodal_values=self._job.output.solution[frame], nodes=nodes, add_colorbar=add_colorbar)
