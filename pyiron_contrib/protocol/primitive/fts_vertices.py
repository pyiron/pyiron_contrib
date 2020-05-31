# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from pyiron_contrib.protocol.generic import PrimitiveVertex
from pyiron_contrib.protocol.utils import IODictionary
import numpy as np
from pyiron.atomistics.job.interactive import GenericInteractive
from pyiron.lammps.lammps import LammpsInteractive
from scipy.constants import physical_constants
from ase.geometry import find_mic, get_distances  # TODO: Wrap things using pyiron functionality
from pyiron import Project
from pyiron_contrib.protocol.utils import ensure_iterable
from os.path import split
from abc import ABC, abstractmethod
from scipy.linalg import toeplitz

"""
Vertices whose present application extends only to finite temperature string-based protocols.

TODO: Naming consistency among vertices, e.g. `all_centroid_positions` vs `centroid_pos_list`.
"""

__author__ = "Raynol Dsouza, Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "20 July, 2019"


class StringDistances(PrimitiveVertex):
    """
    A parent class for vertices which care about the distance from an image to various centroids on the string.
    """
    def __init__(self, name=None):
        super(PrimitiveVertex, self).__init__(name=name)
        self.input.default.eps = 1e-6

    @abstractmethod
    def command(self, *args, **kwargs):
        pass

    @staticmethod
    def check_closest_to_parent(positions, centroid_positions, all_centroid_positions, cell, pbc, eps):
        """
        Checks which centroid the image is closest too, then measures whether or not that closest centroid is sufficiently
        close to the image's parent centroid.

        Args:
            positions (numpy.ndarray): Atomic positions of this image.
            centroid_positions (numpy.ndarray): The positions of the image's centroid.
            all_centroid_positions (list/numpy.ndarray): A list of positions for all centroids in the string.
            cell (numpy.ndarray): The 3x3 cell vectors for pbcs.
            pbc (numpy.ndarray): Three booleans declaring which dimensions have periodic boundary conditions for finding
                the minimum distance convention.
            eps (float): The maximum distance between the closest centroid and the parent centroid to be considered a match
                (i.e. no recentering necessary).

        Returns:
            (bool): Whether the image is closest to its own parent centroid.
        """
        distances = [np.linalg.norm(find_mic(c_pos - positions, cell, pbc)[0]) for c_pos in all_centroid_positions]
        closest_centroid_positions = all_centroid_positions[np.argmin(distances)]
        match_distance = np.linalg.norm(find_mic(closest_centroid_positions - centroid_positions, cell, pbc)[0])
        return match_distance < eps


class StringRecenter(StringDistances):
    """
    If not, the image's positions and forces are reset to match its centroid.

    Input attributes:
        positions (numpy.ndarray): Atomic positions of the image.
        forces (numpy.ndarray): Atomic forces on the image.
        centroid_positions (numpy.ndarray): The positions of the image's centroid.
        centroid_forces (numpy.ndarray): The forces on the image's centroid.
        all_centroid_positions (list/numpy.ndarray): A list of positions for all centroids in the string.
        cell (numpy.ndarray): The 3x3 cell vectors for pbcs.
        pbc (numpy.ndarray): Three booleans declaring which dimensions have periodic boundary conditions for finding
            the minimum distance convention.
        eps (float): The maximum distance between the closest centroid and the parent centroid to be considered a match
            (i.e. no recentering necessary). (Default is 1e-6.)

    Output attributes:
        positions (numpy.ndarray): Either the original positions passed in, or the centroid positions.
        forces (numpy.ndarray): Either the original forces passed in, or the centroid forces.
        recentered (bool): Whether or not the image got recentered.
    """
    def command(self, positions, forces, centroid_positions, centroid_forces, all_centroid_positions, cell, pbc, eps):
        if self.check_closest_to_parent(positions, centroid_positions, all_centroid_positions, cell, pbc, eps):
            return {
                'positions': positions,
                'forces': forces,
                'recentered': False,
            }
        else:
            return {
                'positions': centroid_positions,
                'forces': centroid_forces,
                'recentered': True
            }


class StringReflect(StringDistances):
    """
    If not, the image's positions and forces are reset to match its centroid.

    Input attributes:
        positions (numpy.ndarray): Atomic positions of the image.
        velocities (numpy.ndarray): Atomic velocities of the image.
        previous_positions (numpy.ndarray): Atomic positions of the image from the previous timestep.
        previous_velocities (numpy.ndarray): Atomic velocities of the image from the previous timestep.
        centroid_positions (numpy.ndarray): The positions of the image's centroid.
        centroid_forces (numpy.ndarray): The forces on the image's centroid.
        all_centroid_positions (list/numpy.ndarray): A list of positions for all centroids in the string.
        cell (numpy.ndarray): The 3x3 cell vectors for pbcs.
        pbc (numpy.ndarray): Three booleans declaring which dimensions have periodic boundary conditions for finding
            the minimum distance convention.
        eps (float): The maximum distance between the closest centroid and the parent centroid to be considered a match
            (i.e. no recentering necessary). (Default is 1e-6.)
        forces (numpy.ndarray): Forces corresponding to the input state. (Default is None.)
        previous_forces (numpy.ndarray): Forces corresponding to the previous state. (Default is None.)

    Output attributes:
        positions (numpy.ndarray): Either the original positions passed in, or the previous ones.
        forces (numpy.ndarray): Either the original forces passed in, or the previous ones.
        reflected (bool): Whether or not the image got recentered.
    """
    def __init__(self, name=None):
        super(StringReflect, self).__init__(name=name)
        self.input.default.forces = None
        self.input.default.previous_forces = None

    def command(self, positions, velocities, previous_positions, previous_velocities, centroid_positions,
                all_centroid_positions, cell, pbc, eps, forces, previous_forces):
        if self.check_closest_to_parent(positions, centroid_positions, all_centroid_positions, cell, pbc, eps):
            return {
                'positions': positions,
                'velocities': velocities,
                'forces': forces,
                'reflected': False
            }
        else:
            return {
                'positions': previous_positions,
                'velocities': -previous_velocities,
                'forces': previous_forces,
                'reflected': True
            }


class PositionsRunningAverage(PrimitiveVertex):
    """
    Calculates the running average of input positions at each call.

    Input attributes:
        positions_list (list/numpy.ndarray): The instantaneous position, which will be updated to the running average
        running_average_list (list/numpy.ndarray): List of existing running averages
        cell (numpy.ndarray): The cell of the structure
        pbc (numpy.ndarray): Periodic boundary condition of the structure

    Output attributes:
        running_average_list (list/numpy.ndarray): The updated running average list

    TODO:
        Handle non-static cells, or at least catch them.
        Refactor this so there are list and serial versions equally available
    """

    def __init__(self, name=None):
        super(PositionsRunningAverage, self).__init__(name=name)
        self._divisor = 1

    def command(self, positions_list, running_average_list, relax_endpoints, cell, pbc):
        # On the first step, divide by 2 to average two positions
        self._divisor += 1
        # How much of the current step to mix into the average
        weight = 1. / self._divisor
        running_average_list = np.array(running_average_list)  # Don't modify this input in place

        for i, pos in enumerate(positions_list):
            if (i == 0 or i == len(positions_list) - 1) and not relax_endpoints:
                continue
            else:
                disp = find_mic(pos - running_average_list[i], cell, pbc)[0]
                running_average_list[i] += weight * disp

        return {
            'running_average_list': running_average_list
        }


class CentroidsRunningAverageMix(PrimitiveVertex):
    """
    Mix in the running average of the positions to the centroid, moving the centroid towards that
    running average by a fraction.

    Input attributes:
        mixing_fraction (float): The fraction of the running average to mix into centroid (Default is 0.1)
        centroids_pos_list (list/numpy.ndarray): List of all the centroids along the string
        running_average_list (list/numpy.ndarray): List of running averages
        cell (numpy.ndarray): The cell of the structure
        pbc (numpy.ndarray): Periodic boundary condition of the structure

    Output attributes:
        centroids_pos_list (list/numpy.ndarray): List centroids updated towards the running average

    TODO:
        Re-write Command base class(es) to better handle serial/list/parallel.
    """

    def __init__(self, name=None):
        super(CentroidsRunningAverageMix, self).__init__(name=name)
        self.input.default.mixing_fraction = 0.1

    def command(self, mixing_fraction, centroids_pos_list, running_average_list, cell, pbc):
        centroids_pos_list = np.array(centroids_pos_list)
        for i, cent in enumerate(centroids_pos_list):
            disp = find_mic(running_average_list[i] - cent, cell, pbc)[0]
            update = mixing_fraction * disp
            centroids_pos_list[i] += update

        return {
            'centroids_pos_list': centroids_pos_list
        }


class CentroidsSmoothing(PrimitiveVertex):
    """
    Global smoothing following Vanden-Eijnden and Venturoli (2009). The actual smoothing strength is the product of the
    nominal smoothing strength (`kappa`), the number of images, and the mixing fraction (`dtau`).

    Input Attributes:
        kappa (float): Nominal smoothing strength.
        dtau (float): Mixing fraction (from updating the string towards the moving average of the image positions).
        all_centroid_positions (list/numpy.ndarray): List of all the centroid positions along the string.

    Output Attributes:
        all_centroid_positions (list/numpy.ndarray): List of smoothed centroid positions.
    """
    def command(self, kappa, dtau, all_centroid_positions):
        # Get the smoothing matrix
        n_images = len(all_centroid_positions)
        smoothing_strength = kappa * n_images * dtau
        smoothing_matrix = self._get_smoothing_matrix(n_images, smoothing_strength)
        smoothed_centroid_positions = np.tensordot(smoothing_matrix, np.array(all_centroid_positions), axes=1)

        return {
            'all_centroid_positions': smoothed_centroid_positions
        }

    @staticmethod
    def _get_smoothing_matrix(n_images, smoothing_strength):
        """
        A function that returns the smoothing matrix used in global smoothing.

        Attributes:
            n_images (int): number of images
            smoothing_strength (float): the smoothing penalty

        Returns:
            smoothing_matrix
        """
        toeplitz_rowcol = np.zeros(n_images)
        toeplitz_rowcol[0] = -2
        toeplitz_rowcol[1] = 1
        second_order_deriv = toeplitz(toeplitz_rowcol, toeplitz_rowcol)
        second_order_deriv[0] = np.zeros(n_images)
        second_order_deriv[-1] = np.zeros(n_images)
        smooth_mat_inv = np.eye(n_images) - smoothing_strength * second_order_deriv

        return np.linalg.inv(smooth_mat_inv)


class CentroidsReparameterization(PrimitiveVertex):
    """
    Use linear interpolation to equally space the jobs between the first and last job in 3N dimensional space,
        using a piecewise function

    Input attributes:
        centroids_pos_list (list/numpy.ndarray): List of all the centroids along the string
        cell (numpy.ndarray): The cell of the structure
        pbc (numpy.ndarray): Periodic boundary condition of the structure

    Output attributes:
        centroids_pos_list (list/numpy.ndarray): List of equally spaced centroids
    """

    def __init__(self, name=None):
        super(CentroidsReparameterization, self).__init__(name=name)

    def command(self, centroids_pos_list, cell, pbc):
        # How long is the piecewise parameterized path to begin with?
        lengths = self._find_lengths(centroids_pos_list, cell, pbc)
        length_tot = lengths[-1]
        length_per_frame = length_tot / (len(centroids_pos_list) - 1)

        # Find new positions for the re-parameterized jobs
        new_positions = [centroids_pos_list[0]]
        for n_left, cent in enumerate(centroids_pos_list[1:-1]):
            n = n_left + 1
            length_target = n * length_per_frame

            # Find the last index not in excess of the target length
            try:
                all_not_over = np.argwhere(lengths < length_target)
                highest_not_over = np.amax(all_not_over)
            except ValueError:
                # If all_not_over is empty
                highest_not_over = 0

            # Interpolate from the last position not in excess
            start = centroids_pos_list[highest_not_over]
            end = centroids_pos_list[highest_not_over + 1]
            disp = find_mic(end - start, cell, pbc)[0]
            interp_dir = disp / np.linalg.norm(disp)
            interp_mag = length_target - lengths[highest_not_over]

            new_positions.append(start + interp_mag * interp_dir)
        new_positions.append(centroids_pos_list[-1])

        # Apply the new positions all at once
        centroids_pos_list = new_positions

        return {
            'centroids_pos_list': centroids_pos_list
        }

    @staticmethod
    def _find_lengths(a_list, cell, pbc):
        """
        Finds the cummulative distance from job to job.

        Attribute:
            a_list (list/numpy.ndarray): List of positions whose lengths are to be calculated
            cell (numpy.ndarray): The cell of the structure
            pbc (numpy.ndarray): Periodic boundary condition of the structure

        Returns:
            lengths (list): Lengths of the positions in the list
        """
        length_cummulative = 0
        lengths = [length_cummulative]
        # First length is zero, all other lengths are wrt the first position in the list
        for n_left, term in enumerate(a_list[1:]):
            disp = find_mic(term - a_list[n_left], cell, pbc)[0]
            length_cummulative += np.linalg.norm(disp)
            lengths.append(length_cummulative)
        return lengths


class MilestoningVertex(PrimitiveVertex):
    """
    Checks whether positions are closest to its own centroid. If not, reverses the velocities, and resets the
        positions and forces to the values from the previous step. Logs total reflections, reflections off edges
        and time spent at each edge

    Input attributes:
        positions_list (list/numpy.ndarray): List of current positions
        velocities_list (list/numpy.ndarray): List of current velocities
        forces_list (list/numpy.ndarray): List of current forces
        previous_positions_list (list/numpy.ndarray): List of previous positions
        prev_velocities_list (list/numpy.ndarray): List of previous velocities
        prev_forces_list (list/numpy.ndarray): List of previous forces
        centroids_pos_list (list/numpy.ndarraymile.graph.milestone): List of all the centroids along the string
        thermalization_steps (int): Number of steps after which to start tracking reflections.
        cell (numpy.ndarray): The cell of the structure
        pbc (numpy.ndarray): Periodic boundary condition of the structure

    Output attributes:
        positions_list (list/numpy.ndarray): The modified positions
        velocities_list (list/numpy.ndarray): The modified velocities
        forces_list (list/numpy.ndarray): The modified forces
        reflections_matrix (numpy.ndarray): The reflections matrix
        edge_reflections_matrix (numpy.ndarray): The edge reflections matrix
        edge_time_matrix (numpy.ndarray): The edge time matrix
    """

    def __init__(self, name=None):
        super(MilestoningVertex, self).__init__(name=name)
        # initialize matrices to be tracked
        self.tracker_list = None
        self.reflections_matrix = None
        self.edge_reflections_matrix = None
        self.edge_time_matrix = None

    def command(self, positions_list, velocities_list, forces_list, prev_positions_list, prev_velocities_list,
                prev_forces_list, all_centroid_positions, thermalization_steps, cell, pbc):
        n_images = len(all_centroid_positions)
        n_edges = int(n_images * (n_images - 1) / 2)
        current_step = self.archive.clock

        reflected_positions = []
        reflected_velocities = []
        reflected_forces = []

        if current_step == 0:
            # form the matrices
            self.tracker_list = [[None, None] for _ in np.arange(n_images)]
            self.reflections_matrix = np.zeros((n_images, n_images))
            self.edge_reflections_matrix = np.array([np.zeros((n_edges, n_edges)) for _ in np.arange(n_images)])
            self.edge_time_matrix = np.array([np.zeros(n_edges) for _ in np.arange(n_images)])

        # Find distance between positions and each of the centroids
        for n, (pos, cent) in enumerate(zip(positions_list, all_centroid_positions)):
            closest_centroid_id = self.get_closest_centroid_index(pos, all_centroid_positions, cell, pbc)
            tracker = self.tracker_list[n]

            if closest_centroid_id == n:
                # Return current positions, velocities and forces
                reflected_positions.append(positions_list[n])
                reflected_velocities.append(velocities_list[n])
                reflected_forces.append(forces_list[n])

                if current_step >= thermalization_steps:
                    # Start reflection tracking
                    if tracker[1] is not None:  # If no reflections, increment time
                        self.edge_time_matrix[n][tracker[1]] += 1
                    # End reflection tracking
            else:
                # Update previous positions, velocities and forces, if positions are not closest to parent centroid
                reflected_positions.append(prev_positions_list[n])
                reflected_velocities.append(-prev_velocities_list[n])
                reflected_forces.append(prev_forces_list[n])

                if current_step >= thermalization_steps:
                    # Start reflection tracking
                    self.reflections_matrix[n, closest_centroid_id] += 1  # Save the reflection
                    indices = np.zeros((n_images, n_images))  # images x images
                    indices[n, closest_centroid_id] = 1  # Record the edge
                    ind = np.tril(indices) + np.triu(indices).T  # Convert to triangular matrix
                    # Record the index of the edge (N_j)
                    n_j = int(np.nonzero(ind[np.tril_indices(n_images, k=-1)])[0][0])

                    if tracker[1] is None:
                        tracker[1] = n_j  # Initialize N_j
                    elif tracker[1] == n_j:
                        self.edge_time_matrix[n][n_j] += 1
                    else:
                        tracker[0] = tracker[1]  # If reflecting off a different edge, change N_j to N_i
                        tracker[1] = n_j  # Set new N_j
                        self.edge_reflections_matrix[n][tracker[0], tracker[1]] += 1
                        # End reflection tracking

        return {
            'positions_list': reflected_positions,
            'velocities_list': reflected_velocities,
            'forces_list': reflected_forces,
            'reflections_matrix': self.reflections_matrix,
            'edge_reflections_matrix': self.edge_reflections_matrix,
            'edge_time_matrix': self.edge_time_matrix
        }

    @staticmethod
    def get_closest_centroid_index(positions, all_centroid_positions, cell, pbc):
        distances = [np.linalg.norm(find_mic(c_pos - positions, cell, pbc)[0]) for c_pos in all_centroid_positions]
        return np.argmin(distances)
