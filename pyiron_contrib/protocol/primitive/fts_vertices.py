# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from pyiron_contrib.protocol.generic import PrimitiveVertex
import numpy as np
from scipy.constants import femto as FS
from ase.geometry import find_mic  # TODO: Wrap things using pyiron functionality
from abc import ABC, abstractmethod
from scipy.linalg import toeplitz

"""
Vertices whose present application extends only to finite temperature string-based protocols.

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

    def command(self, positions, velocities, previous_positions, previous_velocities, centroid_positions,
                all_centroid_positions, cell, pbc, eps):
        if self.check_closest_to_parent(positions, centroid_positions, all_centroid_positions, cell, pbc, eps):
            return {
                'positions': positions,
                'velocities': velocities,
                'reflected': False
            }
        else:
            return {
                'positions': previous_positions,
                'velocities': -previous_velocities,
                'reflected': True
            }


class PositionsRunningAverage(PrimitiveVertex):
    """
    Calculates the running average of input positions at each call.

    Input attributes:
        positions (list/numpy.ndarray): The instantaneous position, which will be updated to the running average.
        running_average_positions (list/numpy.ndarray): The running average of positions.
        cell (numpy.ndarray): The cell of the structure.
        pbc (numpy.ndarray): Periodic boundary condition of the structure.
        divisor (int): The divisor for the running average positions. Increments by 1, each time the vertex is
            called. (Default is 1.)

    Output attributes:
        running_average_positions (list/numpy.ndarray): The updated running average list.
        divisor (int): The updated divisor.

    TODO:
        Handle non-static cells, or at least catch them.
        Refactor this so there are list and serial versions equally available
    """

    def __init__(self, name=None):
        super(PositionsRunningAverage, self).__init__(name=name)
        self.input.total_steps = 0
        self.input.default.divisor = 1

    def command(self, positions, running_average_positions, total_steps, thermalization_steps, divisor, cell, pbc):
        total_steps += 1
        if total_steps > thermalization_steps:
            divisor += 1  # On the first step, divide by 2 to average two positions
            weight = 1. / divisor  # How much of the current step to mix into the average
            displacement = find_mic(positions - running_average_positions, cell, pbc)[0]
            new_running_average = running_average_positions + (weight * displacement)
            return {
                'running_average_positions': new_running_average,
                'total_steps': total_steps,
                'divisor': divisor,
            }
        else:
            return {
                'running_average_positions': running_average_positions,
                'total_steps': total_steps,
                'divisor': divisor,
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
        self.input.default.relax_endpoints = False

    def command(self, mixing_fraction, centroids_pos_list, running_average_positions, cell, pbc, relax_endpoints):

        centroids_pos_list = np.array(centroids_pos_list)
        running_average_positions = np.array(running_average_positions)

        updated_centroids = []

        # for i, (cent, avg) in enumerate(zip(centroids_pos_list, running_average_positions)):
        #     if i == 0 or i == (len(centroids_pos_list) - 1):
        #         updated_centroids.append(cent)
        #     else:
        #         displacement = find_mic(avg - cent, cell, pbc)[0]
        #         update = mixing_fraction * displacement
        #         updated_centroids.append(cent + update)

        for i, (cent, avg) in enumerate(zip(centroids_pos_list, running_average_positions)):
            displacement = find_mic(avg - cent, cell, pbc)[0]
            update = mixing_fraction * displacement
            updated_centroids.append(cent + update)

        return {
            'centroids_pos_list': np.array(updated_centroids)
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
    def command(self, kappa, dtau, centroids_pos_list, cell, pbc, smooth_style='global'):
        n_images = len(centroids_pos_list)
        smoothing_strength = kappa * n_images * dtau
        if smooth_style == 'global':
            smoothing_matrix = self._get_smoothing_matrix(n_images, smoothing_strength)
            smoothed_centroid_positions = np.tensordot(smoothing_matrix, np.array(centroids_pos_list), axes=1)
        elif smooth_style == 'local':
            smoothed_centroid_positions = self.locally_smoothed(smoothing_strength, centroids_pos_list, cell, pbc)
        else:
            raise TypeError('Smoothing: choose style = "global" or "local"')
        return {
            'centroids_pos_list': smoothed_centroid_positions
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

    @staticmethod
    def locally_smoothed(smoothing_strength, centroids_pos_list, cell, pbc):
        """
        A function that applies local smoothing by taking into account immediate neighbors.

        Attributes:
            n_images (int): number of images
            smoothing_strength (float): the smoothing penalty

        Returns:
            smoothing_matrix
        """
        smoothed_centroid_positions = [centroids_pos_list[0]]
        for i, cent in enumerate(centroids_pos_list[1:-1]):
            left = centroids_pos_list[i]
            right = centroids_pos_list[i+2]
            disp_left = find_mic(cent - left, cell, pbc)[0]
            disp_right = find_mic(right - cent, cell, pbc)[0]
            switch = (1 + np.cos(np.pi * np.tensordot(disp_left, disp_right) / (
                        np.linalg.norm(disp_left) * (np.linalg.norm(disp_right))))) / 2
            r_star = smoothing_strength * switch * (disp_right - disp_left)
            smoothed_centroid_positions.append(cent + r_star)
        smoothed_centroid_positions.append(centroids_pos_list[-1])

        return smoothed_centroid_positions


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


class ReflectAndMilestone(PrimitiveVertex):
    """
    Checks whether positions are closest to its own centroid. If not, reverses the velocities, and resets the
        positions to the values from the previous step. For the milestoning part, logs total reflections,
        reflections off edges and time spent at each edge.

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
        super(ReflectAndMilestone, self).__init__(name=name)
        # initialize matrices to be tracked
        self.input.default.image_number = None
        self.input.default.tracker_list = None
        self.input.default.reflections_matrix = None
        self.input.default.edge_reflections_matrix = None
        self.input.default.edge_time_matrix = None

    def command(self, positions, velocities, previous_positions, previous_velocities, centroid_positions,
                all_centroid_positions, thermalization_steps, image_number, tracker_list, reflections_matrix,
                edge_reflections_matrix, edge_time_matrix, cell, pbc):
        n_images = len(all_centroid_positions)
        n_edges = int(n_images * (n_images - 1) / 2)
        current_step = self.archive.clock

        if self.archive.clock == 1:  # clock always starts from 1
            # form the matrices
            image_number = self.get_closest_centroid_index(centroid_positions, all_centroid_positions, cell, pbc)
            tracker_list = [[None, None] for _ in np.arange(n_images)]
            reflections_matrix = np.zeros((n_images, n_images))
            edge_reflections_matrix = np.array([np.zeros((n_edges, n_edges)) for _ in np.arange(n_images)])
            edge_time_matrix = np.array([np.zeros(n_edges) for _ in np.arange(n_images)])

        tracker = tracker_list[image_number]

        # Find distance between positions and each of the centroids
        closest_centroid_id = self.get_closest_centroid_index(positions, all_centroid_positions, cell, pbc)

        if closest_centroid_id == image_number:
            # Return current positions and velocities
            reflected_positions = positions
            reflected_velocities = velocities

            if current_step >= thermalization_steps:
                # Start reflection tracking
                if tracker[1] is not None:  # If no reflections, increment time
                    edge_time_matrix[image_number][tracker[1]] += 1
                # End reflection tracking
        else:
            # Update to previous positions and velocities, if positions are not closest to parent centroid
            reflected_positions = previous_positions
            reflected_velocities = -previous_velocities

            if current_step >= thermalization_steps:
                # Start reflection tracking
                reflections_matrix[image_number, closest_centroid_id] += 1  # Save the reflection
                indices = np.zeros((n_images, n_images))  # images x images
                indices[image_number, closest_centroid_id] = 1  # Record the edge
                ind = np.tril(indices) + np.triu(indices).T  # Convert to triangular matrix
                # Record the index of the edge (N_j)
                n_j = int(np.nonzero(ind[np.tril_indices(n_images, k=-1)])[0][0])

                if tracker[1] is None:
                    tracker[1] = n_j  # Initialize N_j
                elif tracker[1] == n_j:
                    edge_time_matrix[image_number][n_j] += 1
                else:
                    tracker[0] = tracker[1]  # If reflecting off a different edge, change N_j to N_i
                    tracker[1] = n_j  # Set new N_j
                    edge_reflections_matrix[image_number][tracker[0], tracker[1]] += 1
                    # End reflection tracking

        return {
            'positions': reflected_positions,
            'velocities': reflected_velocities,
            'image_number': image_number,
            'tracker_list': tracker_list,
            'reflections_matrix': reflections_matrix,
            'edge_reflections_matrix': edge_reflections_matrix,
            'edge_time_matrix': edge_time_matrix,
        }

    @staticmethod
    def get_closest_centroid_index(positions, all_centroid_positions, cell, pbc):
        distances = [np.linalg.norm(find_mic(c_pos - positions, cell, pbc)[0]) for c_pos in all_centroid_positions]
        return np.argmin(distances)


class MilestonePostProcess(PrimitiveVertex):
    """
    Generates jump frequencies of each centroid to the final centroid form the reflections matrix, edge reflections
    matrix, and the edge time matrix stored by the milestoning vertex.

    Returns:
        (float): The mean time of first passage from the 0th image to the final image.
        (list): `n_images - 1` mean times of passage from the 0th Voronoi cell to the final cell.
        (list): Respective equilibrium probabilities to find the system in each Voronoi cell.

    TODO: Convert the final frequency to THz?
    """

    def command(self, time_step, reflections_matrix, edge_reflections_matrix, edge_time_matrix, n_images):
        n_edges = int(n_images * (n_images - 1) / 2)
        reflections_matrix = np.sum(reflections_matrix, axis=0)
        edge_reflections_matrix = np.sum(edge_reflections_matrix, axis=0)
        edge_time_matrix = np.sum(edge_time_matrix, axis=0)
        pis = self._get_pi(reflections_matrix, n_images)

        # for terminology, refer the following paper: https://doi.org/10.1063/1.3129843
        n_ij = np.zeros((n_edges, n_edges))
        r_i = np.zeros(n_edges)
        for img in np.arange(n_images):
            n = pis[img] * edge_reflections_matrix[img]
            r = pis[img] * edge_time_matrix[img]
            n_ij += n
            r_i += r

        # The paper uses shows ways to calculate the mean free passage time. The following section is
        # commented out, as it is redundant. May come in handy for a consistency check?

        # q_ij = []
        # for i in np.arange(n_edges):
        #     if r_i[i] != 0:
        #         q_ij += [n_ij[i] / r_i[i]]
        #     else:
        #         q_ij += [np.zeros(n_edges)]
        #
        # q_ij = np.array(q_ij)

        p_ij = []
        tau_i = []
        n_ji = n_ij.T
        with np.errstate(invalid='ignore'):  # just ignores the divide by zero error
            for i in np.arange(n_edges):
                p_ij += [n_ji[i] / np.sum(n_ji[i])]
                tau_i += [r_i[i] / np.sum(n_ji[i])]
        for i in np.arange(n_edges):
            for j in np.arange(n_edges):
                if np.isnan(p_ij[i][j]):
                    p_ij[i][j] = 0
                    tau_i[i] = 0
        p_ij = np.array(p_ij)
        p_new = np.delete(p_ij, n_edges - 1, 0)
        p_new = np.delete(p_new, n_edges - 1, 1)
        tau_new = np.delete(tau_i, n_edges - 1, 0)

        t_n = np.linalg.lstsq(np.eye(n_edges - 1) - p_new, tau_new, rcond=None)[0] * time_step * FS

        summation = 0
        mean_free_passage_times = [t_n[summation]]
        for i in np.arange(2, len(t_n)):
            summation += i
            if summation < len(t_n):
                mean_free_passage_times += [t_n[summation]]
        mean_free_passage_times = np.array(mean_free_passage_times)
        jump_frequencies = 1. / mean_free_passage_times

        return {
            'jump_frequencies': jump_frequencies,
            'mean_free_passage_times': mean_free_passage_times,
            'equilibrium_probability': pis,
            'reflections_matrix': reflections_matrix,
            'edge_reflections_matrix': edge_reflections_matrix,
            'edge_time_matrix': edge_time_matrix,
        }

    @staticmethod
    def _get_pi(reflections_matrix, n_images):
        dia = np.eye(n_images) * np.sum(reflections_matrix, axis=1)
        pi_mat_a = np.append(reflections_matrix.T - dia, [np.ones(n_images)], axis=0)
        pi_vec_b = np.append(np.zeros(n_images), [1])

        return np.linalg.lstsq(pi_mat_a, pi_vec_b, rcond=None)[0]
