# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

from pyiron_contrib.protocol.generic import CompoundVertex, Protocol
from pyiron_contrib.protocol.primitive.one_state import Counter, ExternalHamiltonian, GradientDescent, \
    NEBForces, InitialPositions
from pyiron_contrib.protocol.primitive.two_state import IsGEq
from pyiron_contrib.protocol.list import SerialList, ParallelList, AutoList
from pyiron_contrib.protocol.utils import Pointer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

"""
Protocol for nudged elastic band minimization.
"""

__author__ = "Liam Huber, Raynol Dsouza, Jan Janssen"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "18 July, 2019"


class NEB(CompoundVertex):
    """
    Relaxes a system according to the nudged elastic band method (Jonsson et al).

    Input dictionary:
        ref_job_full_path (str): Path to the pyiron job to use for evaluating forces and energies.
        structure_initial (Atoms): The starting structure for the elastic band.
        structure_final (Atoms): The final structure for the elastic band. Warning: must have the same numper and
            species of atoms, cell, and periodic boundary conditions as the initial structure.
        n_images (int): How many images create by interpolating between the two structures.
        n_steps (int): How many steps to run for. (Default is 100.)
        f_tol (float): Ionic force convergence (largest atomic force). (Default is 1e-4 eV/angstrom.)
        spring_constant (float): Spring force between atoms in adjacent images. (Default is 1.0 eV/angstrom^2.)
        tangent_style ('plain'/'improved'/'upwinding'): How to calculate the image tangent in 3N-dimensional space.
            (Default is 'upwinding'.)
        use_climbing_image (bool): Whether to replace the force with one that climbs along the tangent direction for
            the job with the highest energy. (Default is True.)
        smoothing (float): Strength of the smoothing spring when consecutive images form an angle. (Default is None, do
            not apply such a force.)
        gamma0 (float): Initial step size as a multiple of the force. (Default is 0.1.)
        fix_com (bool): Whether the center of mass motion should be subtracted off of the position update. (Default is
            True)
        use_adagrad (bool): Whether to have the step size decay according to adagrad. (Default is False)

    Output dictionary:
        energy_pot (list[float]): Total potential energy of the system in eV.
        positions_list (list[numpy.ndarray]): Atomic positions in angstroms for each image.
        forces_list (list[numpy.ndarray]): Atomic forces in eV/angstrom for each image, including spring forces.
    """

    DefaultWhitelist = {
        'calc_static': {
            'output': {
                'energy_pot': 1,
            },
        },
    }

    def __init__(self, **kwargs):
        super(NEB, self).__init__(**kwargs)

        # Protocol defaults
        id_ = self.input.default
        id_.n_steps = 100
        id_.f_tol = 1e-4

        id_.spring_constant = 1.
        id_.tangent_style = 'upwinding'
        id_.use_climbing_image = True
        id_.smoothing = None

        id_.gamma0 = 0.1
        id_.fix_com = True
        id_.use_adagrad = False

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.interpolate_images = InitialPositions()
        g.check_steps = IsGEq()
        g.calc_static = AutoList(ExternalHamiltonian)
        g.neb_forces = NEBForces()
        g.gradient_descent = SerialList(GradientDescent)
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.interpolate_images,
            g.check_steps, 'false',
            g.calc_static,
            g.neb_forces,
            g.gradient_descent,
            g.clock,
            g.check_steps
        )
        g.starting_vertex = self.graph.interpolate_images
        g.restarting_vertex = self.graph.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        g.interpolate_images.input.structure_initial = ip.structure_initial
        g.interpolate_images.input.structure_final = ip.structure_final
        g.interpolate_images.input.n_images = ip.n_images

        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        g.calc_static.input.n_children = ip.n_images
        g.calc_static.direct.ref_job_full_path = ip.ref_job_full_path
        g.calc_static.direct.structure = ip.structure_initial
        g.calc_static.broadcast.default.positions = gp.interpolate_images.output.initial_positions[-1]
        g.calc_static.broadcast.positions = gp.gradient_descent.output.positions[-1]

        g.neb_forces.input.default.positions_list = gp.interpolate_images.output.initial_positions[-1]
        g.neb_forces.input.positions_list = gp.gradient_descent.output.positions[-1]
        g.neb_forces.input.energies = gp.calc_static.output.energy_pot[-1]
        g.neb_forces.input.forces_list = gp.calc_static.output.forces[-1]
        g.neb_forces.input.cell = ip.structure_initial.cell
        g.neb_forces.input.pbc = ip.structure_initial.pbc
        g.neb_forces.input.spring_constant = ip.spring_constant
        g.neb_forces.input.tangent_style = ip.tangent_style
        g.neb_forces.input.use_climbing_image = ip.use_climbing_image
        g.neb_forces.input.smoothing = ip.smoothing

        g.gradient_descent.input.n_children = ip.n_images
        g.gradient_descent.broadcast.default.positions = gp.interpolate_images.output.initial_positions[-1]
        g.gradient_descent.broadcast.positions = gp.gradient_descent.output.positions[-1]
        g.gradient_descent.broadcast.forces = gp.neb_forces.output.forces_list[-1]
        g.gradient_descent.direct.masses = ip.structure_initial.get_masses
        g.gradient_descent.direct.gamma0 = ip.gamma0
        g.gradient_descent.direct.fix_com = ip.fix_com
        g.gradient_descent.direct.use_adagrad = ip.use_adagrad

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'energy_pot': ~gp.calc_static.output.energy_pot[-1],
            'positions_list': ~gp.gradient_descent.output.positions[-1],
            'forces_list': ~gp.neb_forces.output.forces_list[-1]
        }

    def _get_energies(self, frame=None):
        if frame is None:
            return self.graph.calc_static.output.energy_pot[-1]
        else:
            return self.graph.calc_static.archive.output.energy_pot.data[frame]

    def plot_elastic_band(self, ax=None, frame=None, plot_kwargs=None):
        if ax is None:
            _, ax = plt.subplots()
        if plot_kwargs is None:
            plot_kwargs = {}
        if 'marker' not in plot_kwargs.keys():
            plot_kwargs = {'marker': 'o'}
        energies = np.array(self._get_energies(frame=frame))
        ax.plot(energies - energies[0], **plot_kwargs)
        ax.set_ylabel("Energy")
        ax.set_xlabel("Image")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return ax

    def get_forward_barrier(self, frame=None):
        """
        Get the energy barrier from the 0th image to the highest energy (saddle state).
        """
        energies = self._get_energies(frame=frame)
        return np.amax(energies) - energies[0]

    def get_reverse_barrier(self, frame=None):
        """
        Get the energy barrier from the final image to the highest energy (saddle state).
        """
        energies = self._get_energies(frame=frame)
        return np.amax(energies) - energies[-1]

    def get_barrier(self, frame=None):
        return self.get_forward_barrier(frame=frame)


class NEBSerial(NEB):
    """
    The same as the NEB protocol, but ensures that the force/energy calls are treated in series.
    """

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.interpolate_images = InterpolatePositions()
        g.check_steps = IsGEq()
        g.calc_static = SerialList(ExternalHamiltonian)  # Enforce serial
        g.neb_forces = NEBForces()
        g.gradient_descent = SerialList(GradientDescent)
        g.clock = Counter()


class NEBParallel(NEB):
    """
    The same as the NEB protocol, but ensures that the force/energy calls are treated in parallel.
    """

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.interpolate_images = InterpolatePositions()
        g.check_steps = IsGEq()
        g.calc_static = ParallelList(ExternalHamiltonian)  # Enforce parallel
        g.neb_forces = NEBForces()
        g.gradient_descent = SerialList(GradientDescent)
        g.clock = Counter()


class ProtocolNEB(Protocol, NEB):
    pass
