# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function


from pyiron_contrib.protocol.generic import Protocol
from pyiron_contrib.protocol.list import SerialList, AutoList, ParallelList
from pyiron_contrib.protocol.primitive.fts_vertices import StringRecenter, StringReflect, PositionsRunningAverage, \
    CentroidsRunningAverageMix, CentroidsSmoothing, CentroidsReparameterization, MilestoningVertex
from pyiron_contrib.protocol.primitive.one_state import InterpolatePositions, RandomVelocity, \
    ExternalHamiltonian, VerletPositionUpdate, VerletVelocityUpdate, \
    Counter, Zeros, WelfordOnline, SphereReflection
from pyiron_contrib.protocol.primitive.two_state import IsGEq, ModIsZero
from pyiron_contrib.protocol.utils import Pointer
import numpy as np
import matplotlib.pyplot as plt
from ase.geometry import find_mic
from scipy.constants import femto as FS

"""
Protocols and vertices for the Finite Temperature String method.
"""

__author__ = "Raynol Dsouza, Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "23 July, 2019"


class StringRelaxation(Protocol):
    """
    Runs Finite Temperature String relaxation

    Input attributes:
        ref_job_full_path (string): The path to the saved reference job to use for calculating forces and energies.
        structure_initial (Atoms): The initial structure.
        structure_initial (Atoms): The final structure.
        n_images (int): Number of centroids/images.
        n_steps (int): How many steps to run for.
        thermalization_steps (int): How many steps to run for before beginning to update the string.
        sampling_period (int): How often to sample the images and update the centroids. (Default is to synchronize with
            the archiving period; it is highly recommended to set this value to divide the archiving period evenly, so
            that vertices in this branch of the graph get their output archived as normal.)
        temperature (float): Temperature to run at in K.
        reflection_cutoff_distance (float): How far (angstroms) to allow each individual atom to stray from its
            centroid value before reflecting it.
        relax_endpoints (bool): Whether or not to allow string endpoint centroids to evolve. (Default is False.)
        mixing_fraction (float): How much of the images' running average of position to mix into the centroid positions
            each time the mixer is called. (Default is 0.1.)
        nominal_smoothing (float): How much smoothing to apply to the updating centroid positions (endpoints are
            not effected). The actual smoothing is the product of this nominal value, the number of images, and the
            mixing fraction, ala Vanden-Eijnden and Venturoli (2009). (Default is 0.1.)
        temperature_damping_timescale (float): Langevin thermostat timescale in fs. (Default is 100 fs.)
        overheat_fraction (float): The fraction by which to overheat the initial velocities. This can be useful for
            more quickly equilibrating a system whose initial structure is its fully relaxed positions -- in which
            case equipartition of energy tells us that the kinetic energy should be initialized to double the
            desired value. (Default is 2.0, rely on equipartition.)
        time_step (float): MD timestep in fs. (Default is 1 fs.)

    TODO: Wire it so sphere reflection is optional
    """

    def __init__(self, project=None, name=None, job_name=None):
        super(StringRelaxation, self).__init__(project=project, name=name, job_name=job_name)

        # Protocol defaults
        id_ = self.input.default

        id_.relax_endpoints = False
        id_.mixing_fraction = 0.1
        id_.nominal_smoothing = 0.1
        id_.temperature_damping_timescale = 100.
        id_.overheat_fraction = 2.
        id_.time_step = 1.

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.interpolate_positions = InterpolatePositions()
        g.initial_velocities = SerialList(RandomVelocity)
        g.initial_forces = Zeros()
        g.check_steps = IsGEq()
        g.verlet_positions = SerialList(VerletPositionUpdate)
        g.calc_static_images = AutoList(ExternalHamiltonian)
        g.verlet_velocities = SerialList(VerletVelocityUpdate)
        g.reflect_string = SerialList(StringReflect)
        g.reflect_atoms = SerialList(SphereReflection)
        g.check_thermalized = IsGEq()
        g.check_sampling_period = ModIsZero()
        g.running_average = PositionsRunningAverage()
        g.mix = CentroidsRunningAverageMix()
        g.smooth = CentroidsSmoothing()
        g.reparameterize = CentroidsReparameterization()
        g.calc_static_centroids = AutoList(ExternalHamiltonian)
        g.recenter = SerialList(StringRecenter)
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.interpolate_positions,
            g.initial_velocities,
            g.initial_forces,
            g.check_steps, 'false',
            g.verlet_positions,
            g.reflect_string,  # Comes before atomic reflecting so we can actually trigger a full string reflection!
            g.reflect_atoms,  # Comes after, since even if the string doesn't reflect, on atom might have migrated
            g.calc_static_images,
            g.verlet_velocities,
            g.check_thermalized, 'true',
            g.check_sampling_period, 'true',
            g.running_average,
            g.mix,
            g.smooth,
            g.reparameterize,
            g.calc_static_centroids,
            g.recenter,
            g.clock,
            g.check_steps
        )
        g.make_edge(g.check_thermalized, g.recenter, 'false')
        g.make_edge(g.check_sampling_period, g.recenter, 'false')
        g.starting_vertex = g.interpolate_positions
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # interpolate_positions
        g.interpolate_positions.input.structure_initial = ip.structure_initial
        g.interpolate_positions.input.structure_final = ip.structure_final
        g.interpolate_positions.input.n_images = ip.n_images

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_images
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure_initial.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # initial_forces
        g.initial_forces.input.shape = ip.structure_initial.positions.shape

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # verlet_positions
        g.verlet_positions.input.n_children = ip.n_images
        g.verlet_positions.direct.time_step = ip.time_step
        g.verlet_positions.direct.masses = ip.structure_initial.get_masses
        g.verlet_positions.direct.temperature = ip.temperature
        g.verlet_positions.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_positions.broadcast.default.positions = gp.interpolate_positions.output.interpolated_positions[-1]
        g.verlet_positions.broadcast.default.velocities = gp.initial_velocities.output.velocities[-1]
        g.verlet_positions.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.verlet_positions.broadcast.positions = gp.recenter.output.positions[-1]
        g.verlet_positions.broadcast.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.broadcast.forces = gp.recenter.output.forces[-1]

        # reflect string
        g.reflect_string.input.n_children = ip.n_images
        g.reflect_string.direct.cell = ip.structure_initial.cell
        g.reflect_string.direct.pbc = ip.structure_initial.pbc

        g.reflect_string.direct.default.all_centroid_positions = \
            gp.interpolate_positions.output.interpolated_positions[-1]
        g.reflect_string.broadcast.default.centroid_positions = \
            gp.interpolate_positions.output.interpolated_positions[-1]
        g.reflect_string.broadcast.default.previous_positions = \
            gp.interpolate_positions.output.interpolated_positions[-1]
        g.reflect_string.broadcast.default.previous_velocities = gp.initial_velocities.output.velocities[-1]

        g.reflect_string.direct.all_centroid_positions = gp.reparameterize.output.centroids_pos_list[-1]
        g.reflect_string.broadcast.centroid_positions = gp.reparameterize.output.centroids_pos_list[-1]
        g.reflect_string.broadcast.previous_positions = gp.recenter.output.positions[-1]
        g.reflect_string.broadcast.previous_velocities = gp.verlet_velocities.output.velocities[-1]

        g.reflect_string.broadcast.positions = gp.verlet_positions.output.positions[-1]
        g.reflect_string.broadcast.velocities = gp.verlet_positions.output.velocities[-1]

        # reflect individual atoms which stray too far
        g.reflect_atoms.input.n_children = ip.n_images
        g.reflect_atoms.broadcast.default.reference_positions = \
            gp.interpolate_positions.output.interpolated_positions[-1]
        g.reflect_atoms.broadcast.default.previous_positions = \
            gp.interpolate_positions.output.interpolated_positions[-1]
        g.reflect_atoms.broadcast.default.previous_velocities = gp.initial_velocities.output.velocities[-1]

        g.reflect_atoms.broadcast.reference_positions = gp.reparameterize.output.centroids_pos_list[-1]
        g.reflect_atoms.broadcast.positions = gp.reflect_string.output.positions[-1]
        g.reflect_atoms.broadcast.velocities = gp.reflect_string.output.velocities[-1]
        g.reflect_atoms.broadcast.previous_positions = gp.recenter.output.positions[-1]
        g.reflect_atoms.broadcast.previous_velocities = gp.verlet_velocities.output.velocities[-1]
        g.reflect_atoms.direct.cutoff_distance = ip.reflection_cutoff_distance
        g.reflect_atoms.direct.cell = ip.structure_initial.cell
        g.reflect_atoms.direct.pbc = ip.structure_initial.pbc

        # calc_static_images
        g.calc_static_images.input.n_children = ip.n_images
        g.calc_static_images.direct.ref_job_full_path = ip.ref_job_full_path
        g.calc_static_images.direct.structure = ip.structure_initial
        g.calc_static_images.broadcast.positions = gp.reflect_atoms.output.positions[-1]

        # verlet_velocities
        g.verlet_velocities.input.n_children = ip.n_images
        g.verlet_velocities.direct.time_step = ip.time_step
        g.verlet_velocities.direct.masses = ip.structure_initial.get_masses
        g.verlet_velocities.direct.temperature = ip.temperature
        g.verlet_velocities.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_velocities.broadcast.velocities = gp.reflect_atoms.output.velocities[-1]
        g.verlet_velocities.broadcast.forces = gp.calc_static_images.output.forces[-1]

        # check_thermalized
        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # check_sampling_period
        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = Pointer(self.archive.period)
        g.check_sampling_period.input.mod = ip.sampling_period

        # running_average
        g.running_average.input.default.running_average_list = \
            gp.interpolate_positions.output.interpolated_positions[-1]
        g.running_average.input.running_average_list = gp.running_average.output.running_average_list[-1]
        g.running_average.input.positions_list = gp.reflect_atoms.output.positions[-1]
        g.running_average.input.relax_endpoints = ip.relax_endpoints
        g.running_average.input.cell = ip.structure_initial.cell
        g.running_average.input.pbc = ip.structure_initial.pbc

        # mix
        g.mix.input.default.centroids_pos_list = gp.interpolate_positions.output.interpolated_positions[-1]
        g.mix.input.centroids_pos_list = gp.reparameterize.output.centroids_pos_list[-1]
        g.mix.input.mixing_fraction = ip.mixing_fraction
        g.mix.input.running_average_list = gp.running_average.output.running_average_list[-1]
        g.mix.input.cell = ip.structure_initial.cell
        g.mix.input.pbc = ip.structure_initial.pbc

        # smooth
        g.smooth.input.kappa = ip.nominal_smoothing
        g.smooth.input.dtau = ip.mixing_fraction
        g.smooth.input.all_centroid_positions = gp.mix.output.centroids_pos_list[-1]

        # reparameterize
        g.reparameterize.input.centroids_pos_list = gp.smooth.output.all_centroid_positions[-1]
        g.reparameterize.input.cell = ip.structure_initial.cell
        g.reparameterize.input.pbc = ip.structure_initial.pbc

        # calc_static_centroids
        g.calc_static_centroids.input.n_children = ip.n_images
        g.calc_static_centroids.direct.ref_job_full_path = ip.ref_job_full_path
        g.calc_static_centroids.direct.structure = ip.structure_initial
        g.calc_static_centroids.broadcast.positions = gp.reparameterize.output.centroids_pos_list[-1]

        # recenter
        g.recenter.input.n_children = ip.n_images
        g.recenter.direct.cell = ip.structure_initial.cell
        g.recenter.direct.pbc = ip.structure_initial.pbc

        g.recenter.direct.default.all_centroid_positions = gp.interpolate_positions.output.interpolated_positions[-1]
        g.recenter.broadcast.default.centroid_positions = gp.interpolate_positions.output.interpolated_positions[-1]
        g.recenter.direct.default.centroid_forces = gp.initial_forces.output.zeros[-1]

        g.recenter.direct.all_centroid_positions = gp.reparameterize.output.centroids_pos_list[-1]
        g.recenter.broadcast.centroid_positions = gp.reparameterize.output.centroids_pos_list[-1]
        g.recenter.broadcast.centroid_forces = gp.calc_static_centroids.output.forces[-1]
        g.recenter.broadcast.positions = gp.reflect_atoms.output.positions[-1]
        g.recenter.broadcast.forces = gp.calc_static_images.output.forces[-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'energy_pot': ~gp.calc_static_centroids.output.energy_pot[-1],
            'positions': ~gp.reparameterize.output.centroids_pos_list[-1],
            'forces': ~gp.calc_static_centroids.output.forces[-1]
        }

    def plot_potential_energy(self, frame=-1):
        energies = self.graph.calc_static_centroids.archive.output.energy_pot[frame]
        energies -= np.amin(energies)
        plt.plot(energies, marker='o')
        plt.xlabel('Image')
        plt.ylabel('Energy [eV]')
        return energies


class VirtualWork(Protocol):
    """
    Calculates the virtual work to follow a transition path by integrating over the average force of MD images
    constrained to stay within the Voronoi domain of their Voronoi centroid along the path.

    Input attributes:
        structure (Atoms): The initial or final structure whose number and species of atoms, cell, and pbc are those of
            the Voronoi centroids.
        all_centroid_positions (list/numpy.ndarray): The centroid positions along the discretized transition path.
        n_images (int): How many images the path is divided into. TODO: Extract this from `all_centroid_positions`.
        n_steps (int): How many steps to run for.
        thermalization_steps (int): How many steps to run for before beginning to track the average force.
        sampling_period (int): How often to sample the images and update the centroids. (Default is to synchronize with
            the archiving period; it is highly recommended to set this value to divide the archiving period evenly, so
            that vertices in this branch of the graph get their output archived as normal.)
        temperature (float): Temperature to run at in K.
        reflection_cutoff_distance (float): How far (angstroms) to allow each individual atom to stray from its
            centroid value before reflecting it.
        temperature_damping_timescale (float): Langevin thermostat timescale in fs. (Default is 100 fs.)
        overheat_fraction (float): The fraction by which to overheat the initial velocities. This can be useful for
            more quickly equilibrating a system whose initial structure is its fully relaxed positions -- in which
            case equipartition of energy tells us that the kinetic energy should be initialized to double the
            desired value. (Default is 2.0, rely on equipartition.)
        time_step (float): MD timestep in fs. (Default is 1 fs.)

    TODO: Wire it so sphere reflection is optional
    """

    def __init__(self, project=None, name=None, job_name=None):
        super(VirtualWork, self).__init__(project=project, name=name, job_name=job_name)

        # Protocol defaults
        id_ = self.input.default
        id_.time_step = 1.
        id_.overheat_fraction = 2.
        id_.damping_timescale = 100.

        self._displacements = None

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.initial_velocities = SerialList(RandomVelocity)
        g.initial_forces = Zeros()
        g.check_steps = IsGEq()
        g.verlet_positions = SerialList(VerletPositionUpdate)
        g.reflect_string = SerialList(StringReflect)
        g.reflect_atoms = SerialList(SphereReflection)
        g.calc_static = AutoList(ExternalHamiltonian)
        g.verlet_velocities = SerialList(VerletVelocityUpdate)
        g.check_thermalized = IsGEq()
        g.check_sampling_period = ModIsZero()
        g.average_forces = SerialList(WelfordOnline)
        g.average_positions = SerialList(WelfordOnline)
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.initial_velocities,
            g.initial_forces,
            g.check_steps, 'false',
            g.verlet_positions,
            g.reflect_string,
            g.reflect_atoms,
            g.calc_static,
            g.verlet_velocities,
            g.check_thermalized, 'true',
            g.check_sampling_period, 'true',
            g.average_forces,
            g.average_positions,
            g.clock,
            g.check_steps
        )
        g.make_edge(g.check_thermalized, g.clock, 'false')
        g.make_edge(g.check_sampling_period, g.clock, 'false')
        g.starting_vertex = g.initial_velocities
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_images
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # verlet_positions
        g.verlet_positions.input.n_children = ip.n_images
        g.verlet_positions.direct.time_step = ip.time_step
        g.verlet_positions.direct.masses = ip.structure.get_masses
        g.verlet_positions.direct.temperature = ip.temperature
        g.verlet_positions.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_positions.broadcast.default.positions = ip.all_centroid_positions
        g.verlet_positions.broadcast.default.velocities = gp.initial_velocities.output.velocities[-1]
        g.verlet_positions.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.verlet_positions.broadcast.positions = gp.reflect_atoms.output.positions[-1]
        g.verlet_positions.broadcast.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.broadcast.forces = gp.calc_static.output.forces[-1]

        # reflect entire string
        g.reflect_string.input.n_children = ip.n_images
        g.reflect_string.direct.cell = ip.structure.cell
        g.reflect_string.direct.pbc = ip.structure.pbc

        g.reflect_string.broadcast.default.previous_positions = ip.all_centroid_positions
        g.reflect_string.broadcast.default.previous_velocities = gp.initial_velocities.output.velocities[-1]

        g.reflect_string.broadcast.previous_positions = gp.reflect_atoms.output.positions[-1]
        g.reflect_string.broadcast.previous_velocities = gp.reflect_atoms.output.velocities[-1]

        g.reflect_string.broadcast.positions = gp.verlet_positions.output.positions[-1]
        g.reflect_string.broadcast.velocities = gp.verlet_positions.output.velocities[-1]

        g.reflect_string.direct.all_centroid_positions = ip.all_centroid_positions
        g.reflect_string.broadcast.centroid_positions = ip.all_centroid_positions

        # reflect individual atoms which stray too far
        g.reflect_atoms.input.n_children = ip.n_images
        g.reflect_atoms.direct.cutoff_distance = ip.reflection_cutoff_distance
        g.reflect_atoms.direct.cell = ip.structure.cell
        g.reflect_atoms.direct.pbc = ip.structure.pbc

        g.reflect_atoms.broadcast.default.previous_positions = ip.all_centroid_positions
        g.reflect_atoms.broadcast.default.previous_velocities = gp.initial_velocities.output.velocities[-1]

        g.reflect_atoms.broadcast.previous_positions = gp.reflect_atoms.output.positions[-1]
        g.reflect_atoms.broadcast.previous_velocities = gp.reflect_atoms.output.velocities[-1]

        g.reflect_atoms.broadcast.positions = gp.reflect_string.output.positions[-1]
        g.reflect_atoms.broadcast.velocities = gp.reflect_string.output.velocities[-1]

        g.reflect_atoms.broadcast.reference_positions = ip.all_centroid_positions

        # calc_static
        g.calc_static.input.n_children = ip.n_images
        g.calc_static.direct.ref_job_full_path = ip.ref_job_full_path
        g.calc_static.direct.structure = ip.structure

        g.calc_static.broadcast.positions = gp.reflect_atoms.output.positions[-1]

        # verlet_velocities
        g.verlet_velocities.input.n_children = ip.n_images
        g.verlet_velocities.direct.time_step = ip.time_step
        g.verlet_velocities.direct.masses = ip.structure.get_masses
        g.verlet_velocities.direct.temperature = ip.temperature
        g.verlet_velocities.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_velocities.broadcast.velocities = gp.reflect_atoms.output.velocities[-1]
        g.verlet_velocities.broadcast.forces = gp.calc_static.output.forces[-1]

        # check_thermalized
        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # check_sampling_period
        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = Pointer(self.archive.period)
        g.check_sampling_period.input.mod = ip.sampling_period

        # average_forces
        g.average_forces.input.n_children = ip.n_images
        g.average_forces.broadcast.sample = gp.calc_static.output.forces[-1]

        # average_positions
        g.average_positions.input.n_children = ip.n_images
        g.average_positions.broadcast.sample = gp.reflect_atoms.output.positions[-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'mean_forces': ~gp.average_forces.output.mean[-1],
            'standard_deviations': ~gp.average_forces.output.std[-1],
            'n_samples': ~gp.average_forces.output.n_samples[-1],
        }

    def get_displacements(self, use_average=False):
        if use_average:
            all_positions = np.array(self.graph.average_positions.output.mean[-1])
        else:
            all_positions = np.array(self.input.all_centroid_positions)
        cell = self.input.structure.cell
        pbc = self.input.structure.pbc
        return np.array([
            find_mic(front - back, cell, pbc)[0]
            for front, back
            in zip(all_positions[1:], all_positions[:-1])
        ])

    def get_work_steps(self, frame=-1, use_average=False):
        mean_forces = np.array(self.graph.average_forces.archive.output.mean[frame])
        midpoint_forces = 0.5 * (mean_forces[1:] + mean_forces[:-1])
        displacements = self.get_displacements(use_average=use_average)
        return np.array([np.tensordot(-f, d) for f, d in zip(midpoint_forces, displacements)])

    def get_virtual_work(self, frame=-1, use_average=False):
        work_steps = self.get_work_steps(frame=frame, use_average=use_average)
        mid_pt = int(np.ceil(len(work_steps) / 2.))
        # TODO: account for odd/even number of steps
        return 0.5 * (np.sum(work_steps[:mid_pt]) - np.sum(work_steps[mid_pt:]))


class VirtualWorkFullStep(VirtualWork):
    """
    Reflects after a full velocity Verlet step instead of reflecting at the half-step.
    The advantage is that reflected velocities are *exactly* the negative of the last step's velocities. The
    disadvantage is that the code is a bit uglier because we need to pass around extra force variables in the reflectors
    so that we don't need to repeat the force calculation after reflection...shit, you can't do per-atom reflections
    this way without fucking up the forces...if you reflect anyone you need to recalculate all the forces! (Well, ok,
    you only need to recalulate the reflected atom(s) and their neighbours, but I don't have that sort of optimization
    right now.)
    """

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.initial_velocities,
            g.initial_forces,
            g.check_steps, 'false',
            g.verlet_positions,
            g.calc_static,
            g.verlet_velocities,
            g.reflect_string,
            g.reflect_atoms,
            g.check_thermalized, 'true',
            g.check_sampling_period, 'true',
            g.average_forces,
            g.average_positions,
            g.clock,
            g.check_steps
        )
        g.make_edge(g.check_thermalized, g.clock, 'false')
        g.make_edge(g.check_sampling_period, g.clock, 'false')
        g.starting_vertex = g.initial_velocities
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_images
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # verlet_positions
        g.verlet_positions.input.n_children = ip.n_images
        g.verlet_positions.direct.time_step = ip.time_step
        g.verlet_positions.direct.masses = ip.structure.get_masses
        g.verlet_positions.direct.temperature = ip.temperature
        g.verlet_positions.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_positions.broadcast.default.positions = ip.all_centroid_positions
        g.verlet_positions.broadcast.default.velocities = gp.initial_velocities.output.velocities[-1]
        g.verlet_positions.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.verlet_positions.broadcast.positions = gp.reflect_atoms.output.positions[-1]
        g.verlet_positions.broadcast.velocities = gp.reflect_atoms.output.velocities[-1]
        g.verlet_positions.broadcast.forces = gp.calc_static.output.forces[-1]

        # calc_static
        g.calc_static.input.n_children = ip.n_images
        g.calc_static.direct.ref_job_full_path = ip.ref_job_full_path
        g.calc_static.direct.structure = ip.structure

        g.calc_static.broadcast.positions = gp.verlet_positions.output.positions[-1]

        # verlet_velocities
        g.verlet_velocities.input.n_children = ip.n_images
        g.verlet_velocities.direct.time_step = ip.time_step
        g.verlet_velocities.direct.masses = ip.structure.get_masses
        g.verlet_velocities.direct.temperature = ip.temperature
        g.verlet_velocities.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_velocities.broadcast.velocities = gp.verlet_positions.output.velocities[-1]
        g.verlet_velocities.broadcast.forces = gp.calc_static.output.forces[-1]

        # reflect entire string
        g.reflect_string.input.n_children = ip.n_images
        g.reflect_string.direct.cell = ip.structure.cell
        g.reflect_string.direct.pbc = ip.structure.pbc

        g.reflect_string.broadcast.default.previous_positions = ip.all_centroid_positions
        g.reflect_string.broadcast.default.previous_velocities = gp.initial_velocities.output.velocities[-1]
        g.reflect_string.direct.default.previous_forces = gp.initial_forces.output.zeros[-1]

        g.reflect_string.broadcast.previous_positions = gp.reflect_atoms.output.positions[-1]
        g.reflect_string.broadcast.previous_velocities = gp.reflect_atoms.output.velocities[-1]
        g.reflect_string.broadcast.previous_forces = gp.reflect_atoms.output.forces[-1]

        g.reflect_string.broadcast.positions = gp.verlet_positions.output.positions[-1]
        g.reflect_string.broadcast.velocities = gp.verlet_velocities.output.velocities[-1]
        g.reflect_string.broadcast.forces = gp.calc_static.output.forces[-1]

        g.reflect_string.direct.all_centroid_positions = ip.all_centroid_positions
        g.reflect_string.broadcast.centroid_positions = ip.all_centroid_positions

        # reflect individual atoms which stray too far
        g.reflect_atoms.input.n_children = ip.n_images
        g.reflect_atoms.direct.cutoff_distance = ip.reflection_cutoff_distance
        g.reflect_atoms.direct.cell = ip.structure.cell
        g.reflect_atoms.direct.pbc = ip.structure.pbc

        g.reflect_atoms.broadcast.default.previous_positions = ip.all_centroid_positions
        g.reflect_atoms.broadcast.default.previous_velocities = gp.initial_velocities.output.velocities[-1]
        g.reflect_atoms.direct.default.previous_forces = gp.initial_forces.output.zeros[-1]

        g.reflect_atoms.broadcast.previous_positions = gp.reflect_atoms.output.positions[-1]
        g.reflect_atoms.broadcast.previous_velocities = gp.reflect_atoms.output.velocities[-1]
        g.reflect_atoms.broadcast.previous_forces = gp.reflect_atoms.output.forces[-1]

        g.reflect_atoms.broadcast.positions = gp.reflect_string.output.positions[-1]
        g.reflect_atoms.broadcast.velocities = gp.reflect_string.output.velocities[-1]
        g.reflect_atoms.broadcast.forces = gp.reflect_string.output.velocities[-1]

        g.reflect_atoms.broadcast.reference_positions = ip.all_centroid_positions

        # check_thermalized
        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # check_sampling_period
        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = Pointer(self.archive.period)
        g.check_sampling_period.input.mod = ip.sampling_period

        # average_forces
        g.average_forces.input.n_children = ip.n_images
        g.average_forces.broadcast.sample = gp.calc_static.output.forces[-1]

        # average_positions
        g.average_positions.input.n_children = ip.n_images
        g.average_positions.broadcast.sample = gp.reflect_atoms.output.positions[-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])


class ConstrainedMD(Protocol):
    """For parallelizing virtual work. Work in progress, not yet functioning."""

    def __init__(self, project=None, name=None, job_name=None):
        super(ConstrainedMD, self).__init__(project=project, name=name, job_name=job_name)

        # Protocol defaults
        id_ = self.input.default
        id_.time_step = 1.
        id_.overheat_fraction = 2.
        id_.damping_timescale = 100.

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.initial_velocities = RandomVelocity()
        g.initial_forces = Zeros()
        g.check_steps = IsGEq()
        g.verlet_positions = VerletPositionUpdate()
        g.reflect_string = StringReflect()
        g.reflect_atoms = SphereReflection()
        g.calc_static = ExternalHamiltonian()
        g.verlet_velocities = VerletVelocityUpdate()
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.initial_velocities,
            g.initial_forces,
            g.check_steps, 'false',
            g.verlet_positions,
            g.reflect_string,
            g.reflect_atoms,
            g.calc_static,
            g.verlet_velocities,
            g.clock,
            g.check_steps
        )
        g.starting_vertex = g.initial_velocities
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # initial_velocities
        g.initial_velocities.input.temperature = ip.temperature
        g.initial_velocities.input.masses = ip.structure.get_masses
        g.initial_velocities.input.overheat_fraction = ip.overheat_fraction

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # verlet_positions
        g.verlet_positions.input.time_step = ip.time_step
        g.verlet_positions.input.masses = ip.structure.get_masses
        g.verlet_positions.input.temperature = ip.temperature
        g.verlet_positions.input.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_positions.input.default.positions = ip.centroid_positions
        g.verlet_positions.input.default.velocities = gp.initial_velocities.output.velocities[-1]
        g.verlet_positions.input.default.forces = gp.initial_forces.output.zeros[-1]

        g.verlet_positions.input.positions = gp.reflect_atoms.output.positions[-1]
        g.verlet_positions.input.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.input.forces = gp.calc_static.output.forces[-1]

        # reflect
        g.reflect_string.input.cell = ip.structure.cell
        g.reflect_string.input.pbc = ip.structure.pbc

        g.reflect_string.input.default.previous_positions = ip.centroid_positions
        g.reflect_string.input.default.previous_velocities = gp.initial_velocities.output.velocities[-1]

        g.reflect_string.input.previous_positions = gp.reflect_atoms.output.positions[-1]
        g.reflect_string.input.previous_velocities = gp.reflect_atoms.output.velocities[-1]

        g.reflect_string.input.positions = gp.verlet_positions.output.positions[-1]
        g.reflect_string.input.velocities = gp.verlet_positions.output.velocities[-1]

        g.reflect_string.input.all_centroid_positions = ip.all_centroid_positions
        g.reflect_string.input.centroid_positions = ip.centroid_positions

        # reflect individual atoms which stray too far
        g.reflect_atoms.input.cutoff_distance = ip.reflection_cutoff_distance
        g.reflect_atoms.input.cell = ip.structure.cell
        g.reflect_atoms.input.pbc = ip.structure.pbc

        g.reflect_atoms.input.reference_positions = ip.centroid_positions

        g.reflect_atoms.input.default.previous_positions = ip.centroid_positions
        g.reflect_atoms.input.default.previous_velocities = gp.initial_velocities.output.velocities[-1]

        g.reflect_atoms.input.previous_positions = gp.reflect_atoms.output.positions[-1]
        g.reflect_atoms.input.previous_velocities = gp.reflect_atoms.output.velocities[-1]

        g.reflect_atoms.input.positions = gp.reflect_string.output.positions[-1]
        g.reflect_atoms.input.velocities = gp.reflect_string.output.velocities[-1]

        # calc_static
        g.calc_static.input.ref_job_full_path = ip.ref_job_full_path
        g.calc_static.input.structure = ip.structure

        g.calc_static.input.positions = gp.reflect_atoms.output.positions[-1]

        # verlet_velocities
        g.verlet_velocities.input.time_step = ip.time_step
        g.verlet_velocities.input.masses = ip.structure.get_masses
        g.verlet_velocities.input.temperature = ip.temperature
        g.verlet_velocities.input.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_velocities.input.velocities = gp.reflect_atoms.output.velocities[-1]
        g.verlet_velocities.input.forces = gp.calc_static.output.forces[-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'positions': ~gp.reflect_string.output.positions[-1],
            'velocities': ~gp.verlet_velocities.output.velocities[-1],
            'forces': ~gp.calc_static.output.forces[-1]
        }

    def parallel_setup(self):
        super(ConstrainedMD, self).parallel_setup()
        self.graph.calc_static.parallel_setup()


class VirtualWorkParallel(VirtualWork):
    """Work in progress, not yet functioning."""

    def __init__(self, project=None, name=None, job_name=None):
        super(VirtualWorkParallel, self).__init__(project=project, name=name, job_name=job_name)

        # Protocol defaults
        id_ = self.input.default
        id_.time_step = 1.
        id_.overheat_fraction = 2.
        id_.damping_timescale = 100.

        self._displacements = None

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.check_steps = IsGEq()
        g.constrained_evolution = ParallelList(ConstrainedMD)
        g.check_thermalized = IsGEq()
        g.average_forces = SerialList(WelfordOnline)
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.check_steps, 'false',
            g.constrained_evolution,
            g.check_thermalized, 'true',
            g.average_forces,
            g.clock,
            g.check_steps
        )
        g.make_edge(g.check_thermalized, g.clock, 'false')
        g.starting_vertex = g.check_steps
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # constrained_evolution
        g.constrained_evolution.input.n_children = ip.n_images
        g.constrained_evolution.direct.structure = ip.structure
        g.constrained_evolution.direct.temperature = ip.temperature
        g.constrained_evolution.direct.temperature_damping_timescale = ip.temperature_damping_timescale
        g.constrained_evolution.direct.all_centroid_positions = ip.all_centroid_positions
        g.constrained_evolution.broadcast.centroid_positions = ip.all_centroid_positions
        g.constrained_evolution.direct.reflection_cutoff_distance = ip.reflection_cutoff_distance
        g.constrained_evolution.direct.ref_job_full_path = ip.ref_job_full_path
        g.constrained_evolution.direct.n_steps = ip.sampling_period

        # check_thermalized
        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_periods

        # average_forces
        g.average_forces.input.n_children = ip.n_images
        g.average_forces.broadcast.sample = gp.constrained_evolution.output.forces[-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])


class VirtualWorkSerial(VirtualWorkParallel):
    """Work in progress, not yet functioning."""

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.check_steps = IsGEq()
        g.constrained_evolution = SerialList(ConstrainedMD)
        g.check_thermalized = IsGEq()
        g.average_forces = SerialList(WelfordOnline)
        g.clock = Counter()


class Milestoning(Protocol):
    """
    Calculates the jump frequencies of each centroid to the final centroid

    Input attributes:
        structure (Atoms): The initial or final structure to get cell and pbc
        relaxed_centroids_list (list/numpy.ndarray): The centroids on the fully relaxed string
        job_type ('Lammps'/'Vasp'/'Sphinx'): The job type. The listed types are allowed by the code, but may or may not
            be available on your system. (Default is 'Lammps')
        potential (str): The classical potential to use. (Does not exist by default, needs to be set after
            instantiation. Mandatory now, but in the future only needed when the job type is 'Lammps' or another
            classical interpreter that uses potentials.
        n_steps (int): How many steps to run for. (Default is 100.)

        structure (Atoms): The initial or final structure whose number and species of atoms, cell, and pbc are those of
            the Voronoi centroids.
        all_centroid_positions (list/numpy.ndarray): The centroid positions along the discretized transition path.
        n_images (int): How many images the path is divided into. TODO: Extract this from `all_centroid_positions`.
        n_steps (int): How many steps to run for.
        thermalization_steps (int): How many steps to run for before beginning to track transitions.
        temperature (float): Temperature to run at in K.
        temperature_damping_timescale (float): Langevin thermostat timescale in fs. (Default is 100 fs.)
        overheat_fraction (float): The fraction by which to overheat the initial velocities. This can be useful for
            more quickly equilibrating a system whose initial structure is its fully relaxed positions -- in which
            case equipartition of energy tells us that the kinetic energy should be initialized to double the
            desired value. (Default is 2.0, rely on equipartition.)
        time_step (float): MD timestep in fs. (Default is 1 fs.)
    """

    def __init__(self, project=None, name=None, job_name=None):
        super(Milestoning, self).__init__(project=project, name=name, job_name=job_name)

        # Protocol defaults
        id_ = self.input.default
        id_.time_step = 1.
        id_.overheat_fraction = 2.
        id_.damping_timescale = 100.

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.initial_velocities = SerialList(RandomVelocity)
        g.initial_forces = Zeros()
        g.check_steps = IsGEq()
        g.verlet_positions = SerialList(VerletPositionUpdate)
        g.calc_static = AutoList(ExternalHamiltonian)
        g.verlet_velocities = SerialList(VerletVelocityUpdate)
        g.check_thermalized = IsGEq()
        g.milestone = MilestoningVertex()
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.initial_velocities,
            g.initial_forces,
            g.check_steps, 'false',
            g.verlet_positions,
            g.calc_static,
            g.verlet_velocities,
            g.milestone,
            g.clock,
            g.check_steps
        )
        g.starting_vertex = g.initial_velocities
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_images
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # verlet_positions
        g.verlet_positions.input.n_children = ip.n_images
        g.verlet_positions.direct.time_step = ip.time_step
        g.verlet_positions.direct.masses = ip.structure.get_masses
        g.verlet_positions.direct.temperature = ip.temperature
        g.verlet_positions.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_positions.broadcast.default.positions = ip.all_centroid_positions
        g.verlet_positions.broadcast.default.velocities = gp.initial_velocities.output.velocities[-1]
        g.verlet_positions.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.verlet_positions.broadcast.positions = gp.milestone.output.positions[-1]
        g.verlet_positions.broadcast.velocities = gp.milestone.output.velocities[-1]
        g.verlet_positions.broadcast.forces = gp.milestone.output.forces[-1]

        # calc_static
        g.calc_static.input.n_children = ip.n_images
        g.calc_static.direct.ref_job_full_path = ip.ref_job_full_path
        g.calc_static.direct.structure = ip.structure

        g.calc_static.broadcast.positions = gp.verlet_positions.output.positions[-1]

        # verlet_velocities
        g.verlet_velocities.input.n_children = ip.n_images
        g.verlet_velocities.direct.time_step = ip.time_step
        g.verlet_velocities.direct.masses = ip.structure.get_masses
        g.verlet_velocities.direct.temperature = ip.temperature
        g.verlet_velocities.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_velocities.broadcast.velocities = gp.verlet_positions.output.velocities[-1]
        g.verlet_velocities.broadcast.forces = gp.calc_static.output.forces[-1]

        # Milestoning
        g.milestone.input.default.prev_positions_list = ip.all_centroid_positions
        g.milestone.input.default.prev_velocities_list = gp.initial_velocities.output.velocities[-1]
        g.milestone.input.default.prev_forces_list = gp.initial_forces.output.zeros[-1]

        g.milestone.input.prev_positions_list = gp.milestone.output.positions_list[-1]
        g.milestone.input.prev_velocities_list = gp.milestone.output.velocities_list[-1]
        g.milestone.input.prev_forces_list = gp.milestone.output.forces_list[-1]

        g.milestone.input.positions_list = gp.verlet_positions.output.positions[-1]
        g.milestone.input.velocities_list = gp.verlet_velocities.output.velocities[-1]
        g.milestone.input.forces_list = gp.calc_static.output.forces[-1]

        g.milestone.input.all_centroid_positions = ip.all_centroid_positions

        g.milestone.input.thermalization_steps = ip.thermalization_steps
        g.milestone.input.cell = ip.structure.cell
        g.milestone.input.pbc = ip.structure.pbc

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            # Nothing for now
        }

    def get_transitions(self):
        """
        Generates jump frequencies of each centroid to the final centroid form the reflections matrix, edge reflections
        matrix, and the edge time matrix stored by the milestoning vertex.

        Returns:
            (float): The mean time of first passage from the 0th image to the final image.
            (list): `n_images - 1` mean times of passage from the 0th Voronoi cell to the final cell.
            (list): Respective equilibrium probabilities to find the system in each Voronoi cell.

        TODO: Convert the final frequency to THz?
        """
        time_step = self.input.time_step
        reflections_matrix = self.graph.milestone.output.reflections_matrix[-1]
        edge_reflections_matrix = self.graph.milestone.output.edge_reflections_matrix[-1]
        edge_time_matrix = self.graph.milestone.output.edge_time_matrix[-1]
        # TODO: Add a `frame` input variable and rely on `graph.milestone.archive.output.X[frame]` instead

        n_images = len(reflections_matrix)
        n_edges = int(n_images * (n_images - 1) / 2)
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
        # q_ij = np.array(q_ij)  # TODO: pycharm claims this variable is never used again -- remove re-casting or use?

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
        mean_free_passage_times = [t_n[summation]]  # TODO: Consider longer but more transparent variable names
        for i in np.arange(2, len(t_n)):
            summation += i
            if summation < len(t_n):
                mean_free_passage_times += [t_n[summation]]
        mean_free_passage_times = np.array(mean_free_passage_times)
        jump_frequencies = 1. / mean_free_passage_times

        return jump_frequencies, mean_free_passage_times, pis

    @staticmethod
    def _get_pi(reflections_matrix, n_images):
        dia = np.eye(n_images) * np.sum(reflections_matrix, axis=1)
        pi_mat_a = np.append(reflections_matrix.T - dia, [np.ones(n_images)], axis=0)
        pi_vec_b = np.append(np.zeros(n_images), [1])

        return np.linalg.lstsq(pi_mat_a, pi_vec_b, rcond=None)[0]
