# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

from pyiron_contrib.protocol.generic import CompoundVertex, Protocol
from pyiron_contrib.protocol.primitive.one_state import Counter, ExternalHamiltonian, WeightedSum, \
    HarmonicHamiltonian, Transpose, RandomVelocity, LangevinThermostat, \
    VerletPositionUpdate, VerletVelocityUpdate, BuildMixingPairs, DeleteAtom, Overwrite, Slice, VoronoiReflection, \
    WelfordOnline, Zeros, TILDPostProcess
from pyiron_contrib.protocol.primitive.two_state import IsGEq, ModIsZero
from pyiron_contrib.protocol.list import SerialList, ParallelList, AutoList
from pyiron_contrib.protocol.utils import Pointer
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from abc import ABC, abstractmethod

KB = physical_constants['Boltzmann constant in eV/K'][0]
HBAR = physical_constants['Planck constant over 2 pi in eV s'][0]
ROOT_EV_PER_ANGSTROM_SQUARE_PER_AMU_IN_S = 9.82269385e13
# https://www.wolframalpha.com/input/?i=sqrt((eV)+%2F+((atomic+mass+units)*(angstroms%5E2)))

"""
Protocols for thermodynamic integration using langevin dynamics.
"""

__author__ = "Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "24 July, 2019"


class TILDParent(CompoundVertex, ABC):
    """
    A parent class for thermodynamic integration by langevin dynamics. Mostly just to avoid duplicate code in
    `HarmonicTILD` and `VacancyTILD`.

    Assumes the presence of `build_lambdas`, `average` (for the thermodynamic average of the integrand), `reflect`
    (to keep each atom closest to its own lattice site), and `mix` (to combine the forces from different
    representations).

    WARNING: The methods in this parent class require loading of the finished interactive jobs that run within the
    child protocol. Since reloading jobs is (at times) time consuming, a TILDPostProcessing node is added at the end
    of the child protocol. This makes the methods defined in this parent class redundant. BUT, they will be important
    if we wish to see the integrand plots, as I leave the default flag for plotting as False.

    # TODO: Make reflection optional; it makes sense for crystals, but maybe not for molecules
    """

    def get_lambdas(self):
        return self.graph.build_lambdas.output.lambda_pairs[-1][:, 0]

    def get_integrand(self):
        integrand = self.graph.average.output
        return integrand.mean[-1], integrand.std[-1] / np.sqrt(integrand.n_samples[-1])

    def plot_integrand(self):
        fig, ax = plt.subplots()
        lambdas = self.get_lambdas()
        thermal_average, standard_error = self.get_integrand()
        ax.plot(lambdas, thermal_average, marker='o')
        ax.fill_between(lambdas, thermal_average - standard_error, thermal_average + standard_error, alpha=0.3)
        ax.set_xlabel("Lambda")
        ax.set_ylabel("dF/dLambda")
        return fig, ax

    def get_free_energy_change(self):
        return np.trapz(x=self.get_lambdas(), y=self.get_integrand()[0])


class HarmonicTILD(TILDParent):
    """
    """
    DefaultWhitelist = {
    }

    def __init__(self, **kwargs):
        super(HarmonicTILD, self).__init__(**kwargs)

        id_ = self.input.default
        id_.n_steps = 100
        id_.temperature_damping_timescale = 100.
        id_.overheat_fraction = 2.
        id_.time_step = 1.
        id_.sampling_period = 1
        id_.fix_com = True
        id_.use_reflection = True
        # TODO: Need more than input and default, but rather access order, to work without reflection...
        id_.custom_lambdas = None
        id_.thermalization_steps = 10
        id_.plot = False

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.build_lambdas = BuildMixingPairs()
        g.initial_forces = Zeros()
        g.initial_velocity = SerialList(RandomVelocity)
        g.check_steps = IsGEq()
        g.verlet_positions = SerialList(VerletPositionUpdate)
        g.reflect = SerialList(VoronoiReflection)
        g.calc_static = AutoList(ExternalHamiltonian)
        g.harmonic = SerialList(HarmonicHamiltonian)
        g.transpose_forces = Transpose()
        g.mix = SerialList(WeightedSum)
        g.verlet_velocities = SerialList(VerletVelocityUpdate)
        g.check_thermalized = IsGEq()
        g.check_sampling_period = ModIsZero()
        g.average = SerialList(WelfordOnline)
        g.clock = Counter()
        g.post = TILDPostProcess()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.build_lambdas,
            g.initial_forces,
            g.initial_velocity,
            g.check_steps, 'false',
            g.clock,
            g.verlet_positions,
            g.reflect,
            g.calc_static,
            g.harmonic,
            g.transpose_forces,
            g.mix,
            g.verlet_velocities,
            g.check_thermalized, 'true',
            g.check_sampling_period, 'true',
            g.average,
            g.check_steps, 'true',
            g.post
        )
        g.make_edge(g.check_thermalized, g.check_steps, 'false')
        g.make_edge(g.check_sampling_period, g.check_steps, 'false')
        g.starting_vertex = g.build_lambdas
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        g.build_lambdas.input.n_lambdas = ip.n_lambdas
        g.build_lambdas.input.custom_lambdas = ip.custom_lambdas

        g.initial_forces.input.shape = ip.structure.positions.shape

        g.initial_velocity.input.n_children = ip.n_lambdas
        g.initial_velocity.direct.temperature = ip.temperature
        g.initial_velocity.direct.masses = ip.structure.get_masses
        g.initial_velocity.direct.overheat_fraction = ip.overheat_fraction

        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

        g.verlet_positions.input.n_children = ip.n_lambdas
        g.verlet_positions.direct.default.positions = ip.structure.positions
        g.verlet_positions.broadcast.default.velocities = gp.initial_velocity.output.velocities[-1]
        g.verlet_positions.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.verlet_positions.broadcast.positions = gp.reflect.output.positions[-1]
        g.verlet_positions.broadcast.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.broadcast.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_positions.direct.masses = ip.structure.get_masses
        g.verlet_positions.direct.time_step = ip.time_step
        g.verlet_positions.direct.temperature = ip.temperature
        g.verlet_positions.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        g.reflect.input.n_children = ip.n_lambdas
        g.reflect.direct.default.previous_positions = ip.structure.positions
        g.reflect.broadcast.default.previous_velocities = gp.initial_velocity.output.velocities[-1]

        g.reflect.direct.reference_positions = ip.structure.positions
        g.reflect.direct.pbc = ip.structure.pbc
        g.reflect.direct.cell = ip.structure.cell
        g.reflect.broadcast.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.broadcast.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.broadcast.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.broadcast.previous_velocities = gp.reflect.output.velocities[-1]

        g.calc_static.input.n_children = ip.n_lambdas
        g.calc_static.direct.ref_job_full_path = ip.ref_job_full_path
        g.calc_static.direct.structure = ip.structure
        g.calc_static.broadcast.positions = gp.reflect.output.positions[-1]

        g.harmonic.input.n_children = ip.n_lambdas
        g.harmonic.direct.spring_constant = ip.spring_constant
        g.harmonic.direct.home_positions = ip.structure.positions
        g.harmonic.broadcast.positions = gp.reflect.output.positions[-1]
        g.harmonic.direct.cell = ip.structure.cell
        g.harmonic.direct.pbc = ip.structure.pbc

        g.average.input.n_children = ip.n_lambdas
        g.average.broadcast.sample = gp.calc_static.output.energy_pot[-1]

        g.transpose_forces.input.matrix = [
            gp.calc_static.output.forces[-1],
            gp.harmonic.output.forces[-1]
        ]

        g.mix.input.n_children = ip.n_lambdas
        g.mix.broadcast.vectors = gp.transpose_forces.output.matrix_transpose[-1]
        g.mix.broadcast.weights = gp.build_lambdas.output.lambda_pairs[-1]

        g.verlet_velocities.input.n_children = ip.n_lambdas
        g.verlet_velocities.broadcast.velocities = gp.reflect.output.velocities[-1]
        g.verlet_velocities.broadcast.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_velocities.direct.masses = ip.structure.get_masses
        g.verlet_velocities.direct.time_step = ip.time_step
        g.verlet_velocities.direct.temperature = ip.temperature
        g.verlet_velocities.direct.temperature_damping_timescale = ip.temperature_damping_timescale
        g.verlet_velocities.direct.time_step = ip.time_step

        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = ip.sampling_period

        g.post.input.lambda_pairs = gp.build_lambdas.output.lambda_pairs[-1]
        g.post.input.n_samples = gp.average.output.n_samples[-1]
        g.post.input.mean = gp.average.output.mean[-1]
        g.post.input.std = gp.average.output.std[-1]
        g.post.input.plot = ip.plot

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'job_energy_pot': ~gp.calc_static.output.energy_pot[-1],
            'harmonic_energy_pot': ~gp.harmonic.output.energy_pot[-1],
            'energy_kin': ~gp.verlet_velocities.output.energy_kin[-1],
            'positions': ~gp.reflect.output.positions[-1],
            'velocities': ~gp.verlet_velocities.output.velocities[-1],
            'forces': ~gp.mix.output.weighted_sum[-1],
            'free_energy_change': ~gp.post.output.free_energy_change[-1]
        }

    def get_classical_harmonic_free_energy(self, temperatures=None):
        """
        Get the total free energy of a harmonic oscillator with this frequency and these atoms. Temperatures are clipped
        at 1 micro-Kelvin.

        Returns:
            float/np.ndarray: The sum of the free energy of each atom.
        """
        if temperatures is None:
            temperatures = self.input.temperature
        temperatures = np.clip(temperatures, 1e-6, np.inf)
        beta = 1. / (KB * temperatures)

        return -3 * len(self.input.structure) * np.log(np.pi / (self.input.spring_constant * beta)) / (2 * beta)

    def get_quantum_harmonic_free_energy(self, temperatures=None):
        """
        Get the total free energy of a harmonic oscillator with this frequency and these atoms. Temperatures are clipped
        at 1 micro-Kelvin.

        Returns:
            float/np.ndarray: The sum of the free energy of each atom.
        """
        if temperatures is None:
            temperatures = self.input.temperature
        temperatures = np.clip(temperatures, 1e-6, np.inf)
        beta = 1. / (KB * temperatures)
        f = 0
        for m in self.input.structure.get_masses():
            hbar_omega = HBAR * np.sqrt(self.input.spring_constant / m) * ROOT_EV_PER_ANGSTROM_SQUARE_PER_AMU_IN_S
            f += (3. / 2) * hbar_omega + ((3. / beta) * np.log(1 - np.exp(-beta * hbar_omega)))
        return f


class ProtocolHarmonicTILD(Protocol, HarmonicTILD):
    pass


class VacancyTILD(TILDParent):
    """

    """
    DefaultWhitelist = {
    }

    def __init__(self, **kwargs):
        super(VacancyTILD, self).__init__(**kwargs)

        id_ = self.input.default
        id_.n_steps = 100
        id_.vacancy_id = 0
        id_.temperature_damping_timescale = 100.
        id_.overheat_fraction = 2.
        id_.time_step = 1.
        id_.sampling_period = 1
        id_.fix_com = True
        id_.use_reflection = True
        # TODO: Need more than input and default, but rather access order, to work without reflection...
        id_.custom_lambdas = None
        id_.thermalization_steps = 10
        id_.plot = False

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.delete_vacancy = DeleteAtom()
        g.build_lambdas = BuildMixingPairs()
        g.random_velocity = SerialList(RandomVelocity)
        g.initial_forces = Zeros()
        g.slice_structure = Slice()
        g.check_steps = IsGEq()
        g.clock = Counter()
        g.verlet_positions = SerialList(VerletPositionUpdate)
        g.reflect = SerialList(VoronoiReflection)
        g.calc_full = SerialList(ExternalHamiltonian)
        g.slice_positions = SerialList(Slice)
        g.calc_vac = SerialList(ExternalHamiltonian)
        g.slice_harmonic = SerialList(Slice)
        g.harmonic = SerialList(HarmonicHamiltonian)
        g.write_vac_forces = SerialList(Overwrite)
        g.write_harmonic_forces = SerialList(Overwrite)
        g.transpose_lambda = Transpose()
        g.mix = SerialList(WeightedSum)
        g.verlet_velocities = SerialList(VerletVelocityUpdate)
        g.check_thermalized = IsGEq()
        g.check_sampling_period = ModIsZero()
        g.transpose_energies = Transpose()
        g.addition = SerialList(WeightedSum)
        g.average = SerialList(WelfordOnline)
        g.post = TILDPostProcess()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.delete_vacancy,
            g.build_lambdas,
            g.random_velocity,
            g.initial_forces,
            g.slice_structure,
            g.check_steps, 'false',
            g.clock,
            g.verlet_positions,
            g.reflect,
            g.calc_full,
            g.slice_positions,
            g.calc_vac,
            g.slice_harmonic,
            g.harmonic,
            g.write_vac_forces,
            g.write_harmonic_forces,
            g.transpose_lambda,
            g.mix,
            g.verlet_velocities,
            g.check_thermalized, 'true',
            g.check_sampling_period, 'true',
            g.transpose_energies,
            g.addition,
            g.average,
            g.check_steps, 'true',
            g.post
        )
        g.make_edge(g.check_thermalized, g.check_steps, 'false')
        g.make_edge(g.check_sampling_period, g.check_steps, 'false')
        g.starting_vertex = self.graph.delete_vacancy
        g.restarting_vertex = self.graph.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        g.delete_vacancy.input.structure = ip.structure
        g.delete_vacancy.input.id = ip.vacancy_id
        shared_ids = gp.delete_vacancy.output.mask[-1]

        g.build_lambdas.input.n_lambdas = ip.n_lambdas
        g.build_lambdas.input.custom_lambdas = ip.custom_lambdas
        # n_children = graph_pointer.build_lambdas.output.lambda_pairs[-1].__len__
        # This doesn't yet work because utils can't import MethodWrapperType and use it at line 305 until I have py 3.7

        g.random_velocity.input.n_children = ip.n_lambdas  # n_children
        g.random_velocity.direct.temperature = ip.temperature
        g.random_velocity.direct.masses = ip.structure.get_masses
        g.random_velocity.direct.overheat_fraction = ip.overheat_fraction

        g.initial_forces.input.shape = ip.structure.positions.shape

        g.slice_structure.input.vector = ip.structure.positions
        g.slice_structure.input.mask = ip.vacancy_id
        g.slice_structure.input.ensure_iterable_mask = True  # To keep positions (1,3) instead of (3,)

        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

        g.verlet_positions.input.n_children = ip.n_lambdas
        g.verlet_positions.direct.default.positions = ip.structure.positions
        g.verlet_positions.broadcast.default.velocities = gp.random_velocity.output.velocities[-1]
        g.verlet_positions.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.verlet_positions.broadcast.positions = gp.reflect.output.positions[-1]
        g.verlet_positions.broadcast.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.broadcast.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_positions.direct.masses = ip.structure.get_masses
        g.verlet_positions.direct.time_step = ip.time_step
        g.verlet_positions.direct.temperature = ip.temperature
        g.verlet_positions.direct.temperature_damping_timescale = ip.temperature_damping_timescale
        g.verlet_positions.direct.time_step = ip.time_step

        g.reflect.input.n_children = ip.n_lambdas
        g.reflect.direct.default.previous_positions = ip.structure.positions
        g.reflect.broadcast.default.previous_velocities = gp.random_velocity.output.velocities[-1]

        g.reflect.direct.reference_positions = ip.structure.positions
        g.reflect.direct.pbc = ip.structure.pbc
        g.reflect.direct.cell = ip.structure.cell
        g.reflect.broadcast.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.broadcast.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.broadcast.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.broadcast.previous_velocities = gp.verlet_velocities.output.velocities[-1]

        g.calc_full.input.n_children = ip.n_lambdas  # n_children
        g.calc_full.direct.ref_job_full_path = ip.ref_job_full_path
        g.calc_full.direct.structure = ip.structure
        g.calc_full.broadcast.positions = gp.reflect.output.positions[-1]

        g.slice_positions.input.n_children = ip.n_lambdas
        g.slice_positions.broadcast.vector = gp.reflect.output.positions[-1]
        g.slice_positions.direct.mask = shared_ids

        g.calc_vac.input.n_children = ip.n_lambdas  # n_children
        g.calc_vac.direct.ref_job_full_path = ip.ref_job_full_path
        g.calc_vac.direct.structure = gp.delete_vacancy.output.structure[-1]
        g.calc_vac.broadcast.positions = gp.slice_positions.output.sliced[-1]

        g.slice_harmonic.input.n_children = ip.n_lambdas
        g.slice_harmonic.broadcast.vector = gp.reflect.output.positions[-1]
        g.slice_harmonic.direct.mask = ip.vacancy_id
        g.slice_harmonic.direct.ensure_iterable_mask = True

        g.harmonic.input.n_children = ip.n_lambdas
        g.harmonic.direct.spring_constant = ip.spring_constant
        g.harmonic.direct.home_positions = gp.slice_structure.output.sliced[-1]
        g.harmonic.broadcast.positions = gp.slice_harmonic.output.sliced[-1]
        g.harmonic.direct.cell = ip.structure.cell
        g.harmonic.direct.pbc = ip.structure.pbc

        g.write_vac_forces.input.n_children = ip.n_lambdas
        g.write_vac_forces.broadcast.target = gp.calc_full.output.forces[-1]
        g.write_vac_forces.direct.mask = shared_ids
        g.write_vac_forces.broadcast.new_values = gp.calc_vac.output.forces[-1]

        g.write_harmonic_forces.input.n_children = ip.n_lambdas
        g.write_harmonic_forces.broadcast.target = gp.write_vac_forces.output.overwritten[-1]
        g.write_harmonic_forces.direct.mask = ip.vacancy_id
        g.write_harmonic_forces.broadcast.new_values = gp.harmonic.output.forces[-1]

        g.transpose_lambda.input.matrix = [
            gp.calc_full.output.forces[-1],
            gp.write_harmonic_forces.output.overwritten[-1]
        ]

        g.mix.input.n_children = ip.n_lambdas
        g.mix.broadcast.vectors = gp.transpose_lambda.output.matrix_transpose[-1]
        g.mix.broadcast.weights = gp.build_lambdas.output.lambda_pairs[-1]

        g.verlet_velocities.input.n_children = ip.n_lambdas
        g.verlet_velocities.broadcast.velocities = gp.reflect.output.velocities[-1]
        g.verlet_velocities.broadcast.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_velocities.direct.masses = ip.structure.get_masses
        g.verlet_velocities.direct.time_step = ip.time_step
        g.verlet_velocities.direct.temperature = ip.temperature
        g.verlet_velocities.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = ip.sampling_period

        g.transpose_energies.input.matrix = [
            gp.calc_vac.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1],
            gp.calc_full.output.energy_pot[-1]
        ]

        g.addition.input.n_children = ip.n_lambdas
        g.addition.broadcast.vectors = gp.transpose_energies.output.matrix_transpose[-1]
        g.addition.direct.weights = [1, 1, -1]

        g.average.input.n_children = ip.n_lambdas
        g.average.broadcast.sample = gp.addition.output.weighted_sum[-1]

        g.post.input.lambda_pairs = gp.build_lambdas.output.lambda_pairs[-1]
        g.post.input.n_samples = gp.average.output.n_samples[-1]
        g.post.input.mean = gp.average.output.mean[-1]
        g.post.input.std = gp.average.output.std[-1]
        g.post.input.plot = ip.plot

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'energy_kin': ~gp.verlet_velocities.output.energy_kin[-1],
            'positions': ~gp.reflect.output.positions[-1],
            'velocities': ~gp.verlet_velocities.output.velocities[-1],
            'forces': ~gp.mix.output.weighted_sum[-1],
            'average': ~gp.average.output.mean[-1],
            'free_energy_change': ~gp.post.output.free_energy_change[-1]
        }


class ProtocolVacancyTILD(Protocol, VacancyTILD):
    pass


class HarmonicTILDParallel(HarmonicTILD):
    DefaultWhitelist = {}

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.build_lambdas = BuildMixingPairs()
        g.run_lambda_points = ParallelList(HarmonicallyCoupled)

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.build_lambdas,
            g.run_lambda_points
        )
        g.starting_vertex = g.build_lambdas
        g.restarting_vertex = g.run_lambda_points

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        g.build_lambdas.input.n_lambdas = ip.n_lambdas
        g.build_lambdas.input.custom_lambdas = ip.custom_lambdas

        g.run_lambda_points.input.n_children = ip.n_lambdas
        g.run_lambda_points.direct.n_steps = ip.n_steps
        g.run_lambda_points.direct.temperature = ip.temperature
        g.run_lambda_points.direct.masses = ip.structure.get_masses
        g.run_lambda_points.direct.overheat_fraction = ip.overheat_fraction
        g.run_lambda_points.direct.threshold = ip.n_steps
        g.run_lambda_points.direct.ref_job_full_path = ip.ref_job_full_path
        g.run_lambda_points.direct.structure = ip.structure
        g.run_lambda_points.direct.spring_constant = ip.spring_constant
        g.run_lambda_points.broadcast.coupling_weights = gp.build_lambdas.output.lambda_pairs[-1]
        g.run_lambda_points.direct.temperature_damping_timescale = ip.temperature_damping_timescale
        g.run_lambda_points.direct.time_step = ip.time_step
        g.run_lambda_points.direct.fix_com = ip.fix_com
        g.run_lambda_points.direct.use_reflection = ip.use_reflection

    def get_output(self):
        o = Pointer(self.graph.run_lambda_points.output)
        return {
            'job_energy_pot': ~o.job_energy_pot[-1],
            'harmonic_energy_pot': ~o.harmonic_energy_pot[-1],
            'energy_kin': ~o.energy_kin[-1],
            'positions': ~o.positions[-1],
            'velocities': ~o.velocities[-1],
            'forces': ~o.forces[-1],
        }


class ProtocolHarmonicTILDParallel(Protocol, HarmonicTILDParallel):
    pass


class HarmonicallyCoupled(CompoundVertex):
    DefaultWhitelist = {
        'reflect': {
            'output': {
                'positions': 1000,
            },
        },
        'calc_static': {
            'output': {
                'energy_pot': 1,
            },
        },
        'harmonic': {
            'output': {
                'energy_pot': 1,
            },
        },
        'mix': {
            'output': {
                'weighted_sum': 1,
            }
        },
        'verlet_velocities': {
            'output': {
                'energy_kin': 1,
                'velocities': 1000,
            },
        },
    }

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.initial_velocity = RandomVelocity()
        g.initial_forces = Zeros()
        g.check_steps = IsGEq()
        g.verlet_positions = VerletPositionUpdate()
        g.reflect = VoronoiReflection()
        g.calc_static = ExternalHamiltonian()
        g.harmonic = HarmonicHamiltonian()
        g.mix = WeightedSum()
        g.verlet_velocities = VerletVelocityUpdate()
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.initial_velocity,
            g.initial_forces,
            g.check_steps, 'false',
            g.verlet_positions,
            g.reflect,
            g.calc_static,
            g.harmonic,
            g.mix,
            g.verlet_velocities,
            g.clock,
            g.check_steps,
        )
        g.starting_vertex = g.initial_velocity
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        g.initial_velocity.input.temperature = ip.temperature
        g.initial_velocity.input.masses = ip.structure.get_masses
        g.initial_velocity.input.overheat_fraction = ip.overheat_fraction

        g.initial_forces.input.shape = ip.structure.positions.shape

        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        g.verlet_positions.input.default.positions = ip.structure.positions
        g.verlet_positions.input.default.velocities = gp.initial_velocity.output.velocities[-1]
        g.verlet_positions.input.default.forces = gp.initial_forces.output.zeros[-1]
        g.verlet_positions.input.positions = gp.reflect.output.positions[-1]
        g.verlet_positions.input.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.input.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_positions.input.masses = ip.structure.get_masses
        g.verlet_positions.input.time_step = ip.time_step
        g.verlet_positions.input.temperature = ip.temperature
        g.verlet_positions.input.temperature_damping_timescale = ip.temperature_damping_timescale

        g.reflect.on = ip.use_reflection
        g.reflect.input.reference_positions = ip.structure.positions
        g.reflect.input.pbc = ip.structure.pbc
        g.reflect.input.cell = ip.structure.cell
        g.reflect.input.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.input.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.input.default.previous_positions = ip.structure.positions
        g.reflect.input.default.previous_velocities = gp.initial_velocity.output.velocities[-1]
        g.reflect.input.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.input.previous_velocities = gp.reflect.output.velocities[-1]

        g.calc_static.input.ref_job_full_path = ip.ref_job_full_path
        g.calc_static.input.structure = ip.structure
        g.calc_static.input.default.positions = gp.verlet_positions.output.positions[-1]
        g.calc_static.input.positions = gp.reflect.output.positions[-1]

        g.harmonic.input.spring_constant = ip.spring_constant
        g.harmonic.input.home_positions = ip.structure.positions
        g.harmonic.input.default.positions = gp.verlet_positions.output.positions[-1]
        g.harmonic.input.positions = gp.reflect.output.positions[-1]
        g.harmonic.input.cell = ip.structure.cell
        g.harmonic.input.pbc = ip.structure.pbc

        g.mix.input.vectors = [
            gp.calc_static.output.forces[-1],
            gp.harmonic.output.forces[-1]
        ]
        g.mix.input.weights = ip.coupling_weights

        g.verlet_velocities.input.default.velocities = gp.verlet_positions.output.velocities[-1]
        g.verlet_velocities.input.velocities = gp.reflect.output.velocities[-1]
        g.verlet_velocities.input.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_velocities.input.masses = ip.structure.get_masses
        g.verlet_velocities.input.time_step = ip.time_step

        self.archive.clock = gp.clock.output.n_counts[-1]
        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'job_energy_pot': ~gp.calc_static.output.energy_pot[-1],
            'harmonic_energy_pot': ~gp.harmonic.output.energy_pot[-1],
            'energy_kin': ~gp.verlet_velocities.output.energy_kin[-1],
            'positions': ~gp.reflect.output.positions[-1],
            'velocities': ~gp.verlet_velocities.output.velocities[-1],
            'forces': ~gp.mix.output.weighted_sum[-1],
        }

    def parallel_setup(self):
        self.graph.calc_static._initialize(self.input.ref_job_full_path, self.input.structure)
