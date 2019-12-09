# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

from pyiron_contrib.protocol.generic import Protocol
from pyiron_contrib.protocol.primitive.one_state import Counter, ExternalHamiltonian, WeightedSum, \
    HarmonicHamiltonian, Transpose, RandomVelocity, LangevinThermostat, \
    VerletPositionUpdate, VerletVelocityUpdate, BuildMixingPairs, DeleteAtom, Overwrite, Slice, VoronoiReflection, \
    WelfordOnline, Zeros
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


class TILDParent(Protocol, ABC):
    """
    A parent class for thermodynamic integration by langevin dynamics. Mostly just to avoid duplicate code in
    `HarmonicTILD` and `VacancyTILD`.

    Assumes the presence of `build_lambdas`, `average` (for the thermodynamic average of the integrand), `reflect`
    (to keep each atom closest to its own lattice site), and `mix` (to combine the forces from different
    representations).

    # TODO: Make reflection optional; it makes sense for crystals, but maybe not for molecules
    """

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'integrand': ~gp.average.output.mean[-1],
            'std': ~gp.average.output.std[-1],
            'n_samples': ~gp.average.output.n_samples[-1],
            'positions': ~gp.reflect.output.positions[-1],
            'velocities': ~gp.reflect.output.velocities[-1],
            'forces': ~gp.mix.output.weighted_sum[-1]
        }

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

    def __init__(self, project=None, name=None, job_name=None):
        super(HarmonicTILD, self).__init__(project=project, name=name, job_name=job_name)

        self.input.default.n_steps = 100
        self.input.temperature_damping_timescale = 10.

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.build_lambdas = BuildMixingPairs()
        g.initial_forces = Zeros()
        g.random_velocity = SerialList(RandomVelocity)
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
        g.transpose_energies = Transpose()
        g.subtract = SerialList(WeightedSum)
        g.average = SerialList(WelfordOnline)
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.build_lambdas,
            g.initial_forces,
            g.random_velocity,
            g.check_steps, 'false',
            g.verlet_positions,
            g.reflect,
            g.calc_static,
            g.harmonic,
            g.transpose_forces,
            g.mix,
            g.verlet_velocities,
            g.check_thermalized, 'true',
            g.check_sampling_period, 'true',
            g.transpose_energies,
            g.subtract,
            g.average,
            g.clock,
            g.check_steps,
        )
        g.make_edge(g.check_thermalized, g.clock, 'false')
        g.make_edge(g.check_sampling_period, g.clock, 'false')
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

        g.random_velocity.input.n_children = ip.n_lambdas
        g.random_velocity.direct.temperature = ip.temperature
        g.random_velocity.direct.masses = ip.structure.get_masses
        g.random_velocity.direct.overheat_fraction = ip.overheat_fraction

        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

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
        g.check_sampling_period.input.default.mod = Pointer(self.archive.period)
        g.check_sampling_period.input.mod = ip.sampling_period

        g.transpose_energies.input.matrix = [
            gp.calc_static.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1]
        ]

        g.subtract.input.n_children = ip.n_lambdas
        g.subtract.broadcast.vectors = gp.transpose_energies.output.matrix_transpose[-1]
        g.subtract.direct.weights = [1, -1]

        g.average.input.n_children = ip.n_lambdas
        g.average.broadcast.sample = gp.subtract.output.weighted_sum[-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

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


class VacancyTILD(TILDParent):
    """

    """

    def __init__(self, project=None, name=None, job_name=None):
        super(VacancyTILD, self).__init__(project=project, name=name, job_name=job_name)

        self.input.default.n_steps = 100
        self.input.default.vacancy_id = 0
        self._vacancy_structure_init = None
        self.input.temperature_damping_timescale = 10.

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.delete_vacancy = DeleteAtom()
        g.build_lambdas = BuildMixingPairs()
        g.random_velocity = SerialList(RandomVelocity)
        g.initial_forces = Zeros()
        g.slice_structure = Slice()
        g.check_steps = IsGEq()
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
        g.clock = Counter()

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
            g.clock,
            g.check_steps,
        )
        g.make_edge(g.check_thermalized, g.clock, 'false')
        g.make_edge(g.check_sampling_period, g.clock, 'false')
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
        g.check_sampling_period.input.default.mod = Pointer(self.archive.period)
        g.check_sampling_period.input.mod = ip.sampling_period

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

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])


# TODO: Adapt the code to run protocols smoothly as vertices in other protocols
class HarmonicTILDParallel(Protocol):

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.build_lambdas = BuildMixingPairs()
        g.run_lambda_points = ParallelList(HarmonicallyCoupled)
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        # g.make_edge(g.build_lambdas, g.run_lambda_points)
        g.make_pipeline(
            g.build_lambdas,
            g.run_lambda_points,
            g.clock
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
        g.run_lambda_points.direct.temperature = ip.temperature
        g.run_lambda_points.direct.masses = ip.structure
        g.run_lambda_points.direct.overheat_fraction = ip.overheat_fraction
        g.run_lambda_points.direct.threshold = ip.n_steps
        g.run_lambda_points.direct.ref_job_full_path = ip.ref_job_full_path
        g.run_lambda_points.direct.structure = ip.structure
        g.run_lambda_points.direct.spring_constant = ip.spring_constant
        g.run_lambda_points.broadcast.coupling_weights = gp.build_lambdas.output.lambda_pairs
        g.run_lambda_points.direct.damping_timescale = ip.damping_timescale
        g.run_lambda_points.direct.time_step = ip.time_step
        g.run_lambda_points.direct.fix_com = ip.fix_com
        g.run_lambda_points.direct.use_reflection = ip.use_reflection

        o = self.output
        child_output = gp.run_lambda_points.archive
        o.job_energy_pot = child_output.job_energy_pot
        o.harmonic_energy_pot = child_output.harmonic_energy_pot
        o.energy_kin = child_output.energy_kin
        o.positions = child_output.positions
        o.velocities = child_output.velocities
        o.forces = child_output.forces

        a = self.archive
        a.clock = gp.clock.output.n_counts[-1]
        g.run_lambda_points.archive.period = ip.sampling_period
        child_archive_output = gp.run_lambda_points.archive.output
        a.output.job_energy_pot = child_archive_output.job_energy_pot
        a.output.harmonic_energy_pot = child_archive_output.energy_pot
        a.output.energy_kin = child_archive_output.energy_kin
        a.output.positions = child_archive_output.positions
        a.output.velocities = child_archive_output.velocities
        a.output.forces = child_archive_output.forces

    def execute(self):
        print("Executing the parallel protocol")
        super(HarmonicTILDParallel, self).execute()


class HarmonicallyCoupled(Protocol):

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.random_velocity = RandomVelocity()
        g.check_steps = IsGEq()
        g.calc_static = ExternalHamiltonian()
        g.harmonic = HarmonicHamiltonian()
        g.mix = WeightedSum()
        g.thermostat = LangevinThermostat()
        g.sum = WeightedSum()
        g.verlet_velocities = VerletVelocityUpdate()
        g.verlet_positions = VerletPositionUpdate()
        g.reflect = VoronoiReflection()
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.random_velocity,
            g.check_steps, 'false',
            g.calc_static,
            g.harmonic,
            g.mix,
            g.thermostat,
            g.sum,
            g.verlet_velocities,
            g.verlet_positions,
            g.reflect,
            g.clock,
            g.check_steps,
        )
        g.starting_vertex = g.random_velocity
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        g.random_velocity.input.temperature = ip.temperature
        g.random_velocity.input.masses = ip.structure.get_masses
        g.random_velocity.input.overheat_fraction = ip.overheat_fraction

        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        g.calc_static.input.ref_job_full_path = ip.ref_job_full_path
        g.calc_static.input.structure = ip.structure
        g.calc_static.input.positions = gp.reflect.output.positions[-1]

        g.harmonic.input.spring_constant = ip.spring_constant
        g.harmonic.input.home_positions = ip.structure.positions
        g.harmonic.input.default.positions = ip.structure.positions
        g.harmonic.input.positions = gp.reflect.output.positions[-1]
        g.harmonic.input.cell = ip.structure.cell
        g.harmonic.input.pbc = ip.structure.pbc

        g.mix.input.vectors = [
            gp.calc_static.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1]
        ]
        g.mix.input.weights = ip.coupling_weights

        g.thermostat.input.default.velocities = gp.random_velocity.output.velocities[-1]
        g.thermostat.input.velocities = gp.reflect.output.velocities[-1]
        g.thermostat.input.masses = ip.structure.get_masses
        g.thermostat.input.temperature = ip.temperature
        g.thermostat.input.damping_timescale = ip.damping_timescale
        g.thermostat.input.time_step = ip.time_step
        g.thermostat.input.fix_com = ip.fix_com

        g.sum.input.vectors = [
            gp.mix.output.weighted_sum[-1],
            gp.thermostat.output.forces[-1]
        ]
        g.sum.input.weights = [1, 1]

        g.verlet_velocities.input.default.velocities = gp.random_velocity.output.velocities[-1]
        g.verlet_velocities.input.velocities = gp.reflect.output.velocities[-1]
        g.verlet_velocities.input.forces = gp.sum.output.weighted_sum[-1]
        g.verlet_velocities.input.masses = ip.structure.get_masses
        g.verlet_velocities.input.time_step = ip.time_step

        g.verlet_positions.input.default.positions = ip.structure.positions
        g.verlet_positions.input.default.velocities = gp.random_velocity.output.velocities[-1]
        g.verlet_positions.input.positions = gp.reflect.output.positions[-1]
        g.verlet_positions.input.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.input.forces = gp.sum.output.weighted_sum[-1]
        g.verlet_positions.input.masses = ip.structure.get_masses
        g.verlet_positions.input.time_step = ip.time_step

        g.reflect.on = ip.use_reflection
        g.reflect.input.reference_positions = ip.structure.positions
        g.reflect.input.pbc = ip.structure.pbc
        g.reflect.input.cell = ip.structure.cell
        g.reflect.input.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.input.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.input.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.input.default.previous_positions = ip.structure.positions
        g.reflect.input.previous_velocities = gp.verlet_velocities.output.velocities[-1]

        o = self.output
        o.job_energy_pot = gp.calc_static.output.energy_pot
        o.harmonic_energy_pot = gp.harmonic.output.energy_pot
        o.energy_kin = gp.verlet_velocities.output.energy_kin
        o.positions = gp.reflect.output.positions
        o.velocities = gp.reflect.output.velocities
        o.forces = gp.sum.output.weighted_sum

        a = self.archive
        a.clock = gp.clock.output.n_counts[-1]
        a.output.job_energy_pot = gp.calc_static.archive.output.energy_pot
        a.output.harmonic_energy_pot = gp.harmonic.archive.output.energy_pot
        a.output.energy_kin = gp.verlet_velocities.archive.output.energy_kin
        a.output.positions = gp.verlet_positions.archive.output.positions
        a.output.velocities = gp.verlet_velocities.archive.output.velocities
        a.output.forces = gp.sum.archive.output.weighted_sum

    def execute(self):
            print("Executing the child")
            super(HarmonicallyCoupled, self).execute()

    def parallel_setup(self):
        super(HarmonicallyCoupled, self).parallel_setup()