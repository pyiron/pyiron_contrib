# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from abc import ABC
import numpy as np
import matplotlib.pyplot as plt

from pyiron_contrib.protocol.generic import CompoundVertex, Protocol
from pyiron_contrib.protocol.list import SerialList, ParallelList
from pyiron_contrib.protocol.utils import Pointer
from pyiron_contrib.protocol.primitive.one_state import Counter, BuildMixingPairs, DeleteAtom, \
    ExternalHamiltonian, HarmonicHamiltonian, CreateJob, Overwrite, RandomVelocity, Slice, SphereReflection, \
    TILDPostProcess, Transpose, VerletPositionUpdate, VerletVelocityUpdate, WeightedSum, \
    WelfordOnline, Zeros, FEPExponential, AcceptanceCriterion
from pyiron_contrib.protocol.primitive.two_state import IsGEq, ModIsZero

from scipy.constants import physical_constants

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
    child protocol. Since reloading jobs is (at times) time consuming, I add a TILDPostProcessing vertex at the end
    of the child protocol. That makes the methods defined in this parent class redundant.

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
    A serial TILD protocol to compute the free energy change when the system changes from a set of harmonically
        oscillating atoms to a system following the specified potential (for ex. EAM).
    """
    DefaultWhitelist = {
    }

    def __init__(self, **kwargs):
        super(HarmonicTILD, self).__init__(**kwargs)

        id_ = self.input.default
        # Default values
        id_.temperature = 1.
        id_.n_steps = 100
        id_.temperature_damping_timescale = 100.
        id_.overheat_fraction = 2.
        id_.time_step = 1.
        id_.sampling_period = 1
        id_.thermalization_steps = 10
        id_.sleep_time = 0
        id_.custom_lambdas = None
        id_.force_constants = None
        id_.spring_constant = None
        id_.fix_com = True
        id_.use_reflection = True
        # TODO: Need more than input and default, but rather access order, to work without reflection...
        id_.plot = False

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.build_lambdas = BuildMixingPairs()
        g.initialize_jobs = CreateJob()
        g.initial_forces = Zeros()
        g.initial_velocity = SerialList(RandomVelocity)
        g.check_steps = IsGEq()
        g.verlet_positions = SerialList(VerletPositionUpdate)
        g.reflect = SerialList(SphereReflection)
        g.calc_static = SerialList(ExternalHamiltonian)
        g.harmonic = SerialList(HarmonicHamiltonian)
        g.transpose_forces = Transpose()
        g.mix = SerialList(WeightedSum)
        g.verlet_velocities = SerialList(VerletVelocityUpdate)
        g.check_thermalized = IsGEq()
        g.check_sampling_period = ModIsZero()
        g.transpose_energies = Transpose()
        g.addition = SerialList(WeightedSum)
        g.average = SerialList(WelfordOnline)
        g.clock = Counter()
        g.post = TILDPostProcess()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.build_lambdas,
            g.initialize_jobs,
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
            g.transpose_energies,
            g.addition,
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

        # build_lambdas
        g.build_lambdas.input.n_lambdas = ip.n_lambdas
        g.build_lambdas.input.custom_lambdas = ip.custom_lambdas

        # initialize_jobs
        g.initialize_jobs.input.n_images = ip.n_lambdas
        g.initialize_jobs.input.ref_job_full_path = ip.ref_job_full_path
        g.initialize_jobs.input.structure = ip.structure

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # initial_velocity
        g.initial_velocity.input.n_children = ip.n_lambdas
        g.initial_velocity.direct.temperature = ip.temperature
        g.initial_velocity.direct.masses = ip.structure.get_masses
        g.initial_velocity.direct.overheat_fraction = ip.overheat_fraction

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # verlet_positions
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

        # reflect
        g.reflect.input.n_children = ip.n_lambdas
        g.reflect.direct.default.previous_positions = ip.structure.positions
        g.reflect.broadcast.default.previous_velocities = gp.initial_velocity.output.velocities[-1]

        g.reflect.direct.reference_positions = ip.structure.positions
        g.reflect.direct.pbc = ip.structure.pbc
        g.reflect.direct.cell = ip.structure.cell.array
        g.reflect.direct.cutoff_distance = ip.cutoff_distance
        g.reflect.broadcast.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.broadcast.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.broadcast.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.broadcast.previous_velocities = gp.reflect.output.velocities[-1]

        # calc_static
        g.calc_static.input.n_children = ip.n_lambdas
        g.calc_static.direct.project_path = gp.initialize_jobs.output.project_path[-1][-1]
        g.calc_static.direct.structure = ip.structure
        g.calc_static.broadcast.positions = gp.reflect.output.positions[-1]
        g.calc_static.broadcast.job_name = gp.initialize_jobs.output.job_names[-1]

        # harmonic
        g.harmonic.input.n_children = ip.n_lambdas
        g.harmonic.direct.spring_constant = ip.spring_constant
        g.harmonic.direct.force_constants = ip.force_constants
        g.harmonic.direct.home_positions = ip.structure.positions
        g.harmonic.broadcast.positions = gp.reflect.output.positions[-1]
        g.harmonic.direct.cell = ip.structure.cell.array
        g.harmonic.direct.pbc = ip.structure.pbc

        # transpose_forces
        g.transpose_forces.input.matrix = [
            gp.calc_static.output.forces[-1],
            gp.harmonic.output.forces[-1]
        ]

        # mix
        g.mix.input.n_children = ip.n_lambdas
        g.mix.broadcast.vectors = gp.transpose_forces.output.matrix_transpose[-1]
        g.mix.broadcast.weights = gp.build_lambdas.output.lambda_pairs[-1]

        # verlet_velocities
        g.verlet_velocities.input.n_children = ip.n_lambdas
        g.verlet_velocities.broadcast.velocities = gp.reflect.output.velocities[-1]
        g.verlet_velocities.broadcast.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_velocities.direct.masses = ip.structure.get_masses
        g.verlet_velocities.direct.time_step = ip.time_step
        g.verlet_velocities.direct.temperature = ip.temperature
        g.verlet_velocities.direct.temperature_damping_timescale = ip.temperature_damping_timescale
        g.verlet_velocities.direct.time_step = ip.time_step

        # check_thermalized
        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # check_sampling_period
        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = ip.sampling_period

        # transpose_energies
        g.transpose_energies.input.matrix = [
            gp.calc_static.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1]
        ]

        # addition
        g.addition.input.n_children = ip.n_lambdas
        g.addition.broadcast.vectors = gp.transpose_energies.output.matrix_transpose[-1]
        g.addition.direct.weights = [1, -1]

        # average
        g.average.input.n_children = ip.n_lambdas
        g.average.broadcast.sample = gp.addition.output.weighted_sum[-1]

        # post
        g.post.input.lambda_pairs = gp.build_lambdas.output.lambda_pairs[-1]
        g.post.input.n_samples = gp.average.output.n_samples[-1]
        g.post.input.mean = gp.average.output.mean[-1]
        g.post.input.std = gp.average.output.std[-1]
        g.post.input.plot = ip.plot

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
            'free_energy_change': ~gp.post.output.free_energy_change[-1],
            'integrands': ~gp.average.output.mean[-1],
            'integrands_std': ~gp.average.output.std[-1],
            'integrands_n_samples': ~gp.average.output.n_samples[-1],
        }

    def get_classical_harmonic_free_energy(self, temperatures=None):
        """
        Get the total free energy of a harmonic oscillator with this frequency and these atoms. Temperatures are
            clipped at 1 micro-Kelvin.

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
        Get the total free energy of a harmonic oscillator with this frequency and these atoms. Temperatures are
            clipped at 1 micro-Kelvin.

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


class ProtoHarmonicTILD(Protocol, HarmonicTILD, ABC):
    pass


class VacancyTILD(TILDParent):
    """
    A serial TILD protocol to compute the free energy change when the system changes from a regular structure to one
        with a vacancy. This is done by 'decoupling' one of the atoms from the structure and letting it behave as
        a harmonic oscillator, thus creating a pseudo-vacancy. The chemical potential of this harmonically oscillating
        atom is then subtracted from the total free energy change.

    Note: Pressure control is not yet implemented in this protocol.
    """
    DefaultWhitelist = {
    }

    def __init__(self, **kwargs):
        super(VacancyTILD, self).__init__(**kwargs)

        id_ = self.input.default
        id_.vacancy_id = 0
        id_.temperature = 1.
        id_.n_steps = 100
        id_.temperature_damping_timescale = 100.
        id_.overheat_fraction = 2.
        id_.time_step = 1.
        id_.sampling_period = 1
        id_.thermalization_steps = 10
        id_.sleep_time = 0
        id_.custom_lambdas = None
        id_.force_constants = None
        id_.spring_constant = None
        id_.fix_com = True
        id_.use_reflection = True
        # TODO: Need more than input and default, but rather access order, to work without reflection...
        id_.plot = False
        id_.ensure_iterable_mask = True
        id_.accepted = 1

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.create_vacancy = DeleteAtom()
        g.build_lambdas = BuildMixingPairs()
        g.initialize_full_jobs = CreateJob()
        g.initialize_vac_jobs = CreateJob()
        g.random_velocity = SerialList(RandomVelocity)
        g.initial_forces = Zeros()
        g.slice_structure = Slice()
        g.check_steps = IsGEq()
        g.clock = Counter()
        g.verlet_positions = SerialList(VerletPositionUpdate)
        g.reflect = SerialList(SphereReflection)
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
            g.create_vacancy,
            g.build_lambdas,
            g.initialize_full_jobs,
            g.initialize_vac_jobs,
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
        g.starting_vertex = self.graph.create_vacancy
        g.restarting_vertex = self.graph.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # create_vacancy
        g.create_vacancy.input.structure = ip.structure
        g.create_vacancy.input.id = ip.vacancy_id
        shared_ids = gp.create_vacancy.output.mask[-1]

        # build_lambdas
        g.build_lambdas.input.n_lambdas = ip.n_lambdas
        g.build_lambdas.input.custom_lambdas = ip.custom_lambdas

        # initialize_full_jobs
        g.initialize_full_jobs.input.n_images = ip.n_lambdas
        g.initialize_full_jobs.input.ref_job_full_path = ip.ref_job_full_path
        g.initialize_full_jobs.input.structure = ip.structure

        # initialize_vac_jobs
        g.initialize_vac_jobs.input.n_images = ip.n_lambdas
        g.initialize_vac_jobs.input.ref_job_full_path = ip.ref_job_full_path
        g.initialize_vac_jobs.input.structure = gp.create_vacancy.output.structure[-1]

        # random velocity
        g.random_velocity.input.n_children = ip.n_lambdas  # n_children
        g.random_velocity.direct.temperature = ip.temperature
        g.random_velocity.direct.masses = ip.structure.get_masses
        g.random_velocity.direct.overheat_fraction = ip.overheat_fraction

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # slice_Structure
        g.slice_structure.input.vector = ip.structure.positions
        g.slice_structure.input.mask = ip.vacancy_id
        g.slice_structure.input.ensure_iterable_mask = ip.ensure_iterable_mask
        # To keep positions (1,3) instead of (3,)

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # verlet_positions
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

        # reflect
        g.reflect.input.n_children = ip.n_lambdas
        g.reflect.direct.default.previous_positions = ip.structure.positions
        g.reflect.broadcast.default.previous_velocities = gp.random_velocity.output.velocities[-1]

        g.reflect.direct.reference_positions = ip.structure.positions
        g.reflect.direct.pbc = ip.structure.pbc
        g.reflect.direct.cell = ip.structure.cell.array
        g.reflect.direct.cutoff_distance = ip.cutoff_distance
        g.reflect.broadcast.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.broadcast.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.broadcast.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.broadcast.previous_velocities = gp.verlet_velocities.output.velocities[-1]

        # calc_full
        g.calc_full.input.n_children = ip.n_lambdas  # n_children
        g.calc_full.direct.project_path = gp.initialize_full_jobs.output.project_path[-1][-1]
        g.calc_full.broadcast.job_name = gp.initialize_full_jobs.output.job_names[-1]
        g.calc_full.direct.structure = ip.structure
        g.calc_full.broadcast.positions = gp.reflect.output.positions[-1]

        # slice_positions
        g.slice_positions.input.n_children = ip.n_lambdas
        g.slice_positions.broadcast.vector = gp.reflect.output.positions[-1]
        g.slice_positions.direct.mask = shared_ids

        # calc_vac
        g.calc_vac.input.n_children = ip.n_lambdas  # n_children
        g.calc_vac.direct.project_path = gp.initialize_vac_jobs.output.project_path[-1][-1]
        g.calc_vac.broadcast.job_name = gp.initialize_vac_jobs.output.job_names[-1]
        g.calc_vac.direct.structure = gp.create_vacancy.output.structure[-1]
        g.calc_vac.broadcast.positions = gp.slice_positions.output.sliced[-1]

        # slice_harmonic
        g.slice_harmonic.input.n_children = ip.n_lambdas
        g.slice_harmonic.broadcast.vector = gp.reflect.output.positions[-1]
        g.slice_harmonic.direct.mask = ip.vacancy_id
        g.slice_harmonic.direct.ensure_iterable_mask = ip.ensure_iterable_mask

        # harmonic
        g.harmonic.input.n_children = ip.n_lambdas
        g.harmonic.direct.spring_constant = ip.spring_constant
        g.harmonic.direct.force_constants = ip.force_constants
        g.harmonic.direct.home_positions = gp.slice_structure.output.sliced[-1]
        g.harmonic.broadcast.positions = gp.slice_harmonic.output.sliced[-1]
        g.harmonic.direct.cell = ip.structure.cell.array
        g.harmonic.direct.pbc = ip.structure.pbc

        # write_vac_forces
        g.write_vac_forces.input.n_children = ip.n_lambdas
        g.write_vac_forces.broadcast.target = gp.calc_full.output.forces[-1]
        g.write_vac_forces.direct.mask = shared_ids
        g.write_vac_forces.broadcast.new_values = gp.calc_vac.output.forces[-1]

        # write_harmonic_forces
        g.write_harmonic_forces.input.n_children = ip.n_lambdas
        g.write_harmonic_forces.broadcast.target = gp.write_vac_forces.output.overwritten[-1]
        g.write_harmonic_forces.direct.mask = ip.vacancy_id
        g.write_harmonic_forces.broadcast.new_values = gp.harmonic.output.forces[-1]

        # transpose_lambda
        g.transpose_lambda.input.matrix = [
            gp.calc_full.output.forces[-1],
            gp.write_harmonic_forces.output.overwritten[-1]
        ]

        # mix
        g.mix.input.n_children = ip.n_lambdas
        g.mix.broadcast.vectors = gp.transpose_lambda.output.matrix_transpose[-1]
        g.mix.broadcast.weights = gp.build_lambdas.output.lambda_pairs[-1]

        # verlet_velocities
        g.verlet_velocities.input.n_children = ip.n_lambdas
        g.verlet_velocities.broadcast.velocities = gp.reflect.output.velocities[-1]
        g.verlet_velocities.broadcast.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_velocities.direct.masses = ip.structure.get_masses
        g.verlet_velocities.direct.time_step = ip.time_step
        g.verlet_velocities.direct.temperature = ip.temperature
        g.verlet_velocities.direct.temperature_damping_timescale = ip.temperature_damping_timescale

        # check_thermalized
        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # check_sampling_period
        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = ip.sampling_period

        # transpose_energies
        g.transpose_energies.input.matrix = [
            gp.calc_vac.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1],
            gp.calc_full.output.energy_pot[-1]
        ]

        # addition
        g.addition.input.n_children = ip.n_lambdas
        g.addition.broadcast.vectors = gp.transpose_energies.output.matrix_transpose[-1]
        g.addition.direct.weights = [1, 1, -1]

        # average
        g.average.input.n_children = ip.n_lambdas
        g.average.broadcast.sample = gp.addition.output.weighted_sum[-1]

        # post
        g.post.input.lambda_pairs = gp.build_lambdas.output.lambda_pairs[-1]
        g.post.input.n_samples = gp.average.output.n_samples[-1]
        g.post.input.mean = gp.average.output.mean[-1]
        g.post.input.std = gp.average.output.std[-1]
        g.post.input.plot = ip.plot

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'energy_kin': ~gp.verlet_velocities.output.energy_kin[-1],
            'positions': ~gp.reflect.output.positions[-1],
            'velocities': ~gp.verlet_velocities.output.velocities[-1],
            'forces': ~gp.mix.output.weighted_sum[-1],
            'average': ~gp.average.output.mean[-1],
            'free_energy_change': ~gp.post.output.free_energy_change[-1],
            'integrands': ~gp.average.output.mean[-1],
            'integrands_std': ~gp.average.output.std[-1],
            'integrands_n_samples': ~gp.average.output.n_samples[-1],
        }


class ProtoVacancyTILD(Protocol, VacancyTILD, ABC):
    pass


class HarmonicallyCoupled(CompoundVertex):
    """
    A sub-protocol for HarmonicTILDParallel, that is executed in parallel over multiple code using ParallelList.
    """
    DefaultWhitelist = {
    }

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.check_steps = IsGEq()
        g.verlet_positions = VerletPositionUpdate()
        g.calc_static = ExternalHamiltonian()
        g.harmonic = HarmonicHamiltonian()
        g.mix = WeightedSum()
        g.verlet_velocities = VerletVelocityUpdate()
        g.check_thermalized = IsGEq()
        g.check_sampling_period = ModIsZero()
        g.addition = WeightedSum()
        g.average = WelfordOnline()
        g.fep_exp = FEPExponential()
        g.average_fep_exp= WelfordOnline()
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.check_steps, 'false',
            g.clock,
            g.verlet_positions,
            g.calc_static,
            g.harmonic,
            g.mix,
            g.verlet_velocities,
            g.check_thermalized, 'true',
            g.check_sampling_period, 'true',
            g.addition,
            g.average,
            g.fep_exp,
            g.average_fep_exp,
            g.check_steps
        )
        g.make_edge(g.check_thermalized, g.check_steps, 'false')
        g.make_edge(g.check_sampling_period, g.check_steps, 'false')
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

        # verlet_positions
        g.verlet_positions.input.default.positions = ip.structure.positions
        g.verlet_positions.input.default.velocities = ip.velocities
        g.verlet_positions.input.default.forces = ip.forces

        g.verlet_positions.input.positions = gp.verlet_positions.output.positions[-1]
        g.verlet_positions.input.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.input.forces = gp.mix.output.weighted_sum[-1]

        g.verlet_positions.input.masses = ip.structure.get_masses
        g.verlet_positions.input.time_step = ip.time_step
        g.verlet_positions.input.temperature = ip.temperature
        g.verlet_positions.input.temperature_damping_timescale = ip.temperature_damping_timescale

        # calc_static
        g.calc_static.input.project_path = ip.project_path
        g.calc_static.input.job_name = ip.job_name

        g.calc_static.input.structure = ip.structure
        g.calc_static.input.cell = ip.structure.cell.array
        g.calc_static.input.positions = gp.verlet_positions.output.positions[-1]

        # harmonic
        g.harmonic.input.spring_constant = ip.spring_constant
        g.harmonic.input.force_constants = ip.force_constants
        g.harmonic.input.home_positions = ip.structure.positions
        g.harmonic.input.cell = ip.structure.cell.array
        g.harmonic.input.pbc = ip.structure.pbc
        g.harmonic.input.zero_k_energy = ip.zero_k_energy

        g.harmonic.input.positions = gp.verlet_positions.output.positions[-1]

        # mix
        g.mix.input.vectors = [
            gp.calc_static.output.forces[-1],
            gp.harmonic.output.forces[-1]
        ]
        g.mix.input.weights = ip.coupling_weights

        # verlet_velocities
        g.verlet_velocities.input.masses = ip.structure.get_masses
        g.verlet_velocities.input.time_step = ip.time_step
        g.verlet_velocities.input.temperature = ip.temperature
        g.verlet_velocities.input.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_velocities.input.velocities = gp.verlet_positions.output.velocities[-1]
        g.verlet_velocities.input.forces = gp.mix.output.weighted_sum[-1]

        # check_thermalized
        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # check_sampling_period
        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = ip.sampling_period

        # addition
        g.addition.input.vectors = [
            gp.calc_static.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1]
        ]
        g.addition.input.weights = [1, -1]

        # average
        g.average.input.sample = gp.addition.output.weighted_sum[-1]

        # fep_exp
        g.fep_exp.input.u_diff = gp.addition.output.weighted_sum[-1]
        g.fep_exp.input.temperature = ip.temperature
        g.fep_exp.input.delta_lambda = ip.delta_lambdas

        # average_fep_exp
        g.average_fep_exp.input.sample = gp.fep_exp.output.exponential_difference[-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'job_energy_pot': ~gp.calc_static.output.energy_pot[-1],
            'harmonic_energy_pot': ~gp.harmonic.output.energy_pot[-1],
            'energy_kin': ~gp.verlet_velocities.output.energy_kin[-1],
            'positions': ~gp.verlet_positions.output.positions[-1],
            'velocities': ~gp.verlet_velocities.output.velocities[-1],
            'forces': ~gp.mix.output.weighted_sum[-1],
            'harmonic_forces': ~gp.harmonic.output.forces[-1],
            'clock': ~gp.clock.output.n_counts[-1],
            'mean_diff': ~gp.average.output.mean[-1],
            'std_diff': ~gp.average.output.std[-1],
            'fep_exp_mean': ~gp.average_fep_exp.output.mean[-1],
            'fep_exp_std': ~gp.average_fep_exp.output.std[-1],
            'n_samples': ~gp.average.output.n_samples[-1]
        }


class HarmonicTILDParallel(HarmonicTILD):
    """
    A version of HarmonicTILD where the lambda jobs are executed in parallel, thus giving a substantial speed-up.
    """
    DefaultWhitelist = {
    }

    def define_vertices(self):
        # Graph components
        g = self.graph
        ip = Pointer(self.input)
        g.build_lambdas = BuildMixingPairs()
        g.initialize_jobs = CreateJob()
        g.initial_velocities = SerialList(RandomVelocity)
        g.initial_forces = SerialList(Zeros)
        g.run_lambda_points = ParallelList(HarmonicallyCoupled, sleep_time=ip.sleep_time)
        g.clock = Counter()
        g.post = TILDPostProcess()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.build_lambdas,
            g.initialize_jobs,
            g.initial_velocities,
            g.initial_forces,
            g.run_lambda_points,
            g.clock,
            g.post
        )
        g.starting_vertex = g.build_lambdas
        g.restarting_vertex = g.run_lambda_points

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # build_lambdas
        g.build_lambdas.input.n_lambdas = ip.n_lambdas
        g.build_lambdas.input.custom_lambdas = ip.custom_lambdas

        # initialize_integration_points
        g.initialize_jobs.input.n_images = ip.n_lambdas
        g.initialize_jobs.input.ref_job_full_path = ip.ref_job_full_path
        g.initialize_jobs.input.structure = ip.structure

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_lambdas
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # initial_forces
        g.initial_forces.input.n_children = ip.n_lambdas
        g.initial_forces.direct.shape = ip.structure.positions.shape

        # run_lambda_points - initialize
        g.run_lambda_points.input.n_children = ip.n_lambdas

        # run_lambda_points - verlet_positions
        g.run_lambda_points.direct.time_step = ip.time_step
        g.run_lambda_points.direct.temperature = ip.temperature
        g.run_lambda_points.direct.temperature_damping_timescale = ip.temperature_damping_timescale
        g.run_lambda_points.direct.structure = ip.structure

        g.run_lambda_points.broadcast.velocities = gp.initial_velocities.output.velocities[-1]
        g.run_lambda_points.broadcast.forces = gp.initial_forces.output.zeros[-1]

        # run_lambda_points - calc_static
        g.run_lambda_points.broadcast.project_path = gp.initialize_jobs.output.project_path[-1]
        g.run_lambda_points.broadcast.job_name = gp.initialize_jobs.output.job_names[-1]

        # run_lambda_points - harmonic
        g.run_lambda_points.direct.spring_constant = ip.spring_constant
        g.run_lambda_points.direct.force_constants = ip.force_constants
        g.run_lambda_points.direct.zero_k_energy = ip.zero_k_energy

        # run_lambda_points - mix
        g.run_lambda_points.broadcast.coupling_weights = gp.build_lambdas.output.lambda_pairs[-1]

        # run_lambda_points - verlet_velocities
        # takes inputs already specified

        # run_lambda_points - check_thermalized
        g.run_lambda_points.direct.thermalization_steps = ip.thermalization_steps

        # run_lambda_points - check_sampling_period
        g.run_lambda_points.direct.sampling_period = ip.sampling_period

        # run_lambda_points - addition
        g.run_lambda_points.broadcast.delta_lambdas = ip.delta_lambdas

        # run_lambda_points - average
        # no inputs

        # run_lambda_points - clock
        g.run_lambda_points.direct.n_steps = ip.n_steps

        # clock
        g.clock.input.add_counts = gp.run_lambda_points.output.clock[-1][-1]

        # post_processing
        g.post.input.lambda_pairs = gp.build_lambdas.output.lambda_pairs[-1]
        g.post.input.tild_mean = gp.run_lambda_points.output.mean_diff[-1]
        g.post.input.tild_std = gp.run_lambda_points.output.std_diff[-1]
        g.post.input.fep_exp_mean = gp.run_lambda_points.output.fep_exp_mean[-1]
        g.post.input.fep_exp_std = gp.run_lambda_points.output.fep_exp_std[-1]
        g.post.input.temperature = ip.temperature
        g.post.input.n_samples = gp.run_lambda_points.output.n_samples[-1][-1]
        g.post.input.plot = ip.plot

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        o = Pointer(self.graph.run_lambda_points.output)
        return {
            'job_energy_pot': ~o.job_energy_pot[-1],
            'harmonic_energy_pot': ~o.harmonic_energy_pot[-1],
            'energy_kin': ~o.energy_kin[-1],
            'positions': ~o.positions[-1],
            'velocities': ~o.velocities[-1],
            'forces': ~o.forces[-1],
            'harmonic_forces': ~o.harmonic_forces[-1],
            'integrands': ~o.mean_diff[-1],
            'integrands_std': ~o.std_diff[-1],
            'fep_exp_mean': ~o.fep_exp_mean[-1],
            'fep_exp_std': ~o.fep_exp_std[-1],
            'n_samples': ~o.n_samples[-1],
            'tild_free_energy_mean': ~gp.post.output.tild_free_energy_mean[-1],
            'tild_free_energy_std': ~gp.post.output.tild_free_energy_std[-1],
            'tild_free_energy_se': ~gp.post.output.tild_free_energy_se[-1],
            'fep_free_energy_mean': ~gp.post.output.fep_free_energy_mean[-1],
            'fep_free_energy_std': ~gp.post.output.fep_free_energy_std[-1],
            'fep_free_energy_se': ~gp.post.output.fep_free_energy_se[-1]
        }

    def get_integrand(self):
        o = Pointer(self.graph.run_lambda_points.output)
        return ~o.mean[-1], ~o.std[-1] / np.sqrt(~o.n_samples[-1])


class ProtoHarmonicTILDParallel(Protocol, HarmonicTILDParallel, ABC):
    pass


class Decoupling(CompoundVertex):
    """
    A sub-protocol for VacancyTILDParallel that is executed in parallel using ParallelList.
    """
    DefaultWhitelist = {
    }

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.check_steps = IsGEq()
        g.verlet_positions = VerletPositionUpdate()
        g.reflect = SphereReflection()
        g.calc_full = ExternalHamiltonian()
        g.accept = AcceptanceCriterion()
        g.slice_positions = Slice()
        g.calc_vac = ExternalHamiltonian()
        g.harmonic = HarmonicHamiltonian()
        g.write_vac_forces = Overwrite()
        g.write_harmonic_forces = Overwrite()
        g.mix = WeightedSum()
        g.verlet_velocities = VerletVelocityUpdate()
        g.check_thermalized = IsGEq()
        g.check_sampling_period = ModIsZero()
        g.addition = WeightedSum()
        g.average = WelfordOnline()
        g.fep_exp = FEPExponential()
        g.average_fep_exp = WelfordOnline()
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.check_steps, 'false',
            g.verlet_positions,
            g.reflect,
            g.calc_full,
            g.accept,
            g.slice_positions,
            g.calc_vac,
            g.harmonic,
            g.write_vac_forces,
            g.write_harmonic_forces,
            g.mix,
            g.verlet_velocities,
            g.check_thermalized, 'true',
            g.check_sampling_period, 'true',
            g.addition,
            g.average,
            g.fep_exp,
            g.average_fep_exp,
            g.clock,
            g.check_steps
        )
        g.make_edge(g.check_thermalized, g.clock, 'false')
        g.make_edge(g.check_sampling_period, g.clock, 'false')
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

        # verlet_positions
        g.verlet_positions.input.masses = ip.structure.get_masses
        g.verlet_positions.input.time_step = ip.time_step
        g.verlet_positions.input.temperature = ip.temperature
        g.verlet_positions.input.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_positions.input.default.positions = ip.structure.positions
        g.verlet_positions.input.default.velocities = ip.velocities
        g.verlet_positions.input.default.forces = ip.forces

        g.verlet_positions.input.positions = gp.accept.output.positions[-1]
        g.verlet_positions.input.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.input.forces = gp.mix.output.weighted_sum[-1]

        # reflect
        g.reflect.on = ip.use_reflection
        g.reflect.input.default.previous_positions = ip.structure.positions
        g.reflect.input.default.previous_velocities = ip.velocities

        g.reflect.input.reference_positions = ip.structure.positions
        g.reflect.input.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.input.previous_velocities = gp.verlet_velocities.output.velocities[-1]
        g.reflect.input.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.input.velocities = gp.verlet_positions.output.velocities[-1]

        g.reflect.input.cell = ip.structure.cell.array
        g.reflect.input.pbc = ip.structure.pbc
        g.reflect.input.cutoff_distance = ip.cutoff_distance

        # calc_full
        g.calc_full.input.project_path = ip.project_path_full
        g.calc_full.input.job_name = ip.full_job_name

        g.calc_full.input.structure = ip.structure
        g.calc_full.input.cell = ip.structure.cell.array
        g.calc_full.input.positions = gp.reflect.output.positions[-1]

        # accept
        g.accept.input.default.previous_u = gp.calc_full.output.energy_pot[-1]
        g.accept.input.default.previous_positions = ip.structure.positions
        g.accept.input.default.previous_velocities = ip.velocities
        g.accept.input.default.previous_forces = ip.forces
        g.accept.input.default.positions = ip.structure.positions
        g.accept.input.default.velocities = ip.velocities
        g.accept.input.default.forces = ip.forces
        g.accept.input.default.accepted = ip.accepted

        g.accept.input.previous_u = gp.accept.output.energy_pot[-1]
        g.accept.input.previous_positions = gp.accept.output.positions[-1]
        g.accept.input.previous_velocities = gp.verlet_velocities.output.velocities[-1]
        g.accept.input.previous_forces = gp.mix.output.weighted_sum[-1]
        g.accept.input.positions = gp.reflect.output.positions[-1]
        g.accept.input.velocities = gp.reflect.output.velocities[-1]
        g.accept.input.forces = gp.calc_full.output.forces[-1]
        g.accept.input.accepted = gp.accept.output.accepted[-1]

        g.accept.input.u = gp.calc_full.output.energy_pot[-1]
        g.accept.input.temperature = ip.temperature

        # slice_positions
        g.slice_positions.input.vector = gp.accept.output.positions[-1]
        g.slice_positions.input.mask = ip.shared_ids

        # calc_vac
        g.calc_vac.input.project_path = ip.project_path_vac
        g.calc_vac.input.job_name = ip.vac_job_name

        g.calc_vac.input.structure = ip.vacancy_structure
        g.calc_vac.input.cell = ip.vacancy_structure.cell.array
        g.calc_vac.input.positions = gp.slice_positions.output.sliced[-1]

        # harmonic
        g.harmonic.input.spring_constant = ip.spring_constant
        g.harmonic.input.force_constants = ip.force_constants
        g.harmonic.input.home_positions = ip.structure.positions
        g.harmonic.input.cell = ip.structure.cell.array
        g.harmonic.input.pbc = ip.structure.pbc
        g.harmonic.input.mask = ip.vacancy_id
        g.harmonic.input.zero_k_energy = ip.zero_k_energy

        g.harmonic.input.positions = gp.accept.output.positions[-1]

        # write_vac_forces
        g.write_vac_forces.input.target = gp.accept.output.forces[-1]
        g.write_vac_forces.input.mask = ip.shared_ids
        g.write_vac_forces.input.new_values = gp.calc_vac.output.forces[-1]

        # write_harmonic_forces
        g.write_harmonic_forces.input.target = gp.write_vac_forces.output.overwritten[-1]
        g.write_harmonic_forces.input.mask = ip.vacancy_id
        g.write_harmonic_forces.input.new_values = gp.harmonic.output.forces[-1]

        # mix
        g.mix.input.vectors = [
            gp.write_harmonic_forces.output.overwritten[-1],
            gp.accept.output.forces[-1]
        ]
        g.mix.input.weights = ip.coupling_weights

        # verlet_velocities
        g.verlet_velocities.input.masses = ip.structure.get_masses
        g.verlet_velocities.input.time_step = ip.time_step
        g.verlet_velocities.input.temperature = ip.temperature
        g.verlet_velocities.input.temperature_damping_timescale = ip.temperature_damping_timescale

        g.verlet_velocities.input.velocities = gp.accept.output.velocities[-1]
        g.verlet_velocities.input.forces = gp.mix.output.weighted_sum[-1]

        # check_thermalized
        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # check_sampling_period
        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = ip.sampling_period

        # addition
        g.addition.input.vectors = [
            gp.calc_vac.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1],
            gp.calc_full.output.energy_pot[-1]
        ]
        g.addition.input.weights = [1, 1, -1]

        # average
        g.average.input.sample = gp.addition.output.weighted_sum[-1]

        # fep_exp
        g.fep_exp.input.u_diff = gp.addition.output.weighted_sum[-1]
        g.fep_exp.input.temperature = ip.temperature
        g.fep_exp.input.delta_lambda = ip.delta_lambdas

        # average_fep_exp
        g.average_fep_exp.input.sample = gp.fep_exp.output.exponential_difference[-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'full_energy_pot': ~gp.accept.output.energy_pot[-1],
            'harmonic_energy_pot': ~gp.harmonic.output.energy_pot[-1],
            'vac_energy_pot': ~gp.calc_vac.output.energy_pot[-1],
            'energy_kin': ~gp.verlet_velocities.output.energy_kin[-1],
            'positions': ~gp.accept.output.positions[-1],
            'velocities': ~gp.verlet_velocities.output.velocities[-1],
            'forces': ~gp.mix.output.weighted_sum[-1],
            'harmonic_forces': ~gp.harmonic.output.forces[-1],
            'clock': ~gp.clock.output.n_counts[-1],
            'mean_diff': ~gp.average.output.mean[-1],
            'std_diff': ~gp.average.output.std[-1],
            'mean_fep_exp': ~gp.average_fep_exp.output.mean[-1],
            'std_fep_exp': ~gp.average_fep_exp.output.std[-1],
            'n_samples': ~gp.average.output.n_samples[-1],
            'acceptance_rate': ~gp.accept.output.acceptance_rate[-1]
        }


class VacancyTILDParallel(VacancyTILD):
    """
    A version of VacancyTILD where the lambda jobs are executed in parallel, thus giving a substantial speed-up.
    Note: Pressure control is implemented in this protocol.
    """
    DefaultWhitelist = {
    }

    def define_vertices(self):
        # Graph components
        g = self.graph
        ip = Pointer(self.input)
        g.create_vacancy = DeleteAtom()
        g.build_lambdas = BuildMixingPairs()
        g.initialize_full_jobs = CreateJob()
        g.initialize_vac_jobs = CreateJob()
        g.initial_velocities = SerialList(RandomVelocity)
        g.initial_forces = SerialList(Zeros)
        g.run_lambda_points = ParallelList(Decoupling, sleep_time=ip.sleep_time)
        g.clock = Counter()
        g.post = TILDPostProcess()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.create_vacancy,
            g.build_lambdas,
            g.initialize_full_jobs,
            g.initialize_vac_jobs,
            g.initial_velocities,
            g.initial_forces,
            g.run_lambda_points,
            g.clock,
            g.post
        )
        g.starting_vertex = g.create_vacancy
        g.restarting_vertex = g.run_lambda_points

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # create_vacancy
        g.create_vacancy.input.structure = ip.structure
        g.create_vacancy.input.id = ip.vacancy_id

        # build_lambdas
        g.build_lambdas.input.n_lambdas = ip.n_lambdas
        g.build_lambdas.input.custom_lambdas = ip.custom_lambdas

        # initialize_full_jobs
        g.initialize_full_jobs.input.n_images = ip.n_lambdas
        g.initialize_full_jobs.input.ref_job_full_path = ip.ref_job_full_path
        g.initialize_full_jobs.input.structure = ip.structure

        # initialize_vac_jobs
        g.initialize_vac_jobs.input.n_images = ip.n_lambdas
        g.initialize_vac_jobs.input.ref_job_full_path = ip.ref_job_full_path
        g.initialize_vac_jobs.input.structure = gp.create_vacancy.output.structure[-1]

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_lambdas
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # initial_forces
        g.initial_forces.input.n_children = ip.n_lambdas
        g.initial_forces.direct.shape = ip.structure.positions.shape

        # run_lambda_points - initialize
        g.run_lambda_points.input.n_children = ip.n_lambdas

        # run_lambda_points - verlet_positions
        g.run_lambda_points.direct.time_step = ip.time_step
        g.run_lambda_points.direct.temperature = ip.temperature
        g.run_lambda_points.direct.temperature_damping_timescale = ip.temperature_damping_timescale
        g.run_lambda_points.direct.structure = ip.structure

        g.run_lambda_points.broadcast.velocities = gp.initial_velocities.output.velocities[-1]
        g.run_lambda_points.broadcast.forces = gp.initial_forces.output.zeros[-1]

        # run_lambda_points - reflect
        g.run_lambda_points.direct.use_reflection = ip.use_reflection
        g.run_lambda_points.direct.cutoff_distance = ip.cutoff_distance

        # run_lambda_points - calc_full
        g.run_lambda_points.broadcast.project_path_full = gp.initialize_full_jobs.output.project_path[-1]
        g.run_lambda_points.broadcast.full_job_name = gp.initialize_full_jobs.output.job_names[-1]

        # run_lambda_points - accepted
        g.run_lambda_points.direct.accepted = ip.accepted

        # run_lambda_points - slice_positions
        g.run_lambda_points.direct.shared_ids = gp.create_vacancy.output.mask[-1]

        # run_lambda_points - calc_vac
        g.run_lambda_points.direct.project_path_vac = gp.initialize_vac_jobs.output.project_path[-1][-1]
        g.run_lambda_points.broadcast.vac_job_name = gp.initialize_vac_jobs.output.job_names[-1]
        g.run_lambda_points.direct.vacancy_structure = gp.create_vacancy.output.structure[-1]

        # run_lambda_points - harmonic
        g.run_lambda_points.direct.spring_constant = ip.spring_constant
        g.run_lambda_points.direct.force_constants = ip.force_constants
        g.run_lambda_points.direct.vacancy_id = ip.vacancy_id
        g.run_lambda_points.direct.zero_k_energy = ip.zero_k_energy

        # run_lambda_points - write_vac_forces -  takes inputs already specified

        # run_lambda_points - write_harmonic_forces -  takes inputs already specified

        # run_lambda_points - mix
        g.run_lambda_points.broadcast.coupling_weights = gp.build_lambdas.output.lambda_pairs[-1]

        # run_lambda_points - verlet_velocities - takes inputs already specified

        # run_lambda_points - check_thermalized
        g.run_lambda_points.direct.thermalization_steps = ip.thermalization_steps

        # run_lambda_points - check_sampling_period
        g.run_lambda_points.direct.sampling_period = ip.sampling_period

        # run_lambda_points - addition
        g.run_lambda_points.broadcast.delta_lambdas = ip.delta_lambdas

        # run_lambda_points - average - does not need inputs

        # run_lambda_points - clock
        g.run_lambda_points.direct.n_steps = ip.n_steps

        # clock
        g.clock.input.add_counts = gp.run_lambda_points.output.clock[-1][-1]

        # post_processing
        g.post.input.lambda_pairs = gp.build_lambdas.output.lambda_pairs[-1]
        g.post.input.tild_mean = gp.run_lambda_points.output.mean_diff[-1]
        g.post.input.tild_std = gp.run_lambda_points.output.std_diff[-1]
        g.post.input.fep_exp_mean = gp.run_lambda_points.output.mean_fep_exp[-1]
        g.post.input.fep_exp_std = gp.run_lambda_points.output.std_fep_exp[-1]
        g.post.input.temperature = ip.temperature
        g.post.input.n_samples = gp.run_lambda_points.output.n_samples[-1][-1]
        g.post.input.plot = ip.plot

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        o = Pointer(self.graph.run_lambda_points.output)
        return {
            'full_energy_pot': ~o.full_energy_pot[-1],
            'harmonic_energy_pot': ~o.harmonic_energy_pot[-1],
            'vac_energy_pot': ~o.vac_energy_pot[-1],
            'energy_kin': ~o.energy_kin[-1],
            'positions': ~o.positions[-1],
            'velocities': ~o.velocities[-1],
            'forces': ~o.forces[-1],
            'harmonic_forces': ~o.harmonic_forces[-1],
            'integrands': ~o.mean_diff[-1],
            'integrands_std': ~o.std_diff[-1],
            'fep_exp_mean': ~o.mean_fep_exp[-1],
            'fep_exp_std': ~o.std_fep_exp[-1],
            'n_samples': ~o.n_samples[-1],
            'acceptance_rate': ~o.acceptance_rate[-1],
            'tild_free_energy_mean': ~gp.post.output.tild_free_energy_mean[-1],
            'tild_free_energy_std': ~gp.post.output.tild_free_energy_std[-1],
            'tild_free_energy_se': ~gp.post.output.tild_free_energy_se[-1],
            'fep_free_energy_mean': ~gp.post.output.fep_free_energy_mean[-1],
            'fep_free_energy_std': ~gp.post.output.fep_free_energy_std[-1],
            'fep_free_energy_se': ~gp.post.output.fep_free_energy_se[-1]
        }

    def get_integrand(self):
        o = Pointer(self.graph.run_lambda_points.output)
        return ~o.mean[-1], ~o.std[-1] / np.sqrt(~o.n_samples[-1])


class ProtoVacancyTILDParallel(Protocol, VacancyTILDParallel, ABC):
    pass
