# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import physical_constants

from pyiron_contrib.protocol.generic import CompoundVertex, Protocol
from pyiron_contrib.protocol.list import SerialList, ParallelList
from pyiron_contrib.protocol.utils import Pointer
from pyiron_contrib.protocol.primitive.one_state import (
    BuildMixingPairs,
    ComputeFormationEnergy,
    Counter,
    CreateJob,
    CutoffDistance,
    DeleteAtom,
    ExternalHamiltonian,
    FEPExponential,
    HarmonicHamiltonian,
    Overwrite,
    RemoveJob,
    RandomVelocity,
    Slice,
    SphereReflection,
    TILDPostProcess,
    Transpose,
    VerletPositionUpdate,
    VerletVelocityUpdate,
    WeightedSum,
    WelfordOnline,
    Zeros,
)
from pyiron_contrib.protocol.primitive.two_state import (
    AnyVertex,
    IsGEq,
    IsLEq,
    ModIsZero,
)

# Define physical constants that will be used in this script
KB = physical_constants["Boltzmann constant in eV/K"][0]
HBAR = physical_constants["Planck constant over 2 pi in eV s"][0]
ROOT_EV_PER_ANGSTROM_SQUARE_PER_AMU_IN_S = 9.82269385e13
# https://www.wolframalpha.com/input/?i=sqrt((eV)+%2F+((atomic+mass+units)*(angstroms%5E2)))

"""
Protocols for thermodynamic integration using langevin dynamics (TILD).
"""

__author__ = "Liam Huber"
__copyright__ = (
    "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "24 July, 2019"


class _TILDParent(CompoundVertex):
    """
    A parent class for thermodynamic integration by langevin dynamics. Mostly just to avoid duplicate code in
    `HarmonicTILD` and `VacancyTILD`.
    Assumes the presence of `build_lambdas`, `average` (for the thermodynamic average of the integrand), `reflect`
    (to keep each atom closest to its own lattice site), and `mix` (to combine the forces from different
    representations).
    WARNING: The methods in this parent class require loading of the finished interactive jobs that run within the
    child protocol. Since reloading jobs is (at times) time consuming, I add a TILDPostProcessing vertex at the end
    of the child protocol. That makes the methods defined in this parent class redundant. -Raynol
    """

    def get_lambdas(self):
        """
        Get the lambda values.
        """
        return self.graph.build_lambdas.output.lambda_pairs[-1][:, 0]

    def get_tild_integrands(self):
        """
        Get the integrand values from the TILD run.
        """
        integrand = self.graph.average_tild.output
        return np.array(integrand.mean[-1]), integrand.std[-1] / np.sqrt(
            integrand.n_samples[-1]
        )

    def plot_tild_integrands(self):
        """
        Plot the integrand values with their standard errors against the lambda values.
        """
        fig, ax = plt.subplots()
        lambdas = self.get_lambdas()
        thermal_average, standard_error = self.get_tild_integrands()
        ax.plot(lambdas, thermal_average, marker="o")
        ax.fill_between(
            lambdas,
            thermal_average - standard_error,
            thermal_average + standard_error,
            alpha=0.3,
        )
        ax.set_xlabel("Lambda")
        ax.set_ylabel("dF/dLambda")
        return fig, ax

    def get_tild_free_energy_change(self):
        return np.trapz(x=self.get_lambdas(), y=self.get_tild_integrands()[0])


class HarmonicTILD(_TILDParent):
    """
    A serial TILD protocol to compute the free energy change when the system changes from a set of harmonically
        oscillating atoms, to a fully interacting system of atoms. The interactions are described by an
        interatomic potential, for example, an EAM potential.
    NOTE: 1. This protocol is as of now untested with DFT pseudopotentials, and only works for sure, with LAMMPS-
        based potentials.
          2. Convergence criterion is NOT implemented for this protocol, because it runs serially (and would take
        a VERY long time to achieve a good convergence.
    Input attributes:
        ref_job_full_path (str): Path to the pyiron job to use for evaluating forces and energies.
        structure (Atoms): The structure evolve.
        temperature (float): Temperature to run at in K.
        n_steps (int): How many MD steps to run for. (Default is 100.)
        temperature_damping_timescale (float): Langevin thermostat timescale in fs. (Default is None, which runs NVE.)
        time_step (float): MD time step in fs. (Default is 1.)
        overheat_fraction (float): The fraction by which to overheat the initial velocities. This can be useful for
            more quickly equilibrating a system whose initial structure is its fully relaxed positions -- in which
            case equipartition of energy tells us that the kinetic energy should be initialized to double the
            desired value. (Default is 2.0, assume energy equipartition is a good idea.)
        sampling_period (int): Account output every `sampling_period' for the TI operations. (Default is 1, account
            for every MD step.
        thermalization_steps (int): Number of steps the system is thermalized for to reach equilibrium. (Default is
            10 steps.)
        n_lambdas (int): How many mixing pairs to create. (Default is 5.)
        custom_lambdas (list): Specify the set of lambda values as input. (Default is None.)
        spring_constant (float): A single spring / force constant that is used to compute the restoring forces
            on each atom, thus treating every atom as an independent harmonic oscillator (Einstein atom).
            (Default is None.)
        force_constants (NxNx3x3 matrix): The Hessian matrix, obtained from, for ex. Phonopy. (Default is None, treat
            the atoms as independent harmonic oscillators (Einstein atoms.).)
        cutoff_factor (float): The cutoff is obtained by taking the first nearest neighbor distance and multiplying
            it by the cutoff factor. A default value of 0.45 is chosen, because taking a cutoff factor of ~0.5
            sometimes let certain reflections off the hook, and we do not want that to happen. (Default is 0.45.)
        use_reflection (boolean): Turn on or off `SphereReflection` (Default is True.)
        eq_energy (float): The minimized potential energy of the static (expanded) structure. (Default is None.)
    Output attributes:
        total_steps (list): The total number of steps for each integration point, up to convergence, or max steps.
        temperature_mean (list): Mean output temperature for each integration point.
        temperature_std (list): Standard deviation of the output temperature for each integration point.
        integrands_mean (list): Mean of the integrands from TILD.
        integrands_std (list): Standard deviation of the integrands from TILD.
        integrands_n_samples (list): Number of samples over which the mean and standard deviation are calculated.
        tild_free_energy_mean (float): Mean calculated via thermodynamic integration.
        tild_free_energy_std (float): Standard deviation calculated via thermodynamic integration.
        tild_free_energy_se (float): Standard error calculated via thermodynamic integration.
        fep_free_energy_mean (float): Mean calculated via free energy perturbation.
        fep_free_energy_std (float): Standard deviation calculated via free energy perturbation.
        fep_free_energy_se (float): Standard error calculated via free energy perturbation.
    """

    def __init__(self, **kwargs):
        super(HarmonicTILD, self).__init__(**kwargs)

        id_ = self.input.default
        # Default values
        id_.temperature = 1.0
        id_.n_steps = 100
        id_.temperature_damping_timescale = 100.0
        id_.overheat_fraction = 2.0
        id_.time_step = 1.0
        id_.sampling_period = 1
        id_.thermalization_steps = 10
        id_.n_lambdas = 5
        id_.custom_lambdas = None
        id_.force_constants = None
        id_.spring_constant = None
        id_.cutoff_factor = 0.5
        id_.use_reflection = True
        id_.zero_k_energy = None
        id_._total_steps = 0

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.build_lambdas = BuildMixingPairs()
        g.initialize_jobs = CreateJob()
        g.initial_forces = Zeros()
        g.initial_velocities = SerialList(RandomVelocity)
        g.cutoff = CutoffDistance()
        g.check_steps = IsGEq()
        g.verlet_positions = SerialList(VerletPositionUpdate)
        g.reflect = SerialList(SphereReflection)
        g.calc_static = SerialList(ExternalHamiltonian)
        g.harmonic = SerialList(HarmonicHamiltonian)
        g.transpose_forces = Transpose()
        g.mix = SerialList(WeightedSum)
        g.verlet_velocities = SerialList(VerletVelocityUpdate)
        g.check_thermalized = IsGEq()
        g.average_temp = SerialList(WelfordOnline)
        g.check_sampling_period = ModIsZero()
        g.transpose_energies = Transpose()
        g.addition = SerialList(WeightedSum)
        g.average_tild = SerialList(WelfordOnline)
        g.fep_exp = SerialList(FEPExponential)
        g.average_fep_exp = SerialList(WelfordOnline)
        g.clock = Counter()
        g.post = TILDPostProcess()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.build_lambdas,
            g.initialize_jobs,
            g.initial_forces,
            g.initial_velocities,
            g.cutoff,
            g.check_steps,
            "false",
            g.clock,
            g.verlet_positions,
            g.reflect,
            g.calc_static,
            g.harmonic,
            g.transpose_forces,
            g.mix,
            g.verlet_velocities,
            g.check_thermalized,
            "true",
            g.average_temp,
            g.check_sampling_period,
            "true",
            g.transpose_energies,
            g.addition,
            g.average_tild,
            g.fep_exp,
            g.average_fep_exp,
            g.check_steps,
            "true",
            g.post,
        )
        g.make_edge(g.check_thermalized, g.check_steps, "false")
        g.make_edge(g.check_sampling_period, g.check_steps, "false")
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

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_lambdas
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # cutoff
        g.cutoff.input.structure = ip.structure
        g.cutoff.input.cutoff_factor = ip.cutoff_factor

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # verlet_positions
        g.verlet_positions.input.n_children = ip.n_lambdas
        g.verlet_positions.direct.default.positions = ip.structure.positions
        g.verlet_positions.broadcast.default.velocities = (
            gp.initial_velocities.output.velocities[-1]
        )
        g.verlet_positions.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.verlet_positions.broadcast.positions = gp.reflect.output.positions[-1]
        g.verlet_positions.broadcast.velocities = (
            gp.verlet_velocities.output.velocities[-1]
        )
        g.verlet_positions.broadcast.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_positions.direct.masses = ip.structure.get_masses
        g.verlet_positions.direct.time_step = ip.time_step
        g.verlet_positions.direct.temperature = ip.temperature
        g.verlet_positions.direct.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )

        # reflect
        g.reflect.input.n_children = ip.n_lambdas
        g.reflect.direct.default.previous_positions = ip.structure.positions
        g.reflect.broadcast.default.previous_velocities = (
            gp.initial_velocities.output.velocities[-1]
        )
        g.reflect.direct.default.total_steps = ip._total_steps

        g.reflect.direct.reference_positions = ip.structure.positions
        g.reflect.broadcast.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.broadcast.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.broadcast.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.broadcast.previous_velocities = (
            gp.verlet_velocities.output.velocities[-1]
        )
        g.reflect.direct.structure = ip.structure
        g.reflect.direct.cutoff_distance = gp.cutoff.output.cutoff_distance[-1]
        g.reflect.direct.use_reflection = ip.use_reflection
        g.reflect.broadcast.total_steps = gp.reflect.output.total_steps[-1]

        # calc_static
        g.calc_static.input.n_children = ip.n_lambdas
        g.calc_static.direct.structure = ip.structure
        g.calc_static.broadcast.project_path = gp.initialize_jobs.output.project_path[
            -1
        ]
        g.calc_static.broadcast.job_name = gp.initialize_jobs.output.job_names[-1]
        g.calc_static.broadcast.positions = gp.reflect.output.positions[-1]

        # harmonic
        g.harmonic.input.n_children = ip.n_lambdas
        g.harmonic.direct.spring_constant = ip.spring_constant
        g.harmonic.direct.force_constants = ip.force_constants
        g.harmonic.direct.reference_positions = ip.structure.positions
        g.harmonic.broadcast.positions = gp.reflect.output.positions[-1]
        g.harmonic.direct.structure = ip.structure
        g.harmonic.direct.eq_energy = ip.eq_energy

        # transpose_forces
        g.transpose_forces.input.matrix = [
            gp.calc_static.output.forces[-1],
            gp.harmonic.output.forces[-1],
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
        g.verlet_velocities.direct.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )

        # check_thermalized
        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # average_temp
        g.average_temp.input.n_children = ip.n_lambdas
        g.average_temp.broadcast.sample = (
            gp.verlet_velocities.output.instant_temperature[-1]
        )

        # check_sampling_period
        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = ip.sampling_period

        # transpose_energies
        g.transpose_energies.input.matrix = [
            gp.calc_static.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1],
        ]

        # addition
        g.addition.input.n_children = ip.n_lambdas
        g.addition.broadcast.vectors = gp.transpose_energies.output.matrix_transpose[-1]
        g.addition.direct.weights = [1, -1]

        # average_tild
        g.average_tild.input.n_children = ip.n_lambdas
        g.average_tild.broadcast.sample = gp.addition.output.weighted_sum[-1]

        # fep_exp
        g.fep_exp.input.n_children = ip.n_lambdas
        g.fep_exp.broadcast.u_diff = gp.addition.output.weighted_sum[-1]
        g.fep_exp.broadcast.delta_lambda = gp.build_lambdas.output.delta_lambdas[-1]
        g.fep_exp.direct.temperature = ip.temperature

        # average_fep_exp
        g.average_fep_exp.input.n_children = ip.n_lambdas
        g.average_fep_exp.broadcast.sample = gp.fep_exp.output.exponential_difference[
            -1
        ]

        # post_processing
        g.post.input.lambda_pairs = gp.build_lambdas.output.lambda_pairs[-1]
        g.post.input.tild_mean = gp.average_tild.output.mean[-1]
        g.post.input.tild_std = gp.average_tild.output.std[-1]
        g.post.input.fep_exp_mean = gp.average_fep_exp.output.mean[-1]
        g.post.input.fep_exp_std = gp.average_fep_exp.output.std[-1]
        g.post.input.temperature = ip.temperature
        g.post.input.n_samples = gp.average_tild.output.n_samples[-1][-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            "total_steps": ~gp.reflect.output.total_steps[-1],
            "temperature_mean": ~gp.average_temp.output.mean[-1],
            "temperature_std": ~gp.average_temp.output.std[-1],
            "integrands": ~gp.average_tild.output.mean[-1],
            "integrands_std": ~gp.average_tild.output.std[-1],
            "integrands_n_samples": ~gp.average_tild.output.n_samples[-1],
            "tild_free_energy_mean": ~gp.post.output.tild_free_energy_mean[-1],
            "tild_free_energy_std": ~gp.post.output.tild_free_energy_std[-1],
            "tild_free_energy_se": ~gp.post.output.tild_free_energy_se[-1],
            "fep_free_energy_mean": ~gp.post.output.fep_free_energy_mean[-1],
            "fep_free_energy_std": ~gp.post.output.fep_free_energy_std[-1],
            "fep_free_energy_se": ~gp.post.output.fep_free_energy_se[-1],
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
        beta = 1.0 / (KB * temperatures)

        return (
            -3
            * len(self.input.structure)
            * np.log(np.pi / (self.input.spring_constant * beta))
            / (2 * beta)
        )

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
        beta = 1.0 / (KB * temperatures)
        f = 0
        for m in self.input.structure.get_masses():
            hbar_omega = (
                HBAR
                * np.sqrt(self.input.spring_constant / m)
                * ROOT_EV_PER_ANGSTROM_SQUARE_PER_AMU_IN_S
            )
            f += (3.0 / 2) * hbar_omega + (
                (3.0 / beta) * np.log(1 - np.exp(-beta * hbar_omega))
            )
        return f


class ProtoHarmTILDSer(Protocol, HarmonicTILD):
    pass


class _HarmonicallyCoupled(CompoundVertex):
    """
    A sub-protocol for HarmonicTILDParallel for the evolution of each integration point. This sub-protocol is
        executed in parallel over multiple cores using ParallelList.
    """

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.check_steps = IsGEq()
        g.verlet_positions = VerletPositionUpdate()
        g.reflect = SphereReflection()
        g.calc_static = ExternalHamiltonian()
        g.harmonic = HarmonicHamiltonian()
        g.mix = WeightedSum()
        g.verlet_velocities = VerletVelocityUpdate()
        g.check_thermalized = IsGEq()
        g.average_temp = WelfordOnline()
        g.check_sampling_period = ModIsZero()
        g.addition = WeightedSum()
        g.average_tild = WelfordOnline()
        g.fep_exp = FEPExponential()
        g.average_fep_exp = WelfordOnline()
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.check_steps,
            "false",
            g.clock,
            g.verlet_positions,
            g.reflect,
            g.calc_static,
            g.harmonic,
            g.mix,
            g.verlet_velocities,
            g.check_thermalized,
            "true",
            g.average_temp,
            g.check_sampling_period,
            "true",
            g.addition,
            g.average_tild,
            g.fep_exp,
            g.average_fep_exp,
            g.check_steps,
        )
        g.make_edge(g.check_thermalized, g.check_steps, "false")
        g.make_edge(g.check_sampling_period, g.check_steps, "false")
        g.starting_vertex = g.check_steps
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_sub_steps

        # verlet_positions
        g.verlet_positions.input.default.positions = ip.positions
        g.verlet_positions.input.default.velocities = ip.velocities
        g.verlet_positions.input.default.forces = ip.forces

        g.verlet_positions.input.positions = gp.reflect.output.positions[-1]
        g.verlet_positions.input.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.input.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_positions.input.masses = ip.structure.get_masses
        g.verlet_positions.input.time_step = ip.time_step
        g.verlet_positions.input.temperature = ip.temperature
        g.verlet_positions.input.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )

        # reflect
        g.reflect.input.default.previous_positions = ip.positions
        g.reflect.input.default.previous_velocities = ip.velocities
        g.reflect.input.default.total_steps = ip.total_steps

        g.reflect.input.reference_positions = ip.structure.positions
        g.reflect.input.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.input.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.input.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.input.previous_velocities = gp.verlet_velocities.output.velocities[-1]
        g.reflect.input.structure = ip.structure
        g.reflect.input.cutoff_distance = ip.cutoff_distance
        g.reflect.input.use_reflection = ip.use_reflection
        g.reflect.input.total_steps = gp.reflect.output.total_steps[-1]

        # calc_static
        g.calc_static.input.structure = ip.structure
        g.calc_static.input.project_path = ip.project_path
        g.calc_static.input.job_name = ip.job_name
        g.calc_static.input.positions = gp.reflect.output.positions[-1]

        # harmonic
        g.harmonic.input.spring_constant = ip.spring_constant
        g.harmonic.input.force_constants = ip.force_constants
        g.harmonic.input.reference_positions = ip.structure.positions
        g.harmonic.input.positions = gp.reflect.output.positions[-1]
        g.harmonic.input.structure = ip.structure
        g.harmonic.input.eq_energy = ip.eq_energy

        # mix
        g.mix.input.vectors = [
            gp.calc_static.output.forces[-1],
            gp.harmonic.output.forces[-1],
        ]
        g.mix.input.weights = ip.coupling_weights

        # verlet_velocities
        g.verlet_velocities.input.velocities = gp.reflect.output.velocities[-1]
        g.verlet_velocities.input.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_velocities.input.masses = ip.structure.get_masses
        g.verlet_velocities.input.time_step = ip.time_step
        g.verlet_velocities.input.temperature = ip.temperature
        g.verlet_velocities.input.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )

        # check_thermalized
        g.check_thermalized.input.default.target = gp.reflect.output.total_steps[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # average_temp
        g.average_temp.input.default.mean = ip.average_temp_mean
        g.average_temp.input.default.std = ip.average_temp_std
        g.average_temp.input.default.n_samples = ip.average_temp_n_samples
        g.average_temp.input.mean = gp.average_temp.output.mean[-1]
        g.average_temp.input.std = gp.average_temp.output.std[-1]
        g.average_temp.input.n_samples = gp.average_temp.output.n_samples[-1]
        g.average_temp.input.sample = gp.verlet_velocities.output.instant_temperature[
            -1
        ]

        # check_sampling_period
        g.check_sampling_period.input.target = gp.reflect.output.total_steps[-1]
        g.check_sampling_period.input.default.mod = ip.sampling_period

        # addition
        g.addition.input.vectors = [
            gp.calc_static.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1],
        ]
        g.addition.input.weights = [1, -1]

        # average_tild
        g.average_tild.input.default.mean = ip.average_tild_mean
        g.average_tild.input.default.std = ip.average_tild_std
        g.average_tild.input.default.n_samples = ip.average_tild_n_samples
        g.average_tild.input.mean = gp.average_tild.output.mean[-1]
        g.average_tild.input.std = gp.average_tild.output.std[-1]
        g.average_tild.input.n_samples = gp.average_tild.output.n_samples[-1]
        g.average_tild.input.sample = gp.addition.output.weighted_sum[-1]

        # fep_exp
        g.fep_exp.input.u_diff = gp.addition.output.weighted_sum[-1]
        g.fep_exp.input.temperature = ip.temperature
        g.fep_exp.input.delta_lambda = ip.delta_lambdas

        # average_fep_exp
        g.average_fep_exp.input.default.mean = ip.average_fep_exp_mean
        g.average_fep_exp.input.default.std = ip.average_fep_exp_std
        g.average_fep_exp.input.default.n_samples = ip.average_fep_exp_n_samples
        g.average_fep_exp.input.mean = gp.average_fep_exp.output.mean[-1]
        g.average_fep_exp.input.std = gp.average_fep_exp.output.std[-1]
        g.average_fep_exp.input.n_samples = gp.average_fep_exp.output.n_samples[-1]
        g.average_fep_exp.input.sample = gp.fep_exp.output.exponential_difference[-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            "temperature_mean": ~gp.average_temp.output.mean[-1],
            "temperature_std": ~gp.average_temp.output.std[-1],
            "temperature_n_samples": ~gp.average_temp.output.n_samples[-1],
            "positions": ~gp.reflect.output.positions[-1],
            "velocities": ~gp.verlet_velocities.output.velocities[-1],
            "forces": ~gp.mix.output.weighted_sum[-1],
            "total_steps": ~gp.reflect.output.total_steps[-1],
            "mean_diff": ~gp.average_tild.output.mean[-1],
            "std_diff": ~gp.average_tild.output.std[-1],
            "fep_exp_mean": ~gp.average_fep_exp.output.mean[-1],
            "fep_exp_std": ~gp.average_fep_exp.output.std[-1],
            "n_samples": ~gp.average_tild.output.n_samples[-1],
        }


class HarmonicTILDParallel(HarmonicTILD):
    """
    A version of HarmonicTILD where the evolution of each integration point is executed in parallel, thus giving a
        substantial speed-up. A free energy perturbation standard error convergence exit criterion can be applied,
        that is unavailable in the serial version of the HarmonicTILD protocol.
        Maximum efficiency for parallelization can be achieved by setting the number of cores the job can use to
        the number of lambdas, ie., cores / lambdas = 1. Setting the number of cores greater than the number of
        lambdas gives zero gain, and is wasteful if cores % lambdas != 0.
    Input attributes:
        sleep_time (float): A delay in seconds for database access of results. For sqlite, a non-zero delay maybe
            required. (Default is 0 seconds, no delay.)
        convergence_check_steps (int): Check for convergence once every `convergence_check_steps'. (Default is
            once every 10 steps.)
        default_free_energy_se (float): Initialize default free energy standard error to pass into the child
            protocol. (Default is None.)
        fe_tol (float): The free energy standard error tolerance. This is the convergence criterion in eV. (Default
            is 0.01 eV)
    Output attributes:
    For inherited input and output attributes, refer the `HarmonicTILD` protocol.
    """

    def __init__(self, **kwargs):
        super(HarmonicTILDParallel, self).__init__(**kwargs)

        id_ = self.input.default
        # Default values
        id_.sleep_time = 0
        id_.convergence_check_steps = 10
        id_.default_free_energy_se = 1
        id_.fe_tol = 0.01
        id_._project_path = None
        id_._job_name = None
        id_._mean = None
        id_._std = None
        id_._n_samples = None

    def define_vertices(self):
        # Graph components
        g = self.graph
        ip = Pointer(self.input)
        g.build_lambdas = BuildMixingPairs()
        g.initial_forces = Zeros()
        g.initial_velocities = SerialList(RandomVelocity)
        g.cutoff = CutoffDistance()
        g.check_steps = IsGEq()
        g.check_convergence = IsLEq()
        g.remove_jobs = RemoveJob()
        g.create_jobs = CreateJob()
        g.run_lambda_points = ParallelList(
            _HarmonicallyCoupled, sleep_time=ip.sleep_time
        )
        g.clock = Counter()
        g.post = TILDPostProcess()
        g.exit = AnyVertex()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.build_lambdas,
            g.initial_forces,
            g.initial_velocities,
            g.cutoff,
            g.check_steps,
            "false",
            g.check_convergence,
            "false",
            g.remove_jobs,
            g.create_jobs,
            g.run_lambda_points,
            g.clock,
            g.post,
            g.exit,
        )
        g.make_edge(g.check_steps, g.exit, "true")
        g.make_edge(g.check_convergence, g.exit, "true")
        g.make_edge(g.exit, g.check_steps, "false")
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

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_lambdas
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # cutoff
        g.cutoff.input.structure = ip.structure
        g.cutoff.input.cutoff_factor = ip.cutoff_factor

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # check_convergence
        g.check_convergence.input.default.target = ip.default_free_energy_se
        g.check_convergence.input.target = gp.post.output.fep_free_energy_se[-1]
        g.check_convergence.input.threshold = ip.fe_tol

        # remove_jobs
        g.remove_jobs.input.default.project_path = ip._project_path
        g.remove_jobs.input.default.job_names = ip._job_name

        g.remove_jobs.input.project_path = gp.create_jobs.output.project_path[-1][-1]
        g.remove_jobs.input.job_names = gp.create_jobs.output.job_names[-1]

        # create_jobs
        g.create_jobs.input.n_images = ip.n_lambdas
        g.create_jobs.input.ref_job_full_path = ip.ref_job_full_path
        g.create_jobs.input.structure = ip.structure

        # run_lambda_points - initialize
        g.run_lambda_points.input.n_children = ip.n_lambdas

        # run_lambda_points - verlet_positions
        g.run_lambda_points.direct.time_step = ip.time_step
        g.run_lambda_points.direct.temperature = ip.temperature
        g.run_lambda_points.direct.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )
        g.run_lambda_points.direct.structure = ip.structure

        g.run_lambda_points.direct.default.positions = ip.structure.positions
        g.run_lambda_points.broadcast.default.velocities = (
            gp.initial_velocities.output.velocities[-1]
        )
        g.run_lambda_points.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.run_lambda_points.broadcast.positions = gp.run_lambda_points.output.positions[
            -1
        ]
        g.run_lambda_points.broadcast.velocities = (
            gp.run_lambda_points.output.velocities[-1]
        )
        g.run_lambda_points.broadcast.forces = gp.run_lambda_points.output.forces[-1]

        # run_lambda_points - reflect
        g.run_lambda_points.direct.default.total_steps = ip._total_steps
        g.run_lambda_points.broadcast.total_steps = (
            gp.run_lambda_points.output.total_steps[-1]
        )
        g.run_lambda_points.direct.cutoff_distance = gp.cutoff.output.cutoff_distance[
            -1
        ]
        g.run_lambda_points.direct.use_reflection = ip.use_reflection

        # run_lambda_points - calc_static
        g.run_lambda_points.broadcast.project_path = gp.create_jobs.output.project_path[
            -1
        ]
        g.run_lambda_points.broadcast.job_name = gp.create_jobs.output.job_names[-1]

        # run_lambda_points - harmonic
        g.run_lambda_points.direct.spring_constant = ip.spring_constant
        g.run_lambda_points.direct.force_constants = ip.force_constants
        g.run_lambda_points.direct.eq_energy = ip.eq_energy

        # run_lambda_points - mix
        g.run_lambda_points.broadcast.coupling_weights = (
            gp.build_lambdas.output.lambda_pairs[-1]
        )

        # run_lambda_points - verlet_velocities
        # takes inputs already specified

        # run_lambda_points - check_thermalized
        g.run_lambda_points.direct.thermalization_steps = ip.thermalization_steps

        # run_lambda_points - check_sampling_period
        g.run_lambda_points.direct.sampling_period = ip.sampling_period

        # run_lambda_points - average_temp
        g.run_lambda_points.direct.default.average_temp_mean = ip._mean
        g.run_lambda_points.direct.default.average_temp_std = ip._std
        g.run_lambda_points.direct.default.average_temp_n_samples = ip._n_samples
        g.run_lambda_points.broadcast.average_temp_mean = (
            gp.run_lambda_points.output.temperature_mean[-1]
        )
        g.run_lambda_points.broadcast.average_temp_std = (
            gp.run_lambda_points.output.temperature_std[-1]
        )
        g.run_lambda_points.broadcast.average_temp_n_samples = (
            gp.run_lambda_points.output.temperature_n_samples[-1]
        )

        # run_lambda_points - addition
        # no parent inputs

        # run_lambda_points - average_tild
        g.run_lambda_points.direct.default.average_tild_mean = ip._mean
        g.run_lambda_points.direct.default.average_tild_std = ip._std
        g.run_lambda_points.direct.default.average_tild_n_samples = ip._n_samples
        g.run_lambda_points.broadcast.average_tild_mean = (
            gp.run_lambda_points.output.mean_diff[-1]
        )
        g.run_lambda_points.broadcast.average_tild_std = (
            gp.run_lambda_points.output.std_diff[-1]
        )
        g.run_lambda_points.broadcast.average_tild_n_samples = (
            gp.run_lambda_points.output.n_samples[-1]
        )

        # run_lambda_points - fep_exp
        g.run_lambda_points.broadcast.delta_lambdas = (
            gp.build_lambdas.output.delta_lambdas[-1]
        )

        # run_lambda_points - average_fep_exp
        g.run_lambda_points.direct.default.average_fep_exp_mean = ip._mean
        g.run_lambda_points.direct.default.average_fep_exp_std = ip._std
        g.run_lambda_points.direct.default.average_fep_exp_n_samples = ip._n_samples
        g.run_lambda_points.broadcast.average_fep_exp_mean = (
            gp.run_lambda_points.output.fep_exp_mean[-1]
        )
        g.run_lambda_points.broadcast.average_fep_exp_std = (
            gp.run_lambda_points.output.fep_exp_std[-1]
        )
        g.run_lambda_points.broadcast.average_fep_exp_n_samples = (
            gp.run_lambda_points.output.n_samples[-1]
        )

        # run_lambda_points - clock
        g.run_lambda_points.direct.n_sub_steps = ip.convergence_check_steps

        # clock
        g.clock.input.add_counts = ip.convergence_check_steps

        # post_processing
        g.post.input.lambda_pairs = gp.build_lambdas.output.lambda_pairs[-1]
        g.post.input.tild_mean = gp.run_lambda_points.output.mean_diff[-1]
        g.post.input.tild_std = gp.run_lambda_points.output.std_diff[-1]
        g.post.input.fep_exp_mean = gp.run_lambda_points.output.fep_exp_mean[-1]
        g.post.input.fep_exp_std = gp.run_lambda_points.output.fep_exp_std[-1]
        g.post.input.temperature = ip.temperature
        g.post.input.n_samples = gp.run_lambda_points.output.n_samples[-1][-1]

        # exit
        g.exit.input.vertices = [gp.check_steps, gp.check_convergence]
        g.exit.input.print_strings = ["Maximum steps reached", "Convergence reached"]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        o = Pointer(self.graph.run_lambda_points.output)
        return {
            "total_steps": ~o.total_steps[-1],
            "temperature_mean": ~o.temperature_mean[-1],
            "temperature_std": ~o.temperature_std[-1],
            "integrands": ~o.mean_diff[-1],
            "integrands_std": ~o.std_diff[-1],
            "integrands_n_samples": ~o.n_samples[-1],
            "tild_free_energy_mean": ~gp.post.output.tild_free_energy_mean[-1],
            "tild_free_energy_std": ~gp.post.output.tild_free_energy_std[-1],
            "tild_free_energy_se": ~gp.post.output.tild_free_energy_se[-1],
            "fep_free_energy_mean": ~gp.post.output.fep_free_energy_mean[-1],
            "fep_free_energy_std": ~gp.post.output.fep_free_energy_std[-1],
            "fep_free_energy_se": ~gp.post.output.fep_free_energy_se[-1],
        }

    def get_tild_integrands(self):
        o = Pointer(self.graph.run_lambda_points.output)
        return np.array(~o.mean_diff[-1]), ~o.std_diff[-1] / np.sqrt(~o.n_samples[-1])


class ProtoHarmTILDPar(Protocol, HarmonicTILDParallel):
    pass


class VacancyTILD(_TILDParent):
    """
    A serial TILD protocol to compute the free energy change when the system changes from a fully interacting
        system of atoms to the same system with a single vacancy. This is done by 'decoupling' one of the atoms of
        the system from the rest of the atoms, and letting it behave as a harmonic oscillator, thus creating a
        pseudo-vacancy. The chemical potential of this harmonically oscillating atom is then subtracted from the
        total free energy change, to give the free energy change between the fully interacting system, and the same
        system with a vacancy.
        NOTE: 1. This protocol is as of now untested with DFT pseudopotentials, and only works for sure, with LAMMPS-
        based potentials.
        2. Convergence criterion is NOT implemented for this protocol, because it runs serially (and would take
        a VERY long time to achieve a good convergence.
    Input attributes:
        ref_job_full_path (str): Path to the pyiron job to use for evaluating forces and energies.
        structure (Atoms): The structure evolve.
        vacancy_id (int): The id of the atom which will be deleted to create a vacancy. (Default is 0, the 0th atom.)
        temperature (float): Temperature to run at in K.
        n_steps (int): How many MD steps to run for. (Default is 100.)
        temperature_damping_timescale (float): Langevin thermostat timescale in fs. (Default is None, which runs NVE.)
        overheat_fraction (float): The fraction by which to overheat the initial velocities. This can be useful for
            more quickly equilibrating a system whose initial structure is its fully relaxed positions -- in which
            case equipartition of energy tells us that the kinetic energy should be initialized to double the
            desired value. (Default is 2.0, assume energy equipartition is a good idea.)
        time_step (float): MD time step in fs. (Default is 1.)
        sampling_period (int): Account output every `sampling_period' for the TI operations. (Default is 1, account
            for every MD step.
        thermalization_steps (int): Number of steps the system is thermalized for to reach equilibrium. (Default is
            10 steps.)
        n_lambdas (int): How many mixing pairs to create. (Default is 5.)
        custom_lambdas (list): Specify the set of lambda values as input. (Default is None.)
        spring_constant (float): A single spring / force constant that is used to compute the restoring forces
            on each atom. (Default is None.)
        force_constants (NxN matrix): The Hessian matrix, obtained from, for ex. Phonopy. (Default is None, treat
            the atoms as independent harmonic oscillators (Einstein atoms.).)
        cutoff_factor (float): The cutoff is obtained by taking the first nearest neighbor distance and multiplying
            it by the cutoff factor. A default value of 0.45 is chosen, because taking a cutoff factor of ~0.5
            sometimes let certain reflections off the hook, and we do not want that to happen. (Default is 0.45.)
        use_reflection (boolean): Turn on or off `SphereReflection` (Default is True.)
    Output attributes:
        total_steps (list): The total number of steps for each integration point, up to convergence, or max steps.
        temperature_mean (list): Mean output temperature for each integration point.
        temperature_std (list): Standard deviation of the output temperature for each integration point.
        integrands_mean (list): Mean of the integrands from TILD.
        integrands_std (list): Standard deviation of the integrands from TILD.
        integrands_n_samples (list): Number of samples over which the mean and standard deviation are calculated.
        tild_free_energy_mean (float): Mean calculated via thermodynamic integration.
        tild_free_energy_std (float): Standard deviation calculated via thermodynamic integration.
        tild_free_energy_se (float): Standard error calculated via thermodynamic integration.
        fep_free_energy_mean (float): Mean calculated via free energy perturbation.
        fep_free_energy_std (float): Standard deviation calculated via free energy perturbation.
        fep_free_energy_se (float): Standard error calculated via free energy perturbation.
    """

    def __init__(self, **kwargs):
        super(VacancyTILD, self).__init__(**kwargs)

        id_ = self.input.default
        id_.vacancy_id = 0
        id_.temperature = 1.0
        id_.n_steps = 100
        id_.temperature_damping_timescale = 100.0
        id_.overheat_fraction = 2.0
        id_.time_step = 1.0
        id_.sampling_period = 1
        id_.thermalization_steps = 10
        id_.n_lambdas = 5
        id_.custom_lambdas = None
        id_.spring_constant = None
        id_.force_constants = None
        id_.cutoff_factor = 0.5
        id_.use_reflection = True
        id_._total_steps = 0

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.create_vacancy = DeleteAtom()
        g.build_lambdas = BuildMixingPairs()
        g.initialize_full_jobs = CreateJob()
        g.initialize_vac_jobs = CreateJob()
        g.initial_forces = Zeros()
        g.initial_velocities = SerialList(RandomVelocity)
        g.cutoff = CutoffDistance()
        g.check_steps = IsGEq()
        g.verlet_positions = SerialList(VerletPositionUpdate)
        g.reflect = SerialList(SphereReflection)
        g.calc_full = SerialList(ExternalHamiltonian)
        g.slice_positions = SerialList(Slice)
        g.calc_vac = SerialList(ExternalHamiltonian)
        g.harmonic = SerialList(HarmonicHamiltonian)
        g.write_vac_forces = SerialList(Overwrite)
        g.write_harmonic_forces = SerialList(Overwrite)
        g.transpose_lambda = Transpose()
        g.mix = SerialList(WeightedSum)
        g.verlet_velocities = SerialList(VerletVelocityUpdate)
        g.check_thermalized = IsGEq()
        g.average_temp = SerialList(WelfordOnline)
        g.check_sampling_period = ModIsZero()
        g.transpose_energies = Transpose()
        g.addition = SerialList(WeightedSum)
        g.average_tild = SerialList(WelfordOnline)
        g.fep_exp = SerialList(FEPExponential)
        g.average_fep_exp = SerialList(WelfordOnline)
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
            g.initial_forces,
            g.initial_velocities,
            g.cutoff,
            g.check_steps,
            "false",
            g.clock,
            g.verlet_positions,
            g.reflect,
            g.calc_full,
            g.slice_positions,
            g.calc_vac,
            g.harmonic,
            g.write_vac_forces,
            g.write_harmonic_forces,
            g.transpose_lambda,
            g.mix,
            g.verlet_velocities,
            g.check_thermalized,
            "true",
            g.average_temp,
            g.check_sampling_period,
            "true",
            g.transpose_energies,
            g.addition,
            g.average_tild,
            g.fep_exp,
            g.average_fep_exp,
            g.check_steps,
            "true",
            g.post,
        )
        g.make_edge(g.check_thermalized, g.check_steps, "false")
        g.make_edge(g.check_sampling_period, g.check_steps, "false")
        g.starting_vertex = self.graph.create_vacancy
        g.restarting_vertex = self.graph.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # create_vacancy
        g.create_vacancy.input.structure = ip.structure
        g.create_vacancy.input.atom_id = ip.vacancy_id

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

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_lambdas
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # cutoff
        g.cutoff.input.structure = ip.structure
        g.cutoff.input.cutoff_factor = ip.cutoff_factor

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # verlet_positions
        g.verlet_positions.input.n_children = ip.n_lambdas
        g.verlet_positions.direct.default.positions = ip.structure.positions
        g.verlet_positions.broadcast.default.velocities = (
            gp.initial_velocities.output.velocities[-1]
        )
        g.verlet_positions.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.verlet_positions.broadcast.positions = gp.reflect.output.positions[-1]
        g.verlet_positions.broadcast.velocities = (
            gp.verlet_velocities.output.velocities[-1]
        )
        g.verlet_positions.broadcast.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_positions.direct.masses = ip.structure.get_masses
        g.verlet_positions.direct.time_step = ip.time_step
        g.verlet_positions.direct.temperature = ip.temperature
        g.verlet_positions.direct.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )

        # reflect
        g.reflect.input.n_children = ip.n_lambdas
        g.reflect.direct.default.previous_positions = ip.structure.positions
        g.reflect.broadcast.default.previous_velocities = (
            gp.initial_velocities.output.velocities[-1]
        )
        g.reflect.direct.default.total_steps = ip._total_steps

        g.reflect.direct.reference_positions = ip.structure.positions
        g.reflect.broadcast.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.broadcast.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.broadcast.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.broadcast.previous_velocities = (
            gp.verlet_velocities.output.velocities[-1]
        )
        g.reflect.direct.structure = ip.structure
        g.reflect.direct.cutoff_distance = gp.cutoff.output.cutoff_distance[-1]
        g.reflect.direct.use_reflection = ip.use_reflection
        g.reflect.broadcast.total_steps = gp.reflect.output.total_steps[-1]

        # calc_full
        g.calc_full.input.n_children = ip.n_lambdas
        g.calc_full.direct.structure = ip.structure
        g.calc_full.broadcast.project_path = (
            gp.initialize_full_jobs.output.project_path[-1]
        )
        g.calc_full.broadcast.job_name = gp.initialize_full_jobs.output.job_names[-1]
        g.calc_full.broadcast.positions = gp.reflect.output.positions[-1]

        # slice_positions
        g.slice_positions.input.n_children = ip.n_lambdas
        g.slice_positions.broadcast.vector = gp.reflect.output.positions[-1]
        g.slice_positions.direct.mask = gp.create_vacancy.output.mask[-1]

        # calc_vac
        g.calc_vac.input.n_children = ip.n_lambdas
        g.calc_vac.broadcast.project_path = gp.initialize_vac_jobs.output.project_path[
            -1
        ]
        g.calc_vac.broadcast.job_name = gp.initialize_vac_jobs.output.job_names[-1]
        g.calc_vac.direct.structure = gp.create_vacancy.output.structure[-1]
        g.calc_vac.broadcast.positions = gp.slice_positions.output.sliced[-1]

        # harmonic
        g.harmonic.input.n_children = ip.n_lambdas
        g.harmonic.direct.spring_constant = ip.spring_constant
        g.harmonic.direct.force_constants = ip.force_constants
        g.harmonic.direct.reference_positions = ip.structure.positions
        g.harmonic.broadcast.positions = gp.reflect.output.positions[-1]
        g.harmonic.direct.structure = ip.structure
        g.harmonic.direct.mask = ip.vacancy_id

        # write_vac_forces
        g.write_vac_forces.input.n_children = ip.n_lambdas
        g.write_vac_forces.broadcast.target = gp.calc_full.output.forces[-1]
        g.write_vac_forces.direct.mask = gp.create_vacancy.output.mask[-1]
        g.write_vac_forces.broadcast.new_values = gp.calc_vac.output.forces[-1]

        # write_harmonic_forces
        g.write_harmonic_forces.input.n_children = ip.n_lambdas
        g.write_harmonic_forces.broadcast.target = (
            gp.write_vac_forces.output.overwritten[-1]
        )
        g.write_harmonic_forces.direct.mask = ip.vacancy_id
        g.write_harmonic_forces.broadcast.new_values = gp.harmonic.output.forces[-1]

        # transpose_lambda
        g.transpose_lambda.input.matrix = [
            gp.write_harmonic_forces.output.overwritten[-1],
            gp.calc_full.output.forces[-1],
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
        g.verlet_velocities.direct.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )

        # check_thermalized
        g.check_thermalized.input.target = gp.clock.output.n_counts[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # average_temp
        g.average_temp.input.n_children = ip.n_lambdas
        g.average_temp.broadcast.sample = (
            gp.verlet_velocities.output.instant_temperature[-1]
        )

        # check_sampling_period
        g.check_sampling_period.input.target = gp.clock.output.n_counts[-1]
        g.check_sampling_period.input.default.mod = ip.sampling_period

        # transpose_energies
        g.transpose_energies.input.matrix = [
            gp.calc_vac.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1],
            gp.calc_full.output.energy_pot[-1],
        ]

        # addition
        g.addition.input.n_children = ip.n_lambdas
        g.addition.broadcast.vectors = gp.transpose_energies.output.matrix_transpose[-1]
        g.addition.direct.weights = [1, 1, -1]

        # average_tild
        g.average_tild.input.n_children = ip.n_lambdas
        g.average_tild.broadcast.sample = gp.addition.output.weighted_sum[-1]

        # fep_exp
        g.fep_exp.input.n_children = ip.n_lambdas
        g.fep_exp.broadcast.u_diff = gp.addition.output.weighted_sum[-1]
        g.fep_exp.broadcast.delta_lambda = gp.build_lambdas.output.delta_lambdas[-1]
        g.fep_exp.direct.temperature = ip.temperature

        # average_fep_exp
        g.average_fep_exp.input.n_children = ip.n_lambdas
        g.average_fep_exp.broadcast.sample = gp.fep_exp.output.exponential_difference[
            -1
        ]

        # post_processing
        g.post.input.lambda_pairs = gp.build_lambdas.output.lambda_pairs[-1]
        g.post.input.tild_mean = gp.average_tild.output.mean[-1]
        g.post.input.tild_std = gp.average_tild.output.std[-1]
        g.post.input.fep_exp_mean = gp.average_fep_exp.output.mean[-1]
        g.post.input.fep_exp_std = gp.average_fep_exp.output.std[-1]
        g.post.input.temperature = ip.temperature
        g.post.input.n_samples = gp.average_tild.output.n_samples[-1][-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            "total_steps": ~gp.reflect.output.total_steps[-1],
            "temperature_mean": ~gp.average_temp.output.mean[-1],
            "temperature_std": ~gp.average_temp.output.std[-1],
            "integrands_mean": ~gp.average_tild.output.mean[-1],
            "integrands_std": ~gp.average_tild.output.std[-1],
            "integrands_n_samples": ~gp.average_tild.output.n_samples[-1],
            "tild_free_energy_mean": ~gp.post.output.tild_free_energy_mean[-1],
            "tild_free_energy_std": ~gp.post.output.tild_free_energy_std[-1],
            "tild_free_energy_se": ~gp.post.output.tild_free_energy_se[-1],
            "fep_free_energy_mean": ~gp.post.output.fep_free_energy_mean[-1],
            "fep_free_energy_std": ~gp.post.output.fep_free_energy_std[-1],
            "fep_free_energy_se": ~gp.post.output.fep_free_energy_se[-1],
        }


class ProtoVacTILDSer(Protocol, VacancyTILD):
    pass


class _Decoupling(CompoundVertex):
    """
    A sub-protocol for VacancyTILDParallel for the evolution of each integration point. This sub-protocol is
        executed in parallel over multiple cores using ParallelList.
    """

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.check_steps = IsGEq()
        g.verlet_positions = VerletPositionUpdate()
        g.reflect = SphereReflection()
        g.calc_full = ExternalHamiltonian()
        g.slice_positions = Slice()
        g.calc_vac = ExternalHamiltonian()
        g.harmonic = HarmonicHamiltonian()
        g.write_vac_forces = Overwrite()
        g.write_harmonic_forces = Overwrite()
        g.mix = WeightedSum()
        g.verlet_velocities = VerletVelocityUpdate()
        g.check_thermalized = IsGEq()
        g.average_temp = WelfordOnline()
        g.check_sampling_period = ModIsZero()
        g.addition = WeightedSum()
        g.average_tild = WelfordOnline()
        g.fep_exp = FEPExponential()
        g.average_fep_exp = WelfordOnline()
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.check_steps,
            "false",
            g.clock,
            g.verlet_positions,
            g.reflect,
            g.calc_full,
            g.slice_positions,
            g.calc_vac,
            g.harmonic,
            g.write_vac_forces,
            g.write_harmonic_forces,
            g.mix,
            g.verlet_velocities,
            g.check_thermalized,
            "true",
            g.average_temp,
            g.check_sampling_period,
            "true",
            g.addition,
            g.average_tild,
            g.fep_exp,
            g.average_fep_exp,
            g.check_steps,
        )
        g.make_edge(g.check_thermalized, g.clock, "false")
        g.make_edge(g.check_sampling_period, g.clock, "false")
        g.starting_vertex = g.check_steps
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_sub_steps

        # verlet_positions
        g.verlet_positions.input.default.positions = ip.positions
        g.verlet_positions.input.default.velocities = ip.velocities
        g.verlet_positions.input.default.forces = ip.forces

        g.verlet_positions.input.positions = gp.reflect.output.positions[-1]
        g.verlet_positions.input.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.input.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_positions.input.masses = ip.structure.get_masses
        g.verlet_positions.input.time_step = ip.time_step
        g.verlet_positions.input.temperature = ip.temperature
        g.verlet_positions.input.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )

        # reflect
        g.reflect.input.default.previous_positions = ip.positions
        g.reflect.input.default.previous_velocities = ip.velocities
        g.reflect.input.default.total_steps = ip.total_steps

        g.reflect.input.reference_positions = ip.structure.positions
        g.reflect.input.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.input.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.input.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.input.previous_velocities = gp.verlet_velocities.output.velocities[-1]
        g.reflect.input.structure = ip.structure
        g.reflect.input.cutoff_distance = ip.cutoff_distance
        g.reflect.input.use_reflection = ip.use_reflection
        g.reflect.input.total_steps = gp.reflect.output.total_steps[-1]

        # calc_full
        g.calc_full.input.structure = ip.structure
        g.calc_full.input.project_path = ip.project_path_full
        g.calc_full.input.job_name = ip.full_job_name
        g.calc_full.input.positions = gp.reflect.output.positions[-1]

        # slice_positions
        g.slice_positions.input.vector = gp.reflect.output.positions[-1]
        g.slice_positions.input.mask = ip.shared_ids

        # calc_vac
        g.calc_vac.input.structure = ip.vacancy_structure
        g.calc_vac.input.project_path = ip.project_path_vac
        g.calc_vac.input.job_name = ip.vac_job_name
        g.calc_vac.input.positions = gp.slice_positions.output.sliced[-1]

        # harmonic
        g.harmonic.input.spring_constant = ip.spring_constant
        g.harmonic.input.force_constants = ip.force_constants
        g.harmonic.input.reference_positions = ip.structure.positions
        g.harmonic.input.positions = gp.reflect.output.positions[-1]
        g.harmonic.input.structure = ip.structure
        g.harmonic.input.mask = ip.vacancy_id

        # write_vac_forces
        g.write_vac_forces.input.target = gp.calc_full.output.forces[-1]
        g.write_vac_forces.input.mask = ip.shared_ids
        g.write_vac_forces.input.new_values = gp.calc_vac.output.forces[-1]

        # write_harmonic_forces
        g.write_harmonic_forces.input.target = gp.write_vac_forces.output.overwritten[
            -1
        ]
        g.write_harmonic_forces.input.mask = ip.vacancy_id
        g.write_harmonic_forces.input.new_values = gp.harmonic.output.forces[-1]

        # mix
        g.mix.input.vectors = [
            gp.write_harmonic_forces.output.overwritten[-1],
            gp.calc_full.output.forces[-1],
        ]
        g.mix.input.weights = ip.coupling_weights

        # verlet_velocities
        g.verlet_velocities.input.velocities = gp.reflect.output.velocities[-1]
        g.verlet_velocities.input.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_velocities.input.masses = ip.structure.get_masses
        g.verlet_velocities.input.time_step = ip.time_step
        g.verlet_velocities.input.temperature = ip.temperature
        g.verlet_velocities.input.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )

        # check_thermalized
        g.check_thermalized.input.target = gp.reflect.output.total_steps[-1]
        g.check_thermalized.input.threshold = ip.thermalization_steps

        # average_temp
        g.average_temp.input.default.mean = ip.average_temp_mean
        g.average_temp.input.default.std = ip.average_temp_std
        g.average_temp.input.default.n_samples = ip.average_temp_n_samples
        g.average_temp.input.mean = gp.average_temp.output.mean[-1]
        g.average_temp.input.std = gp.average_temp.output.std[-1]
        g.average_temp.input.n_samples = gp.average_temp.output.n_samples[-1]
        g.average_temp.input.sample = gp.verlet_velocities.output.instant_temperature[
            -1
        ]

        # check_sampling_period
        g.check_sampling_period.input.target = gp.reflect.output.total_steps[-1]
        g.check_sampling_period.input.default.mod = ip.sampling_period

        # addition
        g.addition.input.vectors = [
            gp.calc_vac.output.energy_pot[-1],
            gp.harmonic.output.energy_pot[-1],
            gp.calc_full.output.energy_pot[-1],
        ]
        g.addition.input.weights = [1, 1, -1]

        # average_tild
        g.average_tild.input.default.mean = ip.average_tild_mean
        g.average_tild.input.default.std = ip.average_tild_std
        g.average_tild.input.default.n_samples = ip.average_tild_n_samples
        g.average_tild.input.mean = gp.average_tild.output.mean[-1]
        g.average_tild.input.std = gp.average_tild.output.std[-1]
        g.average_tild.input.n_samples = gp.average_tild.output.n_samples[-1]
        g.average_tild.input.sample = gp.addition.output.weighted_sum[-1]

        # fep_exp
        g.fep_exp.input.u_diff = gp.addition.output.weighted_sum[-1]
        g.fep_exp.input.temperature = ip.temperature
        g.fep_exp.input.delta_lambda = ip.delta_lambdas

        # average_fep_exp
        g.average_fep_exp.input.default.mean = ip.average_fep_exp_mean
        g.average_fep_exp.input.default.std = ip.average_fep_exp_std
        g.average_fep_exp.input.default.n_samples = ip.average_fep_exp_n_samples
        g.average_fep_exp.input.mean = gp.average_fep_exp.output.mean[-1]
        g.average_fep_exp.input.std = gp.average_fep_exp.output.std[-1]
        g.average_fep_exp.input.n_samples = gp.average_fep_exp.output.n_samples[-1]
        g.average_fep_exp.input.sample = gp.fep_exp.output.exponential_difference[-1]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            "temperature_mean": ~gp.average_temp.output.mean[-1],
            "temperature_std": ~gp.average_temp.output.std[-1],
            "temperature_n_samples": ~gp.average_temp.output.n_samples[-1],
            "positions": ~gp.reflect.output.positions[-1],
            "velocities": ~gp.verlet_velocities.output.velocities[-1],
            "forces": ~gp.mix.output.weighted_sum[-1],
            "total_steps": ~gp.reflect.output.total_steps[-1],
            "mean_diff": ~gp.average_tild.output.mean[-1],
            "std_diff": ~gp.average_tild.output.std[-1],
            "fep_exp_mean": ~gp.average_fep_exp.output.mean[-1],
            "fep_exp_std": ~gp.average_fep_exp.output.std[-1],
            "n_samples": ~gp.average_tild.output.n_samples[-1],
        }


class VacancyTILDParallel(VacancyTILD):
    """
    A version of VacancyTILD where the evolution of each integration point is executed in parallel, thus giving a
        substantial speed-up. A free energy perturbation standard error convergence exit criterion can be applied,
        that is unavailable in the serial version of the VacancyTILD protocol.
        Maximum efficiency for parallelization can be achieved by setting the number of cores the job can use to
        the number of lambdas, ie., cores / lambdas = 1. Setting the number of cores greater than the number of
        lambdas gives zero gain, and is wasteful if cores % lambdas != 0.
    Input attributes:
        sleep_time (float): A delay in seconds for database access of results. For sqlite, a non-zero delay maybe
            required. (Default is 0 seconds, no delay.)
        convergence_check_steps (int): Check for convergence once every `convergence_check_steps'. (Default is
            once every 10 steps.)
        default_free_energy_se (float): Initialize default free energy standard error to pass into the child
            protocol. (Default is None.)
        fe_tol (float): The free energy standard error tolerance. This is the convergence criterion in eV. (Default
            is 0.01 eV)
    Output attributes:
    For inherited input and output attributes, refer the `HarmonicTILD` protocol.
    """

    def __init__(self, **kwargs):
        super(VacancyTILDParallel, self).__init__(**kwargs)

        id_ = self.input.default
        # Default values
        # The remainder of the default values are inherited from HarmonicTILD
        id_.sleep_time = 0
        id_.convergence_check_steps = 10
        id_.default_free_energy_se = 1
        id_.fe_tol = 0.01
        id_._project_path = None
        id_._job_name = None
        id_._mean = None
        id_._std = None
        id_._n_samples = None

    def define_vertices(self):
        # Graph components
        g = self.graph
        ip = Pointer(self.input)
        g.create_vacancy = DeleteAtom()
        g.build_lambdas = BuildMixingPairs()
        g.initial_forces = Zeros()
        g.initial_velocities = SerialList(RandomVelocity)
        g.cutoff = CutoffDistance()
        g.check_steps = IsGEq()
        g.check_convergence = IsLEq()
        g.remove_full_jobs = RemoveJob()
        g.remove_vac_jobs = RemoveJob()
        g.create_full_jobs = CreateJob()
        g.create_vac_jobs = CreateJob()
        g.run_lambda_points = ParallelList(_Decoupling, sleep_time=ip.sleep_time)
        g.clock = Counter()
        g.post = TILDPostProcess()
        g.exit = AnyVertex()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.create_vacancy,
            g.build_lambdas,
            g.initial_forces,
            g.initial_velocities,
            g.cutoff,
            g.check_steps,
            "false",
            g.check_convergence,
            "false",
            g.remove_full_jobs,
            g.remove_vac_jobs,
            g.create_full_jobs,
            g.create_vac_jobs,
            g.run_lambda_points,
            g.clock,
            g.post,
            g.exit,
        )
        g.make_edge(g.check_steps, g.exit, "true")
        g.make_edge(g.check_convergence, g.exit, "true")
        g.make_edge(g.exit, g.check_steps, "false")
        g.starting_vertex = g.create_vacancy
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # create_vacancy
        g.create_vacancy.input.structure = ip.structure
        g.create_vacancy.input.atom_id = ip.vacancy_id

        # build_lambdas
        g.build_lambdas.input.n_lambdas = ip.n_lambdas
        g.build_lambdas.input.custom_lambdas = ip.custom_lambdas

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_lambdas
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # cutoff
        g.cutoff.input.structure = ip.structure
        g.cutoff.input.cutoff_factor = ip.cutoff_factor

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # check_convergence
        g.check_convergence.input.default.target = ip.default_free_energy_se
        g.check_convergence.input.target = gp.post.output.fep_free_energy_se[-1]
        g.check_convergence.input.threshold = ip.fe_tol

        # remove_full_jobs
        g.remove_full_jobs.input.default.project_path = ip._project_path
        g.remove_full_jobs.input.default.job_names = ip._job_name

        g.remove_full_jobs.input.project_path = gp.create_full_jobs.output.project_path[
            -1
        ][-1]
        g.remove_full_jobs.input.job_names = gp.create_full_jobs.output.job_names[-1]

        # remove_vac_jobs
        g.remove_vac_jobs.input.default.project_path = ip._project_path
        g.remove_vac_jobs.input.default.job_names = ip._job_name

        g.remove_vac_jobs.input.project_path = gp.create_vac_jobs.output.project_path[
            -1
        ][-1]
        g.remove_vac_jobs.input.job_names = gp.create_vac_jobs.output.job_names[-1]

        # create_full_jobs
        g.create_full_jobs.input.n_images = ip.n_lambdas
        g.create_full_jobs.input.ref_job_full_path = ip.ref_job_full_path
        g.create_full_jobs.input.structure = ip.structure

        # create_vac_jobs
        g.create_vac_jobs.input.n_images = ip.n_lambdas
        g.create_vac_jobs.input.ref_job_full_path = ip.ref_job_full_path
        g.create_vac_jobs.input.structure = gp.create_vacancy.output.structure[-1]

        # run_lambda_points - initialize
        g.run_lambda_points.input.n_children = ip.n_lambdas

        # run_lambda_points - verlet_positions
        g.run_lambda_points.direct.time_step = ip.time_step
        g.run_lambda_points.direct.temperature = ip.temperature
        g.run_lambda_points.direct.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )
        g.run_lambda_points.direct.structure = ip.structure

        g.run_lambda_points.direct.default.positions = ip.structure.positions
        g.run_lambda_points.broadcast.default.velocities = (
            gp.initial_velocities.output.velocities[-1]
        )
        g.run_lambda_points.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.run_lambda_points.broadcast.positions = gp.run_lambda_points.output.positions[
            -1
        ]
        g.run_lambda_points.broadcast.velocities = (
            gp.run_lambda_points.output.velocities[-1]
        )
        g.run_lambda_points.broadcast.forces = gp.run_lambda_points.output.forces[-1]

        # run_lambda_points - reflect
        g.run_lambda_points.direct.default.total_steps = ip._total_steps
        g.run_lambda_points.broadcast.total_steps = (
            gp.run_lambda_points.output.total_steps[-1]
        )
        g.run_lambda_points.direct.cutoff_distance = gp.cutoff.output.cutoff_distance[
            -1
        ]
        g.run_lambda_points.direct.use_reflection = ip.use_reflection

        # run_lambda_points - calc_full
        g.run_lambda_points.broadcast.project_path_full = (
            gp.create_full_jobs.output.project_path[-1]
        )
        g.run_lambda_points.broadcast.full_job_name = (
            gp.create_full_jobs.output.job_names[-1]
        )

        # run_lambda_points - slice_positions
        g.run_lambda_points.direct.shared_ids = gp.create_vacancy.output.mask[-1]

        # run_lambda_points - calc_vac
        g.run_lambda_points.broadcast.project_path_vac = (
            gp.create_vac_jobs.output.project_path[-1]
        )
        g.run_lambda_points.broadcast.vac_job_name = (
            gp.create_vac_jobs.output.job_names[-1]
        )
        g.run_lambda_points.direct.vacancy_structure = (
            gp.create_vacancy.output.structure[-1]
        )

        # run_lambda_points - harmonic
        g.run_lambda_points.direct.spring_constant = ip.spring_constant
        g.run_lambda_points.direct.force_constants = ip.force_constants
        g.run_lambda_points.direct.vacancy_id = ip.vacancy_id

        # run_lambda_points - write_vac_forces -  takes inputs already specified

        # run_lambda_points - write_harmonic_forces -  takes inputs already specified

        # run_lambda_points - mix
        g.run_lambda_points.broadcast.coupling_weights = (
            gp.build_lambdas.output.lambda_pairs[-1]
        )

        # run_lambda_points - verlet_velocities - takes inputs already specified

        # run_lambda_points - check_thermalized
        g.run_lambda_points.direct.thermalization_steps = ip.thermalization_steps

        # run_lambda_points - check_sampling_period
        g.run_lambda_points.direct.sampling_period = ip.sampling_period

        # run_lambda_points - average_temp
        g.run_lambda_points.direct.default.average_temp_mean = ip._mean
        g.run_lambda_points.direct.default.average_temp_std = ip._std
        g.run_lambda_points.direct.default.average_temp_n_samples = ip._n_samples
        g.run_lambda_points.broadcast.average_temp_mean = (
            gp.run_lambda_points.output.temperature_mean[-1]
        )
        g.run_lambda_points.broadcast.average_temp_std = (
            gp.run_lambda_points.output.temperature_std[-1]
        )
        g.run_lambda_points.broadcast.average_temp_n_samples = (
            gp.run_lambda_points.output.temperature_n_samples[-1]
        )

        # run_lambda_points - addition
        # no parent inputs

        # run_lambda_points - average_tild
        g.run_lambda_points.direct.default.average_tild_mean = ip._mean
        g.run_lambda_points.direct.default.average_tild_std = ip._std
        g.run_lambda_points.direct.default.average_tild_n_samples = ip._n_samples
        g.run_lambda_points.broadcast.average_tild_mean = (
            gp.run_lambda_points.output.mean_diff[-1]
        )
        g.run_lambda_points.broadcast.average_tild_std = (
            gp.run_lambda_points.output.std_diff[-1]
        )
        g.run_lambda_points.broadcast.average_tild_n_samples = (
            gp.run_lambda_points.output.n_samples[-1]
        )

        # run_lambda_points - fep_exp
        g.run_lambda_points.broadcast.delta_lambdas = (
            gp.build_lambdas.output.delta_lambdas[-1]
        )

        # run_lambda_points - average_fep_exp
        g.run_lambda_points.direct.default.average_fep_exp_mean = ip._mean
        g.run_lambda_points.direct.default.average_fep_exp_std = ip._std
        g.run_lambda_points.direct.default.average_fep_exp_n_samples = ip._n_samples
        g.run_lambda_points.broadcast.average_fep_exp_mean = (
            gp.run_lambda_points.output.fep_exp_mean[-1]
        )
        g.run_lambda_points.broadcast.average_fep_exp_std = (
            gp.run_lambda_points.output.fep_exp_std[-1]
        )
        g.run_lambda_points.broadcast.average_fep_exp_n_samples = (
            gp.run_lambda_points.output.n_samples[-1]
        )

        # run_lambda_points - clock
        g.run_lambda_points.direct.n_sub_steps = ip.convergence_check_steps

        # clock
        g.clock.input.add_counts = ip.convergence_check_steps

        # post_processing
        g.post.input.lambda_pairs = gp.build_lambdas.output.lambda_pairs[-1]
        g.post.input.tild_mean = gp.run_lambda_points.output.mean_diff[-1]
        g.post.input.tild_std = gp.run_lambda_points.output.std_diff[-1]
        g.post.input.fep_exp_mean = gp.run_lambda_points.output.fep_exp_mean[-1]
        g.post.input.fep_exp_std = gp.run_lambda_points.output.fep_exp_std[-1]
        g.post.input.temperature = ip.temperature
        g.post.input.n_samples = gp.run_lambda_points.output.n_samples[-1][-1]

        # exit
        g.exit.input.vertices = [gp.check_steps, gp.check_convergence]
        g.exit.input.print_strings = ["Maximum steps reached", "Convergence reached"]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        o = Pointer(self.graph.run_lambda_points.output)
        return {
            "total_steps": ~o.total_steps[-1],
            "temperature_mean": ~o.temperature_mean[-1],
            "temperature_std": ~o.temperature_std[-1],
            "integrands": ~o.mean_diff[-1],
            "integrands_std": ~o.std_diff[-1],
            "integrands_n_samples": ~o.n_samples[-1],
            "tild_free_energy_mean": ~gp.post.output.tild_free_energy_mean[-1],
            "tild_free_energy_std": ~gp.post.output.tild_free_energy_std[-1],
            "tild_free_energy_se": ~gp.post.output.tild_free_energy_se[-1],
            "fep_free_energy_mean": ~gp.post.output.fep_free_energy_mean[-1],
            "fep_free_energy_std": ~gp.post.output.fep_free_energy_std[-1],
            "fep_free_energy_se": ~gp.post.output.fep_free_energy_se[-1],
        }

    def get_tild_integrands(self):
        """
        Get the integrand values from the TILD run.
        """
        o = Pointer(self.graph.run_lambda_points.output)
        return np.array(~o.mean_diff[-1]), ~o.std_diff[-1] / np.sqrt(~o.n_samples[-1])


class ProtoVacTILDPar(Protocol, VacancyTILDParallel):
    pass


class VacancyFormation(VacancyTILDParallel):
    """
    A protocol which combines HarmonicTILD and VacancyTILD to give the Helmholtz free energy of vacancy formation
        directly. The formation energy is computed via thermodynamic integration, as well as free energy
        perturbation. A formation energy standard error convergence criterion can be applied.
    Input attributes:
        fe_tol (float): The formation energy standard error tolerance. This is the convergence criterion in eV.
            The default is set low, in case maximum number of steps need to be run. (Default is 1e-8 eV.)
        force_constants_harm_to_inter (NxN matrix): The Hessian matrix, obtained from, for ex. Phonopy, for use in
            harmonic to interacting TILD. (Default is None, treat the atoms as independent harmonic oscillators
            (Einstein atoms.).) Note, that another input, force_constants also exists. But that is only used in
            interacting to vacancy TILD.
        harmonic_to_interacting_lambdas (list): Specify the set of lambda values as input for harmonic to
            interacting TILD. (Default is None.)
        interacting_to_vacancy_lambdas (list): Specify the set of lambda values as input for interacting to
            vacancy TILD. (Default is None.)
        default_formation_energy_se (float): Initialize default free energy standard error to pass into the child
            protocol. (Default is None.)
    Output attributes:
        formation_energy_tild (float): The Helmholtz free energy of vacancy formation computed from thermodynamic
            integration.
        formation_energy_tild_std (float): The tild standard deviation.
        formation_energy_tild_se (float): The tild standard error of the mean.
        formation_energy_fep (float): The Helmholtz free energy of vacancy formation computed from free energy
            perturbation.
        formation_energy_fep_std (float): The fep standard deviation.
        formation_energy_fep_se (float): The fep standard error of the mean.
    For inherited input and output attributes, refer the `VacancyTILDParallel` protocol.
    """

    def __init__(self, **kwargs):
        super(VacancyFormation, self).__init__(**kwargs)

        id_ = self.input.default
        # Default values
        # The remainder of the default values are inherited from VacancyTILD
        id_.fe_tol = 1e-8
        id_.force_constants_harm_to_inter = None
        id_.harmonic_to_interacting_lambdas = None
        id_.interacting_to_vacancy_lambdas = None
        id_.default_formation_energy_se = 1

    def define_vertices(self):
        # Graph components
        g = self.graph
        ip = Pointer(self.input)
        g.create_vacancy = DeleteAtom()
        g.build_lambdas_harm_to_inter = BuildMixingPairs()
        g.build_lambdas_inter_to_vac = BuildMixingPairs()
        g.initial_forces = Zeros()
        g.initial_velocities = SerialList(RandomVelocity)
        g.cutoff = CutoffDistance()
        g.check_steps = IsGEq()
        g.check_convergence = IsLEq()
        g.remove_jobs_inter = RemoveJob()
        g.remove_jobs_inter_vac = RemoveJob()
        g.remove_jobs_vac = RemoveJob()
        g.create_jobs_inter = CreateJob()
        g.create_jobs_inter_vac = CreateJob()
        g.create_jobs_vac = CreateJob()
        g.run_harm_to_inter = ParallelList(
            _HarmonicallyCoupled, sleep_time=ip.sleep_time
        )
        g.run_inter_to_vac = ParallelList(_Decoupling, sleep_time=ip.sleep_time)
        g.clock = Counter()
        g.post_harm_to_inter = TILDPostProcess()
        g.post_inter_to_vac = TILDPostProcess()
        g.formation_energy_tild = ComputeFormationEnergy()
        g.formation_energy_fep = ComputeFormationEnergy()
        g.exit = AnyVertex()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.create_vacancy,
            g.build_lambdas_harm_to_inter,
            g.build_lambdas_inter_to_vac,
            g.initial_forces,
            g.initial_velocities,
            g.cutoff,
            g.check_steps,
            "false",
            g.check_convergence,
            "false",
            g.remove_jobs_inter,
            g.remove_jobs_inter_vac,
            g.remove_jobs_vac,
            g.create_jobs_inter,
            g.create_jobs_inter_vac,
            g.create_jobs_vac,
            g.run_harm_to_inter,
            g.run_inter_to_vac,
            g.clock,
            g.post_harm_to_inter,
            g.post_inter_to_vac,
            g.formation_energy_tild,
            g.formation_energy_fep,
            g.exit,
        )
        g.make_edge(g.check_steps, g.exit, "true")
        g.make_edge(g.check_convergence, g.exit, "true")
        g.make_edge(g.exit, g.check_steps, "false")
        g.starting_vertex = g.create_vacancy
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # create_vacancy
        g.create_vacancy.input.structure = ip.structure
        g.create_vacancy.input.atom_id = ip.vacancy_id

        # build_lambdas_harm_to_inter
        g.build_lambdas_harm_to_inter.input.n_lambdas = ip.n_lambdas
        g.build_lambdas_harm_to_inter.input.custom_lambdas = (
            ip.harmonic_to_interacting_lambdas
        )

        # build_lambdas_inter_to_vac
        g.build_lambdas_inter_to_vac.input.n_lambdas = ip.n_lambdas
        g.build_lambdas_inter_to_vac.input.custom_lambdas = (
            ip.interacting_to_vacancy_lambdas
        )

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_lambdas
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = ip.structure.get_masses
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # cutoff
        g.cutoff.input.structure = ip.structure
        g.cutoff.input.cutoff_factor = ip.cutoff_factor

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # check_convergence
        g.check_convergence.input.default.target = ip.default_formation_energy_se
        g.check_convergence.input.target = (
            gp.formation_energy_fep.output.formation_energy_se[-1]
        )
        g.check_convergence.input.threshold = ip.fe_tol

        # remove_jobs_inter
        g.remove_jobs_inter.input.default.project_path = ip._project_path
        g.remove_jobs_inter.input.default.job_names = ip._job_name

        g.remove_jobs_inter.input.project_path = (
            gp.create_jobs_inter.output.project_path[-1][-1]
        )
        g.remove_jobs_inter.input.job_names = gp.create_jobs_inter.output.job_names[-1]

        # remove_jobs_inter_vac
        g.remove_jobs_inter_vac.input.default.project_path = ip._project_path
        g.remove_jobs_inter_vac.input.default.job_names = ip._job_name

        g.remove_jobs_inter_vac.input.project_path = (
            gp.create_jobs_inter_vac.output.project_path[-1][-1]
        )
        g.remove_jobs_inter_vac.input.job_names = (
            gp.create_jobs_inter_vac.output.job_names[-1]
        )

        # remove_jobs_vac
        g.remove_jobs_vac.input.default.project_path = ip._project_path
        g.remove_jobs_vac.input.default.job_names = ip._job_name

        g.remove_jobs_vac.input.project_path = gp.create_jobs_vac.output.project_path[
            -1
        ][-1]
        g.remove_jobs_vac.input.job_names = gp.create_jobs_vac.output.job_names[-1]

        # create_jobs_inter
        g.create_jobs_inter.input.n_images = ip.n_lambdas
        g.create_jobs_inter.input.ref_job_full_path = ip.ref_job_full_path
        g.create_jobs_inter.input.structure = ip.structure

        # create_jobs_inter_vac
        g.create_jobs_inter_vac.input.n_images = ip.n_lambdas
        g.create_jobs_inter_vac.input.ref_job_full_path = ip.ref_job_full_path
        g.create_jobs_inter_vac.input.structure = ip.structure

        # create_jobs_vac
        g.create_jobs_vac.input.n_images = ip.n_lambdas
        g.create_jobs_vac.input.ref_job_full_path = ip.ref_job_full_path
        g.create_jobs_vac.input.structure = gp.create_vacancy.output.structure[-1]

        # run_harm_to_inter - initialize
        g.run_harm_to_inter.input.n_children = ip.n_lambdas

        # run_harm_to_inter - verlet_positions
        g.run_harm_to_inter.direct.time_step = ip.time_step
        g.run_harm_to_inter.direct.temperature = ip.temperature
        g.run_harm_to_inter.direct.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )
        g.run_harm_to_inter.direct.structure = ip.structure

        g.run_harm_to_inter.direct.default.positions = ip.structure.positions
        g.run_harm_to_inter.broadcast.default.velocities = (
            gp.initial_velocities.output.velocities[-1]
        )
        g.run_harm_to_inter.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.run_harm_to_inter.broadcast.positions = gp.run_harm_to_inter.output.positions[
            -1
        ]
        g.run_harm_to_inter.broadcast.velocities = (
            gp.run_harm_to_inter.output.velocities[-1]
        )
        g.run_harm_to_inter.broadcast.forces = gp.run_harm_to_inter.output.forces[-1]

        # run_harm_to_inter - reflect
        g.run_harm_to_inter.direct.default.total_steps = ip._total_steps
        g.run_harm_to_inter.broadcast.total_steps = (
            gp.run_harm_to_inter.output.total_steps[-1]
        )
        g.run_harm_to_inter.direct.cutoff_distance = gp.cutoff.output.cutoff_distance[
            -1
        ]

        # run_harm_to_inter - calc_static
        g.run_harm_to_inter.broadcast.project_path = (
            gp.create_jobs_inter.output.project_path[-1]
        )
        g.run_harm_to_inter.broadcast.job_name = gp.create_jobs_inter.output.job_names[
            -1
        ]

        # run_harm_to_inter - harmonic
        g.run_harm_to_inter.direct.spring_constant = ip.spring_constant
        g.run_harm_to_inter.direct.force_constants = ip.force_constants_harm_to_inter
        g.run_harm_to_inter.direct.eq_energy = ip.eq_energy

        # run_harm_to_inter - mix
        g.run_harm_to_inter.broadcast.coupling_weights = (
            gp.build_lambdas_harm_to_inter.output.lambda_pairs[-1]
        )

        # run_harm_to_inter - verlet_velocities
        # takes inputs already specified

        # run_harm_to_inter - check_thermalized
        g.run_harm_to_inter.direct.thermalization_steps = ip.thermalization_steps

        # run_harm_to_inter - check_sampling_period
        g.run_harm_to_inter.direct.sampling_period = ip.sampling_period

        # run_harm_to_inter - average_temp
        g.run_harm_to_inter.direct.default.average_temp_mean = ip._mean
        g.run_harm_to_inter.direct.default.average_temp_std = ip._std
        g.run_harm_to_inter.direct.default.average_temp_n_samples = ip._n_samples
        g.run_harm_to_inter.broadcast.average_temp_mean = (
            gp.run_harm_to_inter.output.temperature_mean[-1]
        )
        g.run_harm_to_inter.broadcast.average_temp_std = (
            gp.run_harm_to_inter.output.temperature_std[-1]
        )
        g.run_harm_to_inter.broadcast.average_temp_n_samples = (
            gp.run_harm_to_inter.output.temperature_n_samples[-1]
        )

        # run_harm_to_inter - addition
        # no parent inputs

        # run_harm_to_inter - average_tild
        g.run_harm_to_inter.direct.default.average_tild_mean = ip._mean
        g.run_harm_to_inter.direct.default.average_tild_std = ip._std
        g.run_harm_to_inter.direct.default.average_tild_n_samples = ip._n_samples
        g.run_harm_to_inter.broadcast.average_tild_mean = (
            gp.run_harm_to_inter.output.mean_diff[-1]
        )
        g.run_harm_to_inter.broadcast.average_tild_std = (
            gp.run_harm_to_inter.output.std_diff[-1]
        )
        g.run_harm_to_inter.broadcast.average_tild_n_samples = (
            gp.run_harm_to_inter.output.n_samples[-1]
        )

        # run_harm_to_inter - fep_exp
        g.run_harm_to_inter.broadcast.delta_lambdas = (
            gp.build_lambdas_harm_to_inter.output.delta_lambdas[-1]
        )

        # run_harm_to_inter - average_fep_exp
        g.run_harm_to_inter.direct.default.average_fep_exp_mean = ip._mean
        g.run_harm_to_inter.direct.default.average_fep_exp_std = ip._std
        g.run_harm_to_inter.direct.default.average_fep_exp_n_samples = ip._n_samples
        g.run_harm_to_inter.broadcast.average_fep_exp_mean = (
            gp.run_harm_to_inter.output.fep_exp_mean[-1]
        )
        g.run_harm_to_inter.broadcast.average_fep_exp_std = (
            gp.run_harm_to_inter.output.fep_exp_std[-1]
        )
        g.run_harm_to_inter.broadcast.average_fep_exp_n_samples = (
            gp.run_harm_to_inter.output.n_samples[-1]
        )

        # run_harm_to_inter - clock
        g.run_harm_to_inter.direct.n_sub_steps = ip.convergence_check_steps

        # run_inter_to_vac - initialize
        g.run_inter_to_vac.input.n_children = ip.n_lambdas

        # run_inter_to_vac - verlet_positions
        g.run_inter_to_vac.direct.time_step = ip.time_step
        g.run_inter_to_vac.direct.temperature = ip.temperature
        g.run_inter_to_vac.direct.temperature_damping_timescale = (
            ip.temperature_damping_timescale
        )
        g.run_inter_to_vac.direct.structure = ip.structure

        g.run_inter_to_vac.direct.default.positions = ip.structure.positions
        g.run_inter_to_vac.broadcast.default.velocities = (
            gp.initial_velocities.output.velocities[-1]
        )
        g.run_inter_to_vac.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.run_inter_to_vac.broadcast.positions = gp.run_inter_to_vac.output.positions[
            -1
        ]
        g.run_inter_to_vac.broadcast.velocities = gp.run_inter_to_vac.output.velocities[
            -1
        ]
        g.run_inter_to_vac.broadcast.forces = gp.run_inter_to_vac.output.forces[-1]

        # run_inter_to_vac - reflect
        g.run_inter_to_vac.direct.default.total_steps = ip._total_steps
        g.run_inter_to_vac.broadcast.total_steps = (
            gp.run_inter_to_vac.output.total_steps[-1]
        )
        g.run_inter_to_vac.direct.cutoff_distance = gp.cutoff.output.cutoff_distance[-1]

        # run_inter_to_vac - calc_full
        g.run_inter_to_vac.broadcast.project_path_full = (
            gp.create_jobs_inter_vac.output.project_path[-1]
        )
        g.run_inter_to_vac.broadcast.full_job_name = (
            gp.create_jobs_inter_vac.output.job_names[-1]
        )

        # run_inter_to_vac - slice_positions
        g.run_inter_to_vac.direct.shared_ids = gp.create_vacancy.output.mask[-1]

        # run_inter_to_vac - calc_vac
        g.run_inter_to_vac.broadcast.project_path_vac = (
            gp.create_jobs_vac.output.project_path[-1]
        )
        g.run_inter_to_vac.broadcast.vac_job_name = gp.create_jobs_vac.output.job_names[
            -1
        ]
        g.run_inter_to_vac.direct.vacancy_structure = (
            gp.create_vacancy.output.structure[-1]
        )

        # run_inter_to_vac - harmonic
        g.run_inter_to_vac.direct.spring_constant = ip.spring_constant
        g.run_inter_to_vac.direct.force_constants = ip.force_constants
        g.run_inter_to_vac.direct.vacancy_id = ip.vacancy_id

        # run_inter_to_vac - write_vac_forces -  takes inputs already specified

        # run_inter_to_vac - write_harmonic_forces -  takes inputs already specified

        # run_inter_to_vac - mix
        g.run_inter_to_vac.broadcast.coupling_weights = (
            gp.build_lambdas_inter_to_vac.output.lambda_pairs[-1]
        )

        # run_inter_to_vac - verlet_velocities - takes inputs already specified

        # run_inter_to_vac - check_thermalized
        g.run_inter_to_vac.direct.thermalization_steps = ip.thermalization_steps

        # run_inter_to_vac - check_sampling_period
        g.run_inter_to_vac.direct.sampling_period = ip.sampling_period

        # run_inter_to_vac - average_temp
        g.run_inter_to_vac.direct.default.average_temp_mean = ip._mean
        g.run_inter_to_vac.direct.default.average_temp_std = ip._std
        g.run_inter_to_vac.direct.default.average_temp_n_samples = ip._n_samples
        g.run_inter_to_vac.broadcast.average_temp_mean = (
            gp.run_inter_to_vac.output.temperature_mean[-1]
        )
        g.run_inter_to_vac.broadcast.average_temp_std = (
            gp.run_inter_to_vac.output.temperature_std[-1]
        )
        g.run_inter_to_vac.broadcast.average_temp_n_samples = (
            gp.run_inter_to_vac.output.temperature_n_samples[-1]
        )

        # run_inter_to_vac - addition
        # no parent inputs

        # run_inter_to_vac - average_tild
        g.run_inter_to_vac.direct.default.average_tild_mean = ip._mean
        g.run_inter_to_vac.direct.default.average_tild_std = ip._std
        g.run_inter_to_vac.direct.default.average_tild_n_samples = ip._n_samples
        g.run_inter_to_vac.broadcast.average_tild_mean = (
            gp.run_inter_to_vac.output.mean_diff[-1]
        )
        g.run_inter_to_vac.broadcast.average_tild_std = (
            gp.run_inter_to_vac.output.std_diff[-1]
        )
        g.run_inter_to_vac.broadcast.average_tild_n_samples = (
            gp.run_inter_to_vac.output.n_samples[-1]
        )

        # run_inter_to_vac - fep_exp
        g.run_inter_to_vac.broadcast.delta_lambdas = (
            gp.build_lambdas_inter_to_vac.output.delta_lambdas[-1]
        )

        # run_inter_to_vac - average_fep_exp
        g.run_inter_to_vac.direct.default.average_fep_exp_mean = ip._mean
        g.run_inter_to_vac.direct.default.average_fep_exp_std = ip._std
        g.run_inter_to_vac.direct.default.average_fep_exp_n_samples = ip._n_samples
        g.run_inter_to_vac.broadcast.average_fep_exp_mean = (
            gp.run_inter_to_vac.output.fep_exp_mean[-1]
        )
        g.run_inter_to_vac.broadcast.average_fep_exp_std = (
            gp.run_inter_to_vac.output.fep_exp_std[-1]
        )
        g.run_inter_to_vac.broadcast.average_fep_exp_n_samples = (
            gp.run_inter_to_vac.output.n_samples[-1]
        )

        # run_inter_to_vac - clock
        g.run_inter_to_vac.direct.n_sub_steps = ip.convergence_check_steps

        # clock
        g.clock.input.add_counts = ip.convergence_check_steps

        # post_harm_to_inter
        g.post_harm_to_inter.input.lambda_pairs = (
            gp.build_lambdas_harm_to_inter.output.lambda_pairs[-1]
        )
        g.post_harm_to_inter.input.tild_mean = gp.run_harm_to_inter.output.mean_diff[-1]
        g.post_harm_to_inter.input.tild_std = gp.run_harm_to_inter.output.std_diff[-1]
        g.post_harm_to_inter.input.fep_exp_mean = (
            gp.run_harm_to_inter.output.fep_exp_mean[-1]
        )
        g.post_harm_to_inter.input.fep_exp_std = (
            gp.run_harm_to_inter.output.fep_exp_std[-1]
        )
        g.post_harm_to_inter.input.temperature = ip.temperature
        g.post_harm_to_inter.input.n_samples = gp.run_harm_to_inter.output.n_samples[
            -1
        ][-1]

        # post_inter_to_vac
        g.post_inter_to_vac.input.lambda_pairs = (
            gp.build_lambdas_inter_to_vac.output.lambda_pairs[-1]
        )
        g.post_inter_to_vac.input.tild_mean = gp.run_inter_to_vac.output.mean_diff[-1]
        g.post_inter_to_vac.input.tild_std = gp.run_inter_to_vac.output.std_diff[-1]
        g.post_inter_to_vac.input.fep_exp_mean = (
            gp.run_inter_to_vac.output.fep_exp_mean[-1]
        )
        g.post_inter_to_vac.input.fep_exp_std = gp.run_inter_to_vac.output.fep_exp_std[
            -1
        ]
        g.post_inter_to_vac.input.temperature = ip.temperature
        g.post_inter_to_vac.input.n_samples = gp.run_inter_to_vac.output.n_samples[-1][
            -1
        ]

        # formation_energy_tild
        g.formation_energy_tild.input.n_atoms = gp.initial_velocities.output.n_atoms[
            -1
        ][-1]
        g.formation_energy_tild.input.eq_energy = gp.minimize_job.output.energy_pot[-1]
        g.formation_energy_tild.input.harm_to_inter_mean = (
            gp.post_harm_to_inter.output.tild_free_energy_mean[-1]
        )
        g.formation_energy_tild.input.harm_to_inter_std = (
            gp.post_harm_to_inter.output.tild_free_energy_std[-1]
        )
        g.formation_energy_tild.input.harm_to_inter_se = (
            gp.post_harm_to_inter.output.tild_free_energy_se[-1]
        )
        g.formation_energy_tild.input.inter_to_vac_mean = (
            gp.post_inter_to_vac.output.tild_free_energy_mean[-1]
        )
        g.formation_energy_tild.input.inter_to_vac_std = (
            gp.post_inter_to_vac.output.tild_free_energy_std[-1]
        )
        g.formation_energy_tild.input.inter_to_vac_se = (
            gp.post_inter_to_vac.output.tild_free_energy_se[-1]
        )

        # formation_energy_fep
        g.formation_energy_fep.input.n_atoms = gp.initial_velocities.output.n_atoms[-1][
            -1
        ]
        g.formation_energy_fep.input.eq_energy = gp.minimize_job.output.energy_pot[-1]
        g.formation_energy_fep.input.harm_to_inter_mean = (
            gp.post_harm_to_inter.output.fep_free_energy_mean[-1]
        )
        g.formation_energy_fep.input.harm_to_inter_std = (
            gp.post_harm_to_inter.output.fep_free_energy_std[-1]
        )
        g.formation_energy_fep.input.harm_to_inter_se = (
            gp.post_harm_to_inter.output.fep_free_energy_se[-1]
        )
        g.formation_energy_fep.input.inter_to_vac_mean = (
            gp.post_inter_to_vac.output.fep_free_energy_mean[-1]
        )
        g.formation_energy_fep.input.inter_to_vac_std = (
            gp.post_inter_to_vac.output.fep_free_energy_std[-1]
        )
        g.formation_energy_fep.input.inter_to_vac_se = (
            gp.post_inter_to_vac.output.fep_free_energy_se[-1]
        )

        # exit
        g.exit.input.vertices = [gp.check_steps, gp.check_convergence]
        g.exit.input.print_strings = ["Maximum steps reached", "Convergence reached"]

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        o = Pointer(self.graph.run_harm_to_inter.output)
        p = Pointer(self.graph.run_inter_to_vac.output)
        return {
            "total_steps": ~o.total_steps[-1],
            "temperature_mean_harm_to_inter": ~o.temperature_mean[-1],
            "temperature_std_harm_to_inter": ~o.temperature_std[-1],
            "integrands_harm_to_inter": ~o.mean_diff[-1],
            "integrands_std_harm_to_inter": ~o.std_diff[-1],
            "integrands_n_samples_harm_to_inter": ~o.n_samples[-1],
            "temperature_mean_inter_to_vac": ~p.temperature_mean[-1],
            "temperature_std_inter_to_vac": ~p.temperature_std[-1],
            "integrands_inter_to_vac": ~p.mean_diff[-1],
            "integrands_std_inter_to_vac": ~p.std_diff[-1],
            "integrands_n_samples_inter_to_vac": ~p.n_samples[-1],
            "formation_energy_tild": ~gp.formation_energy_tild.output.formation_energy_mean[
                -1
            ],
            "formation_energy_tild_std": ~gp.formation_energy_tild.output.formation_energy_std[
                -1
            ],
            "formation_energy_tild_se": ~gp.formation_energy_tild.output.formation_energy_se[
                -1
            ],
            "formation_energy_fep": ~gp.formation_energy_fep.output.formation_energy_mean[
                -1
            ],
            "formation_energy_fep_std": ~gp.formation_energy_fep.output.formation_energy_std[
                -1
            ],
            "formation_energy_fep_se": ~gp.formation_energy_fep.output.formation_energy_se[
                -1
            ],
        }

    def get_lambdas(self, integrands="harm_to_inter"):
        """
        Get the lambda values.
        """
        if integrands == "harm_to_inter":
            vertex = self.graph.build_lambdas_harm_to_inter.output
        elif integrands == "inter_to_vac":
            vertex = self.graph.build_lambdas_inter_to_vac.output
        else:
            raise KeyError(
                "The value of `integrands` can only be 'harm_to_inter' or 'inter_to_vac'"
            )
        return vertex.lambda_pairs[-1][:, 0]

    def get_tild_integrands(self, integrands="harm_to_inter"):
        """
        Get the integrand values from the TILD run.
        """
        if integrands == "harm_to_inter":
            vertex = self.graph.run_harm_to_inter.output
        elif integrands == "inter_to_vac":
            vertex = self.graph.run_inter_to_vac.output
        else:
            raise KeyError(
                "The value of `integrands` can only be 'harm_to_inter' or 'inter_to_vac'"
            )
        return np.array(vertex.mean_diff[-1]), vertex.std_diff[-1] / np.sqrt(
            vertex.n_samples[-1]
        )


class ProtoVacForm(Protocol, VacancyFormation):
    pass
