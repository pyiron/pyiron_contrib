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
from pyiron_contrib.protocol.primitive.one_state import BuildMixingPairs, Counter, \
    CreateSubJobs, ExternalHamiltonian, FEPExponential, RandomVelocity, SphereReflectionPerAtom, TILDPostProcess, \
    TILDValidate, VerletPositionUpdate, VerletVelocityUpdate, WeightedSum, WelfordOnline, Zeros
from pyiron_contrib.protocol.primitive.two_state import AnyVertex, IsGEq, IsLEq, ModIsZero

# Define physical constants that will be used in this script
KB = physical_constants['Boltzmann constant in eV/K'][0]
HBAR = physical_constants['Planck constant over 2 pi in eV s'][0]
ROOT_EV_PER_ANGSTROM_SQUARE_PER_AMU_IN_S = 9.82269385e13
# https://www.wolframalpha.com/input/?i=sqrt((eV)+%2F+((atomic+mass+units)*(angstroms%5E2)))

"""
Protocols for thermodynamic integration using langevin dynamics (TILD).
"""

__author__ = "Liam Huber, Raynol Dsouza"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "24 July, 2019"


class _TILDLambdaEvolution(CompoundVertex):
    """
    A sub-protocol for TILDParallel for the evolution of each 'lambda'/integration point. This sub-protocol can be
        executed in serial using SerialList, and in parallel over multiple cores using ParallelList.
    """

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.check_steps = IsGEq()
        g.verlet_positions = VerletPositionUpdate()
        g.reflect = SphereReflectionPerAtom()
        g.calc_static_a = ExternalHamiltonian()
        g.calc_static_b = ExternalHamiltonian()
        g.mix = WeightedSum()
        g.verlet_velocities = VerletVelocityUpdate()
        g.check_thermalized = IsGEq()
        g.average_temp = WelfordOnline()
        g.check_sampling_steps = ModIsZero()
        g.addition = WeightedSum()
        g.average_tild = WelfordOnline()
        g.fep_exp = FEPExponential()
        g.average_fep_exp = WelfordOnline()
        g.clock = Counter()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.check_steps, "false",
            g.clock,
            g.verlet_positions,
            g.reflect,
            g.calc_static_a,
            g.calc_static_b,
            g.mix,
            g.verlet_velocities,
            g.check_thermalized, "true",
            g.average_temp,
            g.check_sampling_steps, "true",
            g.addition,
            g.average_tild,
            g.fep_exp,
            g.average_fep_exp,
            g.check_steps
        )
        g.make_edge(g.check_thermalized, g.check_steps, "false")
        g.make_edge(g.check_sampling_steps, g.check_steps, "false")
        g.starting_vertex = g.check_steps
        g.restarting_vertex = g.clock

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
        g.verlet_positions.input.masses = ip.masses
        g.verlet_positions.input.time_step = ip.time_step
        g.verlet_positions.input.temperature = ip.temperature
        g.verlet_positions.input.temperature_damping_timescale = ip.temperature_damping_timescale

        # reflect
        g.reflect.input.use_reflection = ip.use_reflection
        g.reflect.input.default.previous_positions = ip.positions
        g.reflect.input.default.previous_velocities = ip.velocities
        g.reflect.input.default.total_steps = ip.total_steps
        g.reflect.input.default.cutoff_distance = ip.cutoff_distance

        g.reflect.input.reference_positions = ip.structure_a.positions
        g.reflect.input.positions = gp.verlet_positions.output.positions[-1]
        g.reflect.input.velocities = gp.verlet_positions.output.velocities[-1]
        g.reflect.input.previous_positions = gp.reflect.output.positions[-1]
        g.reflect.input.previous_velocities = gp.verlet_velocities.output.velocities[-1]
        g.reflect.input.structure = ip.structure_a
        g.reflect.input.cutoff_factor = ip.cutoff_factor
        g.reflect.input.total_steps = gp.reflect.output.total_steps[-1]
        g.reflect.input.cutoff_distance = gp.reflect.output.cutoff_distance[-1]

        # calc_static_a
        g.calc_static_a.input.job_project_path = ip.job_project_path_a
        g.calc_static_a.input.job_name = ip.job_name_a
        g.calc_static_a.input.positions = gp.reflect.output.positions[-1]
        g.calc_static_a.input.cell = ip.structure_a.cell.array

        # calc_static_b
        g.calc_static_b.input.job_project_path = ip.job_project_path_b
        g.calc_static_b.input.job_name = ip.job_name_b
        g.calc_static_b.input.positions = gp.reflect.output.positions[-1]
        g.calc_static_a.input.cell = ip.structure_b.cell.array

        # mix
        g.mix.input.vectors = [
            gp.calc_static_b.output.forces[-1],
            gp.calc_static_a.output.forces[-1]
        ]
        g.mix.input.weights = ip.coupling_weights

        # verlet_velocities
        g.verlet_velocities.input.velocities = gp.reflect.output.velocities[-1]
        g.verlet_velocities.input.forces = gp.mix.output.weighted_sum[-1]
        g.verlet_velocities.input.masses = ip.masses
        g.verlet_velocities.input.time_step = ip.time_step
        g.verlet_velocities.input.temperature = ip.temperature
        g.verlet_velocities.input.temperature_damping_timescale = ip.temperature_damping_timescale

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
        g.average_temp.input.sample = gp.verlet_velocities.output.instant_temperature[-1]

        # check_sampling_steps
        g.check_sampling_steps.input.target = gp.reflect.output.total_steps[-1]
        g.check_sampling_steps.input.default.mod = ip.sampling_steps

        # addition
        g.addition.input.vectors = [
            gp.calc_static_b.output.energy_pot[-1],
            gp.calc_static_a.output.energy_pot[-1],
            ip.zero_k_energy
        ]
        g.addition.input.weights = [1, -1, -1]

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
            'cutoff_distance': ~gp.reflect.output.cutoff_distance[-1],
            'temperature_mean': ~gp.average_temp.output.mean[-1],
            'temperature_std': ~gp.average_temp.output.std[-1],
            'temperature_n_samples': ~gp.average_temp.output.n_samples[-1],
            'positions': ~gp.reflect.output.positions[-1],
            'velocities': ~gp.verlet_velocities.output.velocities[-1],
            'forces': ~gp.mix.output.weighted_sum[-1],
            'total_steps': ~gp.reflect.output.total_steps[-1],
            'mean_diff': ~gp.average_tild.output.mean[-1],
            'std_diff': ~gp.average_tild.output.std[-1],
            'fep_exp_mean': ~gp.average_fep_exp.output.mean[-1],
            'fep_exp_std': ~gp.average_fep_exp.output.std[-1],
            'n_samples': ~gp.average_tild.output.n_samples[-1]
        }


class TILDParallel(CompoundVertex):
    """
    Protocol that performs Thermodynamic Integration using Langevin Dynamics (TILD) to obtain the free energy change
        between an initial state (reference job A) and the final state (reference job B). The reference jobs can be
        of the job types LammpsInteractive/VaspInteractive/SphinxInteractive/HessianJob/DecoupledOscillators.

    The standard error of mean of the free energy change, obtained over the integration of the 'lambda' or integration
        points, can be applied as a convergence criterion, in addition to letting the computation run for a maximum
        number of steps.

    In this protocol, the 'lambda'/integration points are executed in parallel, ie., each point 'evolves' independently
        of the other, and are only accessed together during the evaluation of the 'integrands'. Parallel execution of
        the integration points gives a substantial speed-up. A serial version also exists, which is an exact copy of
        the parallel version, but executed serially. Maximum efficiency for parallelization can be achieved by setting
        the number of cores the job can use to the number of lambdas, ie., cores / lambdas = 1. Setting the number of
        cores greater than the number of lambdas gives zero gain, and is wasteful if cores % lambdas != 0.

    Input attributes:
        ref_job_a_full_path (str): Path to the job containing the initial state of the system. Should be a job of type
            LammpsInteractive/VaspInteractive/SphinxInteractive/HessianJob/DecoupledOscillators. The structure of this
            job will be used as 'structure_a' in this protocol.
        ref_job_b_full_path (str): Path to the job containing the initial state of the system. Should be a job of type
            LammpsInteractive/VaspInteractive/SphinxInteractive/HessianJob/DecoupledOscillators. The structure of this
            job will be used as 'structure_b' in this protocol.
        temperature (float): Temperature to run at in K. (Default is 300.)
        n_lambdas (int): How many 'lambda'/integration points to create. (Default is 3.)
        lambda_bias (float): A function to generate N points between 0 and 1, with a left, equidistant and right bias.
            bias = 0 makes the points fully left biased. The amount of left bias can be controlled by varying it
            between 0 and 0.49.
            bias = 0.5 keeps the points equidistant.
            bias = 1 makes the points fully right biased. The amount of right bias can be controlled by varying it
            between 0.51 and 1.
            (Default is 0.5, keep the points equidistant.)
        n_steps (int): How many MD steps to run for. (Default is 100.)
        thermalization_steps (int): Number of steps the system is thermalized for to reach equilibrium. Should be
            divisible be 'n_steps'. (Default is 10 steps.)
        sampling_steps (int): Collect a 'sample' every 'sampling_steps' steps. Should be divisible be 'n_steps'.
            (Default is 1, collect sample for every MD step.
        time_step (float): MD time step in fs. (Default is 1.)
        temperature_damping_timescale (float): Langevin thermostat timescale in fs. (Default is None, which runs NVE.)
        overheat_fraction (float): The fraction by which to overheat the initial velocities. This can be useful for
            more quickly equilibrating a system whose initial structure is its fully relaxed positions -- in which
            case equipartition of energy tells us that the kinetic energy should be initialized to double the
            desired value. (Default is 2.0, assuming energy equipartition is a good idea.)
        cutoff_factor (float): The cutoff is obtained by taking the first nearest neighbor distance and multiplying
            it by the cutoff factor. (Default is 0.5.)
        use_reflection (boolean): Turn on or off 'SphereReflectionPerAtom' (Default is False.)
        zero_k_energy (float): The minimized potential energy of the static (expanded) structure. (Default is 0.)
        sleep_time (float): A delay in seconds for database access of results. For sqlite, a non-zero delay maybe
            required. (Default is 0 seconds, no delay.)
        convergence_check_steps (int): Check for convergence once every 'convergence_check_steps'. Should be
            divisible be 'n_steps'. (Default is once every 10 steps.)
        fe_tol (float): The free energy standard error tolerance. This is the convergence criterion in eV. (Default
            is 0.01 eV)
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
        super(TILDParallel, self).__init__(**kwargs)

        id_ = self.input.default
        # Default values
        id_.structure_a = None
        id_.structure_b = None
        id_.temperature = 300.
        id_.n_lambdas = 3
        id_.lambda_bias = 0.5
        id_.n_steps = 100
        id_.thermalization_steps = 10
        id_.sampling_steps = 1
        id_.time_step = 1.
        id_.temperature_damping_timescale = 100.
        id_.overheat_fraction = 2.
        id_.cutoff_factor = 0.5
        id_.use_reflection = False
        id_.zero_k_energy = 0.
        id_.sleep_time = 0.
        id_.convergence_check_steps = 10
        id_.fe_tol = 0.01
        id_._cutoff_distance = None
        id_._total_steps = 0
        id_._default_free_energy_se = 1
        id_._project_path = None
        id_._job_name = None
        id_._mean = None
        id_._std = None
        id_._n_samples = None

    def define_vertices(self):
        # Graph components
        g = self.graph
        ip = Pointer(self.input)
        g.validate = TILDValidate()
        g.build_lambdas = BuildMixingPairs()
        g.initial_forces = Zeros()
        g.mass_mixer = WeightedSum()
        g.initial_velocities = SerialList(RandomVelocity)
        g.create_jobs_a = CreateSubJobs()
        g.create_jobs_b = CreateSubJobs()
        g.check_steps = IsGEq()
        g.run_lambda_points = ParallelList(_TILDLambdaEvolution, sleep_time=ip.sleep_time)
        g.clock = Counter()
        g.post = TILDPostProcess()
        g.check_convergence = IsLEq()
        g.exit = AnyVertex()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.validate,
            g.build_lambdas,
            g.initial_forces,
            g.mass_mixer,
            g.initial_velocities,
            g.create_jobs_a,
            g.create_jobs_b,
            g.check_steps, "false",
            g.check_convergence, "false",
            g.run_lambda_points,
            g.clock,
            g.post,
            g.exit
        )
        g.make_edge(g.check_steps, g.exit, "true")
        g.make_edge(g.check_convergence, g.exit, "true")
        g.make_edge(g.exit, g.check_steps, "false")
        g.starting_vertex = g.validate
        g.restarting_vertex = g.create_jobs_a

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # validate
        g.validate.input.ref_job_a_full_path = ip.ref_job_a_full_path
        g.validate.input.ref_job_b_full_path = ip.ref_job_b_full_path
        g.validate.input.n_steps = ip.n_steps
        g.validate.input.thermalization_steps = ip.thermalization_steps
        g.validate.input.sampling_steps = ip.sampling_steps
        g.validate.input.convergence_check_steps = ip.convergence_check_steps

        # build_lambdas
        g.build_lambdas.input.n_lambdas = ip.n_lambdas
        g.build_lambdas.input.lambda_bias = ip.lambda_bias

        # initial_forces
        g.initial_forces.input.shape = gp.validate.output.structure_a[-1].positions.shape

        # mass_mixer
        g.mass_mixer.input.vectors = [
            gp.validate.output.structure_a[-1].get_masses,
            gp.validate.output.structure_b[-1].get_masses
        ]
        g.mass_mixer.input.weights = [0.5, 0.5]

        # initial_velocities
        g.initial_velocities.input.n_children = ip.n_lambdas
        g.initial_velocities.direct.temperature = ip.temperature
        g.initial_velocities.direct.masses = gp.mass_mixer.output.weighted_sum[-1]
        g.initial_velocities.direct.overheat_fraction = ip.overheat_fraction

        # create_jobs_a
        g.create_jobs_a.input.n_images = ip.n_lambdas
        g.create_jobs_a.input.ref_job_full_path = ip.ref_job_a_full_path
        g.create_jobs_a.input.structure = gp.validate.output.structure_a[-1]

        # create_jobs_b
        g.create_jobs_b.input.n_images = ip.n_lambdas
        g.create_jobs_b.input.ref_job_full_path = ip.ref_job_b_full_path
        g.create_jobs_b.input.structure = gp.validate.output.structure_b[-1]

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # check_convergence
        g.check_convergence.input.default.target = ip._default_free_energy_se
        g.check_convergence.input.target = gp.post.output.tild_free_energy_se[-1]
        g.check_convergence.input.threshold = ip.fe_tol

        # run_lambda_points - initialize
        g.run_lambda_points.input.n_children = ip.n_lambdas

        # run_lambda_points - verlet_positions
        g.run_lambda_points.direct.time_step = ip.time_step
        g.run_lambda_points.direct.temperature = ip.temperature
        g.run_lambda_points.direct.masses = gp.mass_mixer.output.weighted_sum[-1]
        g.run_lambda_points.direct.temperature_damping_timescale = ip.temperature_damping_timescale
        g.run_lambda_points.direct.structure_a = gp.validate.output.structure_a[-1]
        g.run_lambda_points.direct.structure_b = gp.validate.output.structure_b[-1]

        g.run_lambda_points.direct.default.positions = gp.validate.output.structure_a[-1].positions
        g.run_lambda_points.broadcast.default.velocities = gp.initial_velocities.output.velocities[-1]
        g.run_lambda_points.direct.default.forces = gp.initial_forces.output.zeros[-1]

        g.run_lambda_points.broadcast.positions = gp.run_lambda_points.output.positions[-1]
        g.run_lambda_points.broadcast.velocities = gp.run_lambda_points.output.velocities[-1]
        g.run_lambda_points.broadcast.forces = gp.run_lambda_points.output.forces[-1]

        # run_lambda_points - reflect
        g.run_lambda_points.direct.default.total_steps = ip._total_steps
        g.run_lambda_points.direct.default.cutoff_distance = ip._cutoff_distance
        g.run_lambda_points.broadcast.total_steps = gp.run_lambda_points.output.total_steps[-1]
        g.run_lambda_points.broadcast.cutoff_distance = gp.run_lambda_points.output.cutoff_distance[-1]
        g.run_lambda_points.direct.cutoff_factor = ip.cutoff_factor
        g.run_lambda_points.direct.use_reflection = ip.use_reflection

        # run_lambda_points - calc_static_a
        g.run_lambda_points.broadcast.job_project_path_a = gp.create_jobs_a.output.jobs_project_path[-1]
        g.run_lambda_points.broadcast.job_name_a = gp.create_jobs_a.output.jobs_names[-1]

        # run_lambda_points - calc_static_b
        g.run_lambda_points.broadcast.job_project_path_b = gp.create_jobs_b.output.jobs_project_path[-1]
        g.run_lambda_points.broadcast.job_name_b = gp.create_jobs_b.output.jobs_names[-1]

        # run_lambda_points - mix
        g.run_lambda_points.broadcast.coupling_weights = gp.build_lambdas.output.lambda_pairs[-1]

        # run_lambda_points - verlet_velocities
        # takes inputs already specified

        # run_lambda_points - check_thermalized
        g.run_lambda_points.direct.thermalization_steps = ip.thermalization_steps

        # run_lambda_points - check_sampling_steps
        g.run_lambda_points.direct.sampling_steps = ip.sampling_steps

        # run_lambda_points - average_temp
        g.run_lambda_points.direct.default.average_temp_mean = ip._mean
        g.run_lambda_points.direct.default.average_temp_std = ip._std
        g.run_lambda_points.direct.default.average_temp_n_samples = ip._n_samples
        g.run_lambda_points.broadcast.average_temp_mean = gp.run_lambda_points.output.temperature_mean[-1]
        g.run_lambda_points.broadcast.average_temp_std = gp.run_lambda_points.output.temperature_std[-1]
        g.run_lambda_points.broadcast.average_temp_n_samples = gp.run_lambda_points.output.temperature_n_samples[-1]

        # run_lambda_points - addition
        g.run_lambda_points.direct.zero_k_energy = ip.zero_k_energy

        # run_lambda_points - average_tild
        g.run_lambda_points.direct.default.average_tild_mean = ip._mean
        g.run_lambda_points.direct.default.average_tild_std = ip._std
        g.run_lambda_points.direct.default.average_tild_n_samples = ip._n_samples
        g.run_lambda_points.broadcast.average_tild_mean = gp.run_lambda_points.output.mean_diff[-1]
        g.run_lambda_points.broadcast.average_tild_std = gp.run_lambda_points.output.std_diff[-1]
        g.run_lambda_points.broadcast.average_tild_n_samples = gp.run_lambda_points.output.n_samples[-1]

        # run_lambda_points - fep_exp
        g.run_lambda_points.broadcast.delta_lambdas = gp.build_lambdas.output.delta_lambdas[-1]

        # run_lambda_points - average_fep_exp
        g.run_lambda_points.direct.default.average_fep_exp_mean = ip._mean
        g.run_lambda_points.direct.default.average_fep_exp_std = ip._std
        g.run_lambda_points.direct.default.average_fep_exp_n_samples = ip._n_samples
        g.run_lambda_points.broadcast.average_fep_exp_mean = gp.run_lambda_points.output.fep_exp_mean[-1]
        g.run_lambda_points.broadcast.average_fep_exp_std = gp.run_lambda_points.output.fep_exp_std[-1]
        g.run_lambda_points.broadcast.average_fep_exp_n_samples = gp.run_lambda_points.output.n_samples[-1]

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
        g.exit.input.vertex_states = [
            gp.check_steps.vertex_state,
            gp.check_convergence.vertex_state
        ]
        g.exit.input.print_strings = [
            "Maximum steps ({}) reached. Stopping run.",
            "Convergence reached in {} steps. Stopping run.",
            "Convergence not reached in {} steps. Continuing run..."  # 1 extra, if convergence is not reached!
        ]
        g.exit.input.step = gp.clock.output.n_counts[-1]
        g.exit.input.n_steps = ip.n_steps

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        o = Pointer(self.graph.run_lambda_points.output)
        return {
            'total_steps': ~o.total_steps[-1],
            'lambdas': ~gp.build_lambdas.output.lambda_pairs[-1][:, 0],
            'temperature_mean': ~o.temperature_mean[-1],
            'temperature_std': ~o.temperature_std[-1],
            'integrands': ~o.mean_diff[-1],
            'integrands_std': ~o.std_diff[-1],
            'integrands_n_samples': ~o.n_samples[-1],
            'tild_free_energy_mean': ~gp.post.output.tild_free_energy_mean[-1],
            'tild_free_energy_std': ~gp.post.output.tild_free_energy_std[-1],
            'tild_free_energy_se': ~gp.post.output.tild_free_energy_se[-1],
            'fep_free_energy_mean': ~gp.post.output.fep_free_energy_mean[-1],
            'fep_free_energy_std': ~gp.post.output.fep_free_energy_std[-1],
            'fep_free_energy_se': ~gp.post.output.fep_free_energy_se[-1]
        }

    def get_lambdas(self):
        """
        Get the lambda values.
        """
        return self.output.lambdas[-1]

    def _get_tild_integrands(self):
        o = self.output
        return np.array(o.integrands[-1]), o.integrands_std[-1] / np.sqrt(o.integrands_n_samples[-1])

    def plot_tild_integrands(self):
        """
        Plot the integrand values with their standard errors against the lambda values.
        """
        fig, ax = plt.subplots()
        lambdas = self.get_lambdas()
        thermal_average, standard_error = self._get_tild_integrands()
        ax.plot(lambdas, thermal_average, marker='o')
        ax.fill_between(lambdas, thermal_average - standard_error, thermal_average + standard_error, alpha=0.3)
        ax.set_xlabel("Lambda")
        ax.set_ylabel("dF/dLambda")
        plt.show()
        return fig, ax

    # def get_classical_harmonic_free_energy(self, temperatures=None):
    #     """
    #     Get the total free energy of a harmonic oscillator with this frequency and these atoms. Temperatures are
    #         clipped at 1 micro-Kelvin.
    #     Returns:
    #         float/np.ndarray: The sum of the free energy of each atom.
    #     """
    #     if temperatures is None:
    #         temperatures = self.input.temperature
    #     temperatures = np.clip(temperatures, 1e-6, np.inf)
    #     beta = 1. / (KB * temperatures)
    #
    #     return -3 * len(self.input.structure) * np.log(np.pi / (self.input.spring_constant * beta)) / (2 * beta)
    #
    # def get_quantum_harmonic_free_energy(self, temperatures=None):
    #     """
    #     Get the total free energy of a harmonic oscillator with this frequency and these atoms. Temperatures are
    #         clipped at 1 micro-Kelvin.
    #     Returns:
    #         float/np.ndarray: The sum of the free energy of each atom.
    #     """
    #     if temperatures is None:
    #         temperatures = self.input.temperature
    #     temperatures = np.clip(temperatures, 1e-6, np.inf)
    #     beta = 1. / (KB * temperatures)
    #     f = 0
    #     for m in self.input.structure.get_masses():
    #         hbar_omega = HBAR * np.sqrt(self.input.spring_constant / m) * ROOT_EV_PER_ANGSTROM_SQUARE_PER_AMU_IN_S
    #         f += (3. / 2) * hbar_omega + ((3. / beta) * np.log(1 - np.exp(-beta * hbar_omega)))
    #     return f


class ProtoTILDPar(Protocol, TILDParallel):
    pass


class TILDSerial(TILDParallel):
    """
    Serial version of TILDParallel.
    """
    def define_vertices(self):
        # Graph components
        g = self.graph
        g.validate = TILDValidate()
        g.build_lambdas = BuildMixingPairs()
        g.initial_forces = Zeros()
        g.mass_mixer = WeightedSum()
        g.initial_velocities = SerialList(RandomVelocity)
        g.create_jobs_a = CreateSubJobs()
        g.create_jobs_b = CreateSubJobs()
        g.check_steps = IsGEq()
        g.run_lambda_points = SerialList(_TILDLambdaEvolution)
        g.clock = Counter()
        g.post = TILDPostProcess()
        g.check_convergence = IsLEq()
        g.exit = AnyVertex()


class ProtoTILDSer(Protocol, TILDSerial):
    pass
