# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

from pyiron_contrib.protocol.generic import CompoundVertex, Protocol
from pyiron_contrib.protocol.primitive.one_state import BerendsenBarostat, Counter, CreateSubJobs, ExternalHamiltonian,\
    RandomVelocity, VerletPositionUpdate, VerletVelocityUpdate, Zeros
from pyiron_contrib.protocol.primitive.two_state import IsGEq
from pyiron_contrib.protocol.utils import Pointer

"""
Protocols for molecular dynamics.
"""

__author__ = "Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "18 July, 2019"


class MolecularDynamics(CompoundVertex):
    """
    Runs molecular dynamics. This isn't particularly useful as almost every source code/plain job can do this on its
     own, but rather this is intended for testing and teaching. It also serves as a useful starting point for
     developing algorithms with modified dynamics.

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
        pressure (float): The pressure in GPa to be simulated (Default is None GPa)
        pressure_style (string): 'isotropic' or 'anisotropic'. (Default is 'anisotropic')
        pressure_damping_timescale (float): Damping timescale in fs. (Default is None, no barostat is used.)
        compressibility (float): The compressibility of water in bar-1 (Default is 4.57e-5 bar-1)
        previous_volume (float): The default volume. (Defaults is None.)
        energy_kin (float): The default energy_kin. (Default is None.)

    Output attributes:
        energy_pot (float): Total potential energy of the system in eV.
        energy_kin (float): Total kinetic energy of the system in eV.
        positions (numpy.ndarray): Atomic positions in angstroms.
        velocities (numpy.ndarray): Atomic velocities in angstroms/fs.
        forces (numpy.ndarray): Atomic forces in eV/angstrom. Note: These are the potential gradient forces; thermostat
            forces (if any) are not saved.
        pressure (float): The isotropic pressure in GPa.
        volume (float): The volume of the system in Ang3.
    """
    # DefaultWhitelist sets the output which will be stored every `archive period' in the final hdf5 file.

    DefaultWhitelist = {
        'verlet_positions': {
            'output': {
                'positions': 1000,
            },
        },
        'calc_static': {
            'output': {
                'energy_pot': 1,
                'forces': 1000,
            },
        },
        'verlet_velocities': {
            'output': {
                'energy_kin': 1,
                'velocities': 1000,
            },
        },
    }

    def __init__(self, **kwargs):
        super(MolecularDynamics, self).__init__(**kwargs)

        # Protocol defaults
        id_ = self.input.default
        id_.temperature = None
        id_.n_steps = 100
        id_.temperature_damping_timescale = 100.
        id_.time_step = 1.
        id_.overheat_fraction = 2
        id_.pressure = None
        id_.pressure_style = "anisotropic"
        id_.pressure_damping_timescale = 1000.
        id_._compressibility = 4.57e-5  # bar^-1
        id_._previous_volume = None
        id_._energy_kin = None
        id_._n_children = 1

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.initialize_job = CreateSubJobs()
        g.initial_velocities = RandomVelocity()
        g.initial_forces = Zeros()
        g.initial_pressures = Zeros()
        g.check_steps = IsGEq()
        g.clock = Counter()
        g.barostat = BerendsenBarostat()
        g.verlet_positions = VerletPositionUpdate()
        g.calc_static = ExternalHamiltonian()
        g.verlet_velocities = VerletVelocityUpdate()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.initialize_job,
            g.initial_velocities,
            g.initial_forces,
            g.initial_pressures,
            g.check_steps, "false",
            g.barostat,
            g.verlet_positions,
            g.calc_static,
            g.verlet_velocities,
            g.clock,
            g.check_steps
        )
        g.starting_vertex = g.initialize_job
        g.restarting_vertex = g.check_steps

    def define_information_flow(self):
        # Data flow
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)

        # initialize_job
        g.initialize_job.input.n_images = ip._n_children
        g.initialize_job.input.ref_job_full_path = ip.ref_job_full_path
        g.initialize_job.input.structure = ip.structure

        # initial_velocities
        g.initial_velocities.input.temperature = ip.temperature
        g.initial_velocities.input.masses = ip.structure.get_masses
        g.initial_velocities.input.overheat_fraction = ip.overheat_fraction

        # initial_forces
        g.initial_forces.input.shape = ip.structure.positions.shape

        # initial_pressures
        g.initial_pressures.input.shape = ip.structure.cell.array.shape

        # check_steps
        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        # barostat
        g.barostat.input.default.box_pressure = gp.initial_pressures.output.zeros[-1]
        g.barostat.input.default.structure = ip.structure
        g.barostat.input.default.energy_kin = ip._energy_kin
        g.barostat.input.default.previous_volume = ip._previous_volume
        g.barostat.input.default.positions = ip.structure.positions

        g.barostat.input.box_pressure = gp.calc_static.output.pressures[-1]
        g.barostat.input.structure = gp.barostat.output.structure[-1]
        g.barostat.input.energy_kin = gp.verlet_velocities.output.energy_kin[-1]
        g.barostat.input.previous_volume = gp.calc_static.output.volume[-1]
        g.barostat.input.positions = gp.verlet_positions.output.positions[-1]
        g.barostat.input.pressure = ip.pressure
        g.barostat.input.temperature = ip.temperature
        g.barostat.input.time_step = ip.time_step
        g.barostat.input.pressure_damping_timescale = ip.pressure_damping_timescale
        g.barostat.input.compressibility = ip._compressibility
        g.barostat.input.pressure_style = ip.pressure_style

        # verelt_positions
        g.verlet_positions.input.default.velocities = gp.initial_velocities.output.velocities[-1]
        g.verlet_positions.input.default.forces = gp.initial_forces.output.zeros[-1]

        g.verlet_positions.input.positions = gp.barostat.output.positions[-1]
        g.verlet_positions.input.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.input.forces = gp.calc_static.output.forces[-1]
        g.verlet_positions.input.masses = ip.structure.get_masses
        g.verlet_positions.input.time_step = ip.time_step
        g.verlet_positions.input.temperature = ip.temperature
        g.verlet_positions.input.temperature_damping_timescale = ip.temperature_damping_timescale

        # calc_static
        g.calc_static.input.job_project_path = gp.initialize_job.output.jobs_project_path[-1][-1]
        g.calc_static.input.job_name = gp.initialize_job.output.jobs_names[-1][-1]
        g.calc_static.input.positions = gp.verlet_positions.output.positions[-1]
        g.calc_static.input.cell = gp.barostat.output.structure[-1].cell.array

        # verlet_velocities
        g.verlet_velocities.input.velocities = gp.verlet_positions.output.velocities[-1]
        g.verlet_velocities.input.forces = gp.calc_static.output.forces[-1]
        g.verlet_velocities.input.masses = ip.structure.get_masses
        g.verlet_velocities.input.time_step = ip.time_step
        g.verlet_velocities.input.temperature = ip.temperature
        g.verlet_velocities.input.temperature_damping_timescale = ip.temperature_damping_timescale

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'energy_pot': ~gp.calc_static.output.energy_pot[-1],
            'energy_kin': ~gp.verlet_velocities.output.energy_kin[-1],
            'positions': ~gp.verlet_positions.output.positions[-1],
            'velocities': ~gp.verlet_velocities.output.velocities[-1],
            'forces': ~gp.calc_static.output.forces[-1],
            'pressure': ~gp.barostat.output.pressure[-1],
            'volume': ~gp.calc_static.output.volume[-1]
        }


class ProtoMD(Protocol, MolecularDynamics):
    pass
