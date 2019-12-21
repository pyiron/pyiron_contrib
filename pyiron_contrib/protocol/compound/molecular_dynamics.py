# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

from pyiron_contrib.protocol.generic import CompoundVertex, Protocol
from pyiron_contrib.protocol.primitive.one_state import Counter, ExternalHamiltonian, RandomVelocity, Zeros, \
    VerletPositionUpdate, VerletVelocityUpdate
from pyiron_contrib.protocol.primitive.two_state import IsGEq
from pyiron_contrib.protocol.utils import Pointer

"""
Protocol for molecular dynamics.
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
    own, but rather this is intended for testing and teaching. It also serves as a useful starting point for developing
    algorithms with modified dynamics.

    Input attributes:
        ref_job_full_path (str): Path to the pyiron job to use for evaluating forces and energies.
        structure (Atoms): The structure evolve.
        temperature (float): Temperature to run at in K.
        n_steps (int): How many MD steps to run for. (Default is 100.)
        temperature_damping_timescale (float): Langevin thermostat timescale in fs. (Default is None, which runs NVE.)
        time_step (float): MD time step in fs. (Default is 1.)

    Output attributes:
        energy_pot (float): Total potential energy of the system in eV.
        energy_kin (float): Total kinetic energy of the system in eV.
        positions (numpy.ndarray): Atomic positions in angstroms.
        velocities (numpy.ndarray): Atomic velocities in angstroms/fs.
        forces (numpy.ndarray): Atomic forces in eV/angstrom. Note: These are the potential gradient forces; thermostat
            forces (if any) are not saved.
    """

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
        id_.n_steps = 100
        id_.time_step = 1.
        id_.temperature_damping_timescale = None

    def define_vertices(self):
        # Graph components
        g = self.graph
        g.initial_velocity = RandomVelocity()
        g.initial_forces = Zeros()
        g.check_steps = IsGEq()
        g.clock = Counter()
        g.calc_static = ExternalHamiltonian()
        g.verlet_positions = VerletPositionUpdate()
        g.verlet_velocities = VerletVelocityUpdate()

    def define_execution_flow(self):
        # Execution flow
        g = self.graph
        g.make_pipeline(
            g.initial_velocity,
            g.initial_forces,
            g.check_steps, 'false',
            g.verlet_positions,
            g.calc_static,
            g.verlet_velocities,
            g.clock,
            g.check_steps
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

        g.verlet_positions.input.default.positions = ip.structure.positions
        g.verlet_positions.input.default.velocities = gp.initial_velocity.output.velocities[-1]
        g.verlet_positions.input.default.forces = gp.initial_forces.output.zeros[-1]
        g.verlet_positions.input.positions = gp.verlet_positions.output.positions[-1]
        g.verlet_positions.input.velocities = gp.verlet_velocities.output.velocities[-1]
        g.verlet_positions.input.forces = gp.calc_static.output.forces[-1]
        g.verlet_positions.input.masses = ip.structure.get_masses
        g.verlet_positions.input.time_step = ip.time_step
        g.verlet_positions.input.temperature = ip.temperature
        g.verlet_positions.input.temperature_damping_timescale = ip.temperature_damping_timescale

        g.calc_static.input.ref_job_full_path = ip.ref_job_full_path
        g.calc_static.input.structure = ip.structure
        g.calc_static.input.positions = gp.verlet_positions.output.positions[-1]

        g.verlet_velocities.input.velocities = gp.verlet_positions.output.velocities[-1]
        g.verlet_velocities.input.forces = gp.calc_static.output.forces[-1]
        g.verlet_velocities.input.masses = ip.structure.get_masses
        g.verlet_velocities.input.time_step = ip.time_step
        g.verlet_velocities.input.temperature = ip.temperature
        g.verlet_velocities.input.temperature_damping_timescale = ip.temperature_damping_timescale

        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        self.set_graph_archive_clock(gp.clock.output.n_counts[-1])

    def get_output(self):
        gp = Pointer(self.graph)
        return {
            'energy_pot': ~gp.calc_static.output.energy_pot[-1],
            'energy_kin': ~gp.verlet_velocities.output.energy_kin[-1],
            'positions': ~gp.verlet_positions.output.positions[-1],
            'velocities': ~gp.verlet_velocities.output.velocities[-1],
            'forces': ~gp.calc_static.output.forces[-1]
        }


class ProtocolMD(Protocol, MolecularDynamics):
    def __init__(self, project=None, job_name=None):
        super(ProtocolMD, self).__init__(project=project, job_name=job_name)
