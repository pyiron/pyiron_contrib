# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import numpy as np
from os.path import split
from abc import ABC, abstractmethod
from uncertainties import unumpy
from scipy.constants import physical_constants
from scipy.integrate import simps
from ase.geometry import get_distances

from pyiron import Project
from pyiron_atomistics.atomistics.job.interactive import GenericInteractive
from pyiron_atomistics.lammps.lammps import LammpsInteractive
from pyiron_contrib.protocol.generic import PrimitiveVertex
from pyiron_contrib.protocol.utils import Pointer
from pyiron_contrib.protocol.utils import ensure_iterable
from pyiron_contrib.protocol.math import welford_online

KB = physical_constants['Boltzmann constant in eV/K'][0]
EV_TO_U_ANGSQ_PER_FSSQ = 0.00964853322
# https://www.wolframalpha.com/input/?i=1+eV+in+u+*+%28angstrom%2Ffs%29%5E2
U_ANGSQ_PER_FSSQ_TO_EV = 1. / EV_TO_U_ANGSQ_PER_FSSQ
EV_PER_ANGCUB_TO_GPA = 160.21766208  # eV/A^3 to GPa
GPA_TO_BAR = 1e4

"""
Primitive vertices which have only one outbound execution edge.
"""

__author__ = "Liam Huber, Raynol Dsouza, Dominik Noeger"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "20 July, 2019"


class BuildMixingPairs(PrimitiveVertex):
    """
    Builds an array of mixing parameters [lambda, (1-lambda)], and also finds the deltas between consecutive
        lambdas.
    Input attributes:
        n_lambdas (int): How many mixing pairs to create. (Default is 5.)
        custom_lambdas (list/numpy.ndarray): The individual lambda values to use for the first member of each pair.
            (Default is None.)
    Output attributes:
        lambda_pairs (numpy.ndarray): The (`n_lambdas`, 2)-shaped array of mixing pairs.
        delta_lambdas (numpy.ndarray): The delta between two consecutive lambdas. The end deltas are
            halved.
    """

    def __init__(self, name=None):
        super(BuildMixingPairs, self).__init__(name=name)
        self.input.default.n_lambdas = 5
        self.input.default.custom_lambdas = None

    def command(self, n_lambdas, custom_lambdas):

        if custom_lambdas is not None:
            lambdas = np.array(custom_lambdas)
        else:
            lambdas = np.linspace(0, 1, num=n_lambdas)

        delta_lambdas = np.gradient(lambdas)
        delta_lambdas[0] = delta_lambdas[0] / 2
        delta_lambdas[-1] = delta_lambdas[-1] / 2

        return {
                'lambda_pairs': np.array([lambdas, 1 - lambdas]).T,
                'delta_lambdas': delta_lambdas
            }


class Counter(PrimitiveVertex):
    """
    Increments by one at each execution. Can be made to increment from a specific value.
    Input attributes:
        add_counts (int): A specific value from which to increment. (Default is 0.)
    Output attributes:
        n_counts (int): How many executions have passed. (Default is 0.)
    """

    def __init__(self, name=None):
        super(Counter, self).__init__(name=name)
        self.input.default.add_counts = 0
        self.output.n_counts = [0]

    def command(self, add_counts):
        if add_counts > 0:
            count = self.output.n_counts[-1] + add_counts
        else:
            count = self.output.n_counts[-1] + 1
        return {
            'n_counts': count
        }


class Compute(PrimitiveVertex):
    """
    """

    def __init__(self, name=None):
        super(Compute, self).__init__(name=name)

    def command(self, *args, **kwargs):
        function_ = self.input.function
        return function_(*args, **kwargs)


class Zeros(PrimitiveVertex):
    """
    A numpy vector of zeros that is pointer-compatible.
    Input attributes:
        shape (int/tuple): The shape of the array.
    Output attributes:
        zeros (numpy.ndarray): An array of numpy.float64 zeros.
    """

    def command(self, shape):
        return {
            'zeros': np.zeros(shape)
        }


class DeleteAtom(PrimitiveVertex):
    """
    Given a structure, deletes one of the atoms.
    Input attributes:
        structure (Atoms): The structure to delete an atom of.
        atom_id (int): Which atom to delete. (Default is 0, the 0th atom.)
    Output attributes:
        structure (Atoms): The new, modified structure.
        mask (numpy.ndarray): The integer ids shared by both the old and new structure.
    """

    def __init__(self, name=None):
        super(DeleteAtom, self).__init__(name=name)
        self.input.default.atom_id = 0

    def command(self, structure, atom_id):
        vacancy_structure = structure.copy()
        vacancy_structure.pop(atom_id)
        mask = np.delete(np.arange(len(structure)).astype(int), atom_id)

        return {
            'structure': vacancy_structure,
            'mask': mask
        }


class ExternalHamiltonian(PrimitiveVertex):
    """
    Manages calls to an external interpreter (e.g. Lammps, Vasp, Sphinx...) to produce energies, forces,
        and possibly other properties. The collected output can be expanded beyond forces and energies
        (e.g. to magnetic properties or whatever else the interpreting code produces) by modifying the
        `interesting_keys` in the input. The property must have a corresponding interactive getter for
        this property.
    Input attributes:
        ref_job_full_path (string): The full path to the hdf5 file of the job to use as a reference template.
            (Default is None.)
        project_path (string): The path of the project. To be specified ONLY when the jobs are already
            initialized outside of this vertex. If `project_path` is specified, `ref_job_full_path` must
            be set to None. (Default is None.)
        job_name (string): The name of the job. This is specified only when the jobs are already
            initialized outside of this vertex. (Default is None.)
        structure (Atoms): The structure for initializing the external Hamiltonian. Overwrites the reference
            job structure when provided. (Default is None, the reference job needs to have its structure set.)
        positions (numpy.ndarray): New positions to evaluate. Shape must match the shape of the structure.
            (Default is None, only necessary if positions are being updated.)
        cell (numpy.ndarray): The cell, if not same as that in the specified structure. (Default is None,
            same cell as in the structure.)
        interesting_keys (list[str]): String codes for output properties of the underlying job to collect.
            (Default is ['positions', 'forces', 'energy_pot', 'pressures', 'volume', 'cell'].)
    Output attributes:
        keys (list/numpy.ndarray): The output corresponding to the interesting keys.
    """

    def __init__(self, name=None):
        super(ExternalHamiltonian, self).__init__(name=name)
        self._fast_lammps_mode = True  # Set to false only to intentionally be slow for comparison purposes
        self._job_project_path = None
        self._job = None
        self._job_name = None
        id_ = self.input.default
        id_.ref_job_full_path = None
        id_.project_path = None
        id_.job_name = None
        id_.structure = None
        id_.positions = None
        id_.cell = None
        id_.interesting_keys = ['positions', 'forces', 'energy_pot', 'pressures', 'volume', 'cell']

    def command(self, ref_job_full_path, project_path, job_name, structure, positions, cell, interesting_keys):

        if self._job_project_path is None:
            if project_path is None and ref_job_full_path is not None:
                self._job_project_path, self._job_name = self._initialize(self.get_graph_location(),
                                                                          ref_job_full_path,
                                                                          structure,
                                                                          self._fast_lammps_mode
                                                                          )
            elif project_path is not None and ref_job_full_path is None:
                self._job_project_path = project_path
                self._job_name = job_name
            else:
                raise AttributeError('Please specify valid project path OR ref_job_full_path, but not both!')

        if self._job is None:
            self._reload()
        elif not self._job.interactive_is_activated():
            self._job.status.running = True
            self._job.interactive_open()
            self._job.interactive_initialize_interface()

        if isinstance(self._job, LammpsInteractive) and self._fast_lammps_mode:
            if positions is not None:
                self._job.interactive_positions_setter(positions)
            if cell is not None:
                self._job.interactive_cells_setter(cell)
            self._job._interactive_lib_command(self._job._interactive_run_command)
        elif isinstance(self._job, GenericInteractive):
            # DFT codes are slow enough that we can run them the regular way and not care
            # Also we might intentionally run Lammps slowly for comparison purposes
            if positions is not None:
                self._job.structure.positions = positions
            if cell is not None:
                self._job.structure.cell = cell
            self._job.calc_static()
            self._job.run()
        else:
            raise TypeError('Job of class {} is not compatible.'.format(self._job.__class__))

        return {key: self._get_interactive_value(key) for key in interesting_keys}

    @staticmethod
    def _initialize(graph_location, ref_job_full_path, structure, fast_lammps_mode, name=None):
        """
        Initialize / create the interactive job and save it.
        """
        if name is None:
            name = graph_location + '_job'
        project_path, ref_job_path = split(ref_job_full_path)
        pr = Project(path=project_path)
        ref_job = pr.load(ref_job_path)
        job = ref_job.copy_to(
            project=pr,
            new_job_name=name,
            input_only=True,
            new_database_entry=True
        )
        if structure is not None:
            job.structure = structure
        if isinstance(job, GenericInteractive):
            job.interactive_open()
            if isinstance(job, LammpsInteractive) and fast_lammps_mode:
                # Note: This might be done by default at some point in LammpsInteractive,
                # and could then be removed here
                job.interactive_flush_frequency = 10 ** 10
                job.interactive_write_frequency = 10 ** 10
            job.calc_static()
            job.run()
        else:
            raise TypeError('Job of class {} is not compatible.'.format(ref_job.__class__))

        return job.project.path, job.job_name

    def _reload(self):
        """
        Reload a saved job from its `project_path` and `job_name`.
        """
        pr = Project(path=self._job_project_path)
        self._job = pr.load(self._job_name)
        self._job.interactive_open()
        self._job.interactive_initialize_interface()

    def _get_interactive_value(self, key):
        """
        Get output corresponding to the `interesting_keys` from interactive job.
        """
        if key == 'positions':
            val = np.array(self._job.interactive_positions_getter())
        elif key == 'forces':
            val = np.array(self._job.interactive_forces_getter())
        elif key == 'energy_pot':
            val = self._job.interactive_energy_pot_getter()
        elif key == 'pressures':
            val = np.array(self._job.interactive_pressures_getter())
        elif key == 'volume':
            val = self._job.interactive_volume_getter()
        elif key == 'cell':
            val = np.array(self._job.interactive_cells_getter())
        else:
            raise NotImplementedError
        return val

    def finish(self):
        """
        Close the interactive job.
        """
        super(ExternalHamiltonian, self).finish()
        if self._job is not None:
            self._job.interactive_close()


class CreateJob(ExternalHamiltonian):
    """
    Creates a job of an external interpreter (e.g. Lammps, Vasp, Sphinx...) outside of the `ExternalHamiltonian`.
        This vertex does not run the interpreter, but only creates the job and saves it.
    Input attributes:
        ref_job_full_path (string): The full path to the hdf5 file of the job to use as a reference template.
            (Default is None.)
        n_images (int): Number of jobs to create. (Default is 5.)
        structure (Atoms): The structure for initializing the external Hamiltonian. Overwrites the reference
            job structure when provided. (Default is None, the reference job needs to have its structure set.)
    Output attributes:
        project_path (list/string): The path of the project.
        job_names (list/string): The name of the job.
    """

    def __init__(self, name=None):
        super(CreateJob, self).__init__(name=name)
        self._fast_lammps_mode = True
        self._project_path = None
        self._job_names = None
        id_ = self.input.default
        id_.ref_job_full_path = None
        id_.n_images = 5
        id_.structure = None

    def command(self, ref_job_full_path, n_images, structure, *args, **kwargs):
        graph_location = self.get_graph_location()
        job_names = []
        project_path = []
        for i in np.arange(n_images):
            name = graph_location + '_' + str(i)
            output = self._initialize(graph_location, ref_job_full_path, structure,
                                      self._fast_lammps_mode, name)
            project_path.append(output[0])
            job_names.append(output[1])
        self._project_path = project_path
        self._job_names = job_names

        return {
            'project_path': project_path,
            'job_names': job_names
        }

    def finish(self):
        """
        Close the interactive job.
        """
        super(CreateJob, self).finish()
        if all(v is not None for v in [self._project_path, self._job_names]):
            pr = Project(path=self._project_path[-1])
            for jn in self._job_names:
                job = pr.load(jn)
                if isinstance(job, GenericInteractive):
                    job.interactive_close()
                    job.status.finished = True


class MinimizeReferenceJob(ExternalHamiltonian):
    """
    Minimizes a job using an external interpreter (e.g. Lammps, Vasp, Sphinx...) outside of the `ExternalHamiltonian`.
        This vertex minimizes the job at constant volume or constant pressure.
    Input attributes:
        ref_job_full_path (string): The full path to the hdf5 file of the job to use as a reference template.
            (Default is None.)
        pressure (float): Pressure of the system. (Default is None.)
        structure (Atoms): The structure for initializing the external Hamiltonian. Overwrites the reference
            job structure when provided. (Default is None, the reference job needs to have its structure set.)
    Output attributes:
        energy_pot (float): The evaluated potential energy.
        forces (numpy.ndarray): The evaluated forces.
    """

    def __init__(self, name=None):
        super(MinimizeReferenceJob, self).__init__(name=name)
        self._fast_lammps_mode = True
        id_ = self.input.default
        id_.ref_job_full_path = None
        id_.pressure = None
        id_.structure = None

    def command(self, ref_job_full_path, structure, pressure=None, *args, **kwargs):
        graph_location = self.get_graph_location()
        name = 'minimize_ref_job'
        project_path, job_name = self._initialize(graph_location, ref_job_full_path, structure,
                                                  self._fast_lammps_mode, name)
        pr = Project(path=project_path)
        job = pr.load(job_name)
        job.structure = structure
        job.calc_minimize(pressure=pressure)
        job.run(delete_existing_job=True)
        if isinstance(job, GenericInteractive):
            job.interactive_close()

        return {
            'energy_pot': job.output.energy_pot[-1],
            'forces': job.output.forces[-1]
        }


class RemoveJob(PrimitiveVertex):
    """
    Remove an existing job/s from a project path.
    Input attributes:
        project_path (string): The path of the project. (Default is None.)
        job_names (string): The names of the jobs to be removed. (Default is None.)
    """
    def __init__(self, name=None):
        super(RemoveJob, self).__init__(name=name)
        self.input.default.project_path = None
        self.input.default.job_names = None

    def command(self, project_path, job_names):
        if all(v is not None for v in [project_path, job_names]):
            pr = Project(path=project_path)
            for name in job_names:
                pr.remove_job(name)


class GradientDescent(PrimitiveVertex):
    """
    Simple gradient descent update for positions in `flex_output` and structure.
    Input attributes:
        gamma0 (float): Initial step size as a multiple of the force. (Default is 0.1.)
        fix_com (bool): Whether the center of mass motion should be subtracted off of the position update.
            (Default is True)
        use_adagrad (bool): Whether to have the step size decay according to adagrad. (Default is False)
        output_displacements (bool): Whether to return the per-atom displacement vector in the output dictionary.
    Output attributes:
        positions (numpy.ndarray): The updated positions.
        displacements (numpy.ndarray): The displacements, if `output_displacements` is True.
    TODO: Fix adagrad bug when GradientDescent is passed as a Serial vertex
    """

    def __init__(self, name=None):
        super(GradientDescent, self).__init__(name=name)
        self._accumulated_force = 0
        id_ = self.input.default
        id_.gamma0 = 0.1
        id_.fix_com = True
        id_.use_adagrad = False

    def command(self, positions, forces, gamma0, use_adagrad, fix_com, mask=None, masses=None,
                output_displacements=True):

        positions = np.array(positions)
        forces = np.array(forces)
        unmasked_positions = None

        if mask is not None:
            masked = True
            mask = np.array(mask)
            # Preserve data
            unmasked_positions = positions.copy()

            # Mask input data
            positions = positions[mask]
            forces = forces[mask]
            masses = np.array(masses)[mask]
        else:
            masked = False

        if use_adagrad:
            self._accumulated_force += np.sqrt(np.sum(forces * forces))
            gamma0 /= self._accumulated_force

        pos_change = gamma0 * np.array(forces)

        if fix_com:
            masses = np.array(masses)[:, np.newaxis]
            total_mass = np.sum(masses)
            com_change = np.sum(pos_change * masses, axis=0) / total_mass
            pos_change -= com_change
        # TODO: fix angular momentum

        new_pos = positions + pos_change

        if masked:
            unmasked_positions[mask] = new_pos
            new_pos = unmasked_positions
            disp = np.zeros(unmasked_positions.shape)
            disp[mask] = pos_change
        else:
            disp = pos_change

        if output_displacements:
            return {
                'positions': new_pos,
                'displacements': disp
            }
        else:
            return {
                'positions': new_pos
            }


class InitialPositions(PrimitiveVertex):
    """
    Assigns initial positions. If no initial positions are specified, interpolates between the positions of the
        initial and the final structures.
    Input attributes:
        initial_positions (list/numpy.ndarray): The initial positions (Default is None)
        structure_initial (Atoms): The initial structure
        structure_final (Atoms): The final structure
        n_images (int): Number of structures to interpolate
    Output attributes:
        initial_positions (list/numpy.ndarray): if initial_positions is None, a list of (n_images) positions
            interpolated between the positions of the initial and final structures. Else, initial_positions
    """

    def __init__(self, name=None):
        super(InitialPositions, self).__init__(name=name)
        self.input.default.initial_positions = None

    def command(self, initial_positions, structure_initial, structure_final, n_images):

        if initial_positions is None:
            pos_i = structure_initial.positions
            pos_f = structure_final.positions
            displacement = structure_initial.find_mic(pos_f - pos_i)

            initial_positions = []
            for n, mix in enumerate(np.linspace(0, 1, n_images)):
                initial_positions.append(pos_i + (mix * displacement))

        else:
            if len(initial_positions) != n_images:
                raise TypeError("Length of positions is not the same as n_images!")

        return {
            'initial_positions': initial_positions
        }


class HarmonicHamiltonian(PrimitiveVertex):
    """
    Treat the atoms in the structure as harmonic oscillators and calculate the forces on each atom, and the total
        potential energy of the structure. If the spring constant is specified, the atoms act as Einstein atoms
        (independent of each other). If the Hessian / force constant matrix is specified, the atoms act as Debye
        atoms.
    Input attributes:
        positions (numpy.ndarray): Current positions of the atoms.
        reference_positions (numpy.ndarray): Equilibrium positions of the atoms.
        spring_constant (float): A single spring / force constant that is used to compute the restoring forces
            on each atom. (Default is 1.)
        force_constants (NxNx3x3 matrix): The Hessian matrix, obtained from, for ex. Phonopy. (Default is None, treat
            the atoms as independent harmonic oscillators (Einstein atoms).
        structure (Atoms): The reference structure.
        mask (numpy.array): Which of the atoms to consider. The other atoms are ignored. (Default is None, consider
            all atoms.)
        eq_energy (float): The minimized potential energy of the static (expanded) structure. (Default is None.)
    Output attributes:
        energy_pot (float): The harmonic potential energy.
        forces (numpy.ndarray): The harmonic forces.
    """

    def __init__(self, name=None):
        super(HarmonicHamiltonian, self).__init__(name=name)
        id_ = self.input.default
        id_.spring_constant = None
        id_.force_constants = None

    def command(self, positions, reference_positions, structure, spring_constant=None, force_constants=None,
                mask=None, eq_energy=None):

        dr = structure.find_mic(positions - reference_positions)
        if spring_constant is not None and force_constants is None:
            if mask is not None:
                forces = -spring_constant * dr[mask]
                energy = -0.5 * np.dot(dr[mask], forces)
            else:
                forces = -spring_constant * dr
                energy = -0.5 * np.tensordot(dr, forces)

        elif force_constants is not None and spring_constant is None:
            transformed_force_constants = self.transform_force_constants(force_constants, len(structure.positions))
            transformed_displacements = self.transform_displacements(dr)
            transformed_forces = -np.dot(transformed_force_constants, transformed_displacements)
            retransformed_forces = self.retransform_forces(transformed_forces, dr)
            if mask is not None:
                forces = retransformed_forces[mask]
                energy = -0.5 * np.dot(forces, dr[mask])
            else:
                forces = retransformed_forces
                energy = -0.5 * np.tensordot(forces, dr)

        else:
            raise TypeError('Please specify either a spring constant or the force constant matrix')

        if eq_energy is not None:
            energy += eq_energy

        return {
            'forces': forces,
            'energy_pot': energy
        }

    @staticmethod
    def transform_force_constants(force_constants, n_atoms):
        force_shape = np.shape(force_constants)
        if force_shape[2] == 3 and force_shape[3] == 3:
            force_reshape = force_shape[0] * force_shape[2]
            transformed_force_constants = np.transpose(
                force_constants,
                (0, 2, 1, 3)
            ).reshape((force_reshape, force_reshape))
        elif force_shape[1] == 3 and force_shape[3] == 3:
            transformed_force_constants = np.array(force_constants).reshape(3 * n_atoms, 3 * n_atoms)

        return transformed_force_constants

    @staticmethod
    def transform_displacements(displacements):
        return displacements.reshape(displacements.shape[0] * displacements.shape[1])

    @staticmethod
    def retransform_forces(transformed_forces, displacements):
        return transformed_forces.reshape(displacements.shape)


class LangevinThermostat(PrimitiveVertex):
    """
    Calculates the necessary forces for a Langevin thermostat based on drag and a random kick.
    Input dictionary:
        velocities (numpy.ndarray): The per-atom velocities in angstroms/fs.
        masses (numpy.ndarray): The per-atom masses in atomic mass units.
        temperature (float): The target temperature. (Default is 0 K, which just applies a drag force.)
        damping_timescale (float): Damping timescale. (Default is 100 fs.)
        time_step (float): MD time step in fs. (Default is 1 fs.)
        fix_com (bool): Whether to ensure the net force is zero. (Default is True.)
    Output dictionary:
        forces (numpy.ndarray): The per-atom forces from the thermostat.
    TODO: Make a version that uses uniform random numbers like Lammps does (for speed)
    """

    def __init__(self, name=None):
        super(LangevinThermostat, self).__init__(name=name)
        id_ = self.input.default
        id_.temperature = 0.
        id_.damping_timescale = 100.
        id_.time_step = 1.
        id_.fix_com = True

    def command(self, velocities, masses, temperature, damping_timescale, time_step, fix_com):
        # Ensure that masses are a commensurate shape
        masses = np.array(masses)[:, np.newaxis]
        gamma = masses / damping_timescale
        np.random.seed()
        noise = np.sqrt(2 * (gamma / time_step) * KB * temperature) * np.random.randn(*velocities.shape)
        drag = -gamma * velocities
        thermostat_forces = noise + drag

        if fix_com:  # Apply zero net force
            thermostat_forces -= np.mean(noise, axis=0)

        return {
            'forces': thermostat_forces
        }


class Max(PrimitiveVertex):
    """
    Numpy's amax (except no `out` option). Docstrings directly copied from the `numpy docs`_. The `initial`
    field is renamed `initial_val` to not conflict with the input dictionary.
    _`numpy docs`: https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html
    Input attributes:
        a (array_like): Input data.
        axis (None/int/tuple of ints): Axis or axes along which to operate. By default, flattened input is used. If
            this is a tuple of ints, the maximum is selected over multiple axes, instead of a single axis or all the
            axes as before.
        keepdims (bool): If this is set to True, the axes which are reduced are left in the result as dimensions with
            size one. With this option, the result will broadcast correctly against the input array. If the default
            value is passed, then *keepdims* will not be passed through to the `amax` method of sub-classes of
            `ndarray`, however any non-default value will be. If the sub-class’ method does not implement *keepdims*
            any exceptions will be raised.
    Output attributes:
        amax (numpy.ndarray/scalar): Maximum of *a*. If axis is None, the result is a scalar value. If axis is given,
            the result is an array of dimension `a.ndim - 1`.
    Note: This misses the new argument `initial` in the latest version of Numpy.
    """

    def __init__(self, name=None):
        super(Max, self).__init__(name=name)
        id_ = self.input.default
        id_.axis = None
        id_.keepdims = False
        id_.initial_val = None

    def command(self, a, axis, keepdims, initial_val):
        return {
            'amax': np.amax(a, axis=axis, keepdims=keepdims)
        }


class NEBForces(PrimitiveVertex):
    """
    Given a list of positions, forces, and energies for each image along some transition, calculates the tangent
        direction along the transition path for each image and returns new forces which have their original value
        perpendicular to this tangent, and 'spring' forces parallel to the tangent. Endpoints forces are set to
        zero, so start with relaxed endpoint structures.
    Note: All images must use the same cell and contain the same number of atoms.
    Input attributes:
        positions (list/numpy.ndarray): The positions of the images. Each element should have the same shape, and
            the order of the images is relevant!
        energies (list/numpy.ndarray): The potential energies associated with each set of positions. (Not always
            needed.)
        forces (list/numpy.ndarray): The forces from the underlying energy landscape on which the NEB is being run.
            Must have the same shape as `positions`. (Not always needed.)
        structure (Atoms): The reference structure.
        spring_constant (float): Spring force between atoms in adjacent images. (Default is 1.0 eV/angstrom^2.)
        tangent_style ('plain'/'improved'/'upwinding'): How to calculate the image tangent in 3N-dimensional space.
            (Default is 'upwinding', which requires energies to be set.)
        use_climbing_image (bool): Whether to replace the force with one that climbs along the tangent direction for
            the job with the highest energy. (Default is True, which requires energies and a list of forces to
            be set.)
        smoothing (float): Strength of the smoothing spring when consecutive images form an angle. (Default is None,
            do not apply such a force.)
    Output attributes:
        forces (list): The forces after applying nudged elastic band method.
    TODO:
        Add references to papers (Jonsson and others)
        Implement Sheppard's equations for variable cell shape
    """

    def __init__(self, name=None):
        super(NEBForces, self).__init__(name=name)
        id_ = self.input.default
        id_.spring_constant = 1.
        id_.tangent_style = 'upwinding'
        id_.use_climbing_image = True
        id_.smoothing = None

    def command(self, positions, energies, forces, structure, spring_constant, tangent_style, smoothing,
                use_climbing_image):

        if use_climbing_image:
            climbing_image_index = np.argmax(energies)
        else:
            climbing_image_index = None

        n_images = len(positions)
        neb_forces = []
        for i, pos in enumerate(positions):
            if i == 0 or i == n_images - 1:
                neb_forces.append(np.zeros(pos.shape))
                continue

            # Otherwise calculate the spring forces
            pos_left = positions[i - 1]
            pos_right = positions[i + 1]

            # Get displacement to adjacent images
            dr_left = structure.find_mic(pos - pos_left)
            dr_right = structure.find_mic(pos_right - pos)

            # Get unit vectors to adjacent images
            tau_left = dr_left / np.linalg.norm(dr_left)
            tau_right = dr_right / np.linalg.norm(dr_right)

            # Calculate the NEB tangent vector
            if tangent_style == 'plain':
                tau = self.normalize(dr_right + dr_left)
            elif tangent_style == 'improved':
                tau = self.normalize(tau_left + tau_right)
            elif tangent_style == 'upwinding':
                en_left = energies[i - 1]
                en = energies[i]
                en_right = energies[i + 1]
                tau = self._get_upwinding_tau(tau_left, tau_right, en_left, en, en_right)
            else:
                # This branch should be inaccessible due to the setter...
                raise KeyError("No such tangent_style: " + str(tangent_style))

            if smoothing is not None:
                force_smoothing = (self.saturating_angle_control(dr_left, dr_right) *
                                   smoothing * (dr_right - dr_left))
            else:
                force_smoothing = 0

            # Decompose the original forces
            input_force = forces[i]
            input_force_parallel = np.sum(input_force * tau) * tau
            input_force_perpendicular = input_force - input_force_parallel

            # If climbing image is activated, push the highest energy image up along tau while
            # relaxing otherwise
            if use_climbing_image and i == climbing_image_index:
                neb_forces.append(input_force_perpendicular - input_force_parallel)
            else:
                dr_mag = np.linalg.norm(dr_right) - np.linalg.norm(dr_left)
                force_spring = spring_constant * dr_mag * tau
                neb_forces.append(input_force_perpendicular + force_spring + force_smoothing)

        return {
            'forces': neb_forces
        }

    def _get_upwinding_tau(self, tau_left, tau_right, en_left, en, en_right):
        """
        Take direction to the higher-energy image.
        """
        # Find the relative energies
        den_left = en - en_left
        den_right = en - en_right

        # Create booleans to describe the energetic position of the job
        high_left = en_left > en_right
        high_right = en_right > en_left
        extrema = (en_right > en < en_left) or (en_right < en > en_left)

        den_mag_min = min(abs(den_left), abs(den_right))
        den_mag_max = max(abs(den_left), abs(den_right))
        # Calculate the NEB direction
        tau = tau_left * (den_mag_max * high_left + extrema * den_mag_min * high_right) + \
              tau_right * (den_mag_max * high_right + extrema * den_mag_min * high_left)
        return self.normalize(tau)

    @staticmethod
    def normalize(vec):
        l1 = np.linalg.norm(vec)
        if l1 == 0:
            return vec
        return vec / l1

    @staticmethod
    def saturating_angle_control(dr_left, dr_right):
        """
        Return 0 when displacement to left and right neighbours is parallel, and run smoothly to 1
        then saturate at this value when these displacement vectors form an acute angle.
        """
        dr_left_mag = np.linalg.norm(dr_left)
        dr_right_mag = np.linalg.norm(dr_right)
        cos_angle = np.sum(dr_left * dr_right) / (dr_left_mag * dr_right_mag)
        return 0.5 * (1 + np.cos(np.pi * cos_angle))


class Norm(PrimitiveVertex):
    """
    Numpy's linalg norm. Docstrings directly copied from the `numpy docs`_.
    _`numpy docs`: https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html
    Input attributes:
        x (array_like): Input array. If axis is None, x must be 1-D or 2-D.
        ord (non-zero int/inf/-inf/'fro'/'nuc'): Order of the norm. inf means numpy’s inf object. (Default is 2).
        axis (int/2-tuple of ints/None): If axis is an integer, it specifies the axis of x along which to compute the
            vector norms. If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of
            these matrices are computed. If axis is None then either a vector norm (when x is 1-D) or a matrix norm
            (when x is 2-D) is returned. (Default is None.)
        keepdims (bool): If this is set to True, the axes which are normed over are left in the result as dimensions
            with size one. With this option the result will broadcast correctly against the original x. (Default is
            False)
    Output attributes:
        n (float/numpy.ndarray): Norm of the matrix or vector(s).
    """

    def __init__(self, name=None):
        super(Norm, self).__init__(name=name)
        id_ = self.input.default
        id_.ord = 2
        id_.axis = None
        id_.keepdims = False

    def command(self, x, ord, axis, keepdims):
        return {
            'n': np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        }


class Overwrite(PrimitiveVertex):
    """
    Overwrite particular entries of an array with new values.
    Input attributes:
        target (numpy.ndarray): The target array.
        mask (numpy.array): The indices of the target that will be replaced.
        new_values (numpy.ndarray): The targeted indices will be replaced by these values.
    Output attributes:
        overwritten (numpy.ndarray): The overwritten array.
    """

    def command(self, target, mask, new_values):
        overwritten = np.array(target)
        overwritten[mask] = new_values
        return {
            'overwritten': overwritten
        }


class RandomVelocity(PrimitiveVertex):
    """
    Generates a set of random velocities which (on average) have give the requested temperature.
        Hard-coded for 3D systems.
    Input attributes:
        temperature (float): The temperature of the velocities (in Kelvin).
        masses (numpy.ndarray/list): The masses of the atoms.
        overheat_fraction (float): The fraction by which to overheat the initial velocities. This can be useful for
            more quickly equilibrating a system whose initial structure is its fully relaxed positions -- in which
            case equipartition of energy tells us that the kinetic energy should be initialized to double the
            desired value. (Default is 2.0, assume energy equipartition is a good idea.)
    Output attributes:
        velocities (numpy.ndarray): Per-atom velocities.
        energy_kin (float): Total kinetic energy of all atoms.
        n_atoms (int): Number of atoms.
    """

    def __init__(self, name=None):
        super(RandomVelocity, self).__init__(name=name)
        self.input.default.overheat_fraction = 2.

    def command(self, temperature, masses, overheat_fraction):
        masses = np.array(masses)[:, np.newaxis]
        vel_scale = np.sqrt(EV_TO_U_ANGSQ_PER_FSSQ * KB * temperature / masses) * np.sqrt(overheat_fraction)
        np.random.seed(0)
        vel_dir = np.random.randn(len(masses), 3)
        vel = vel_scale * vel_dir
        vel -= np.mean(vel, axis=0)
        energy_kin = 0.5 * np.sum(masses * vel * vel) * U_ANGSQ_PER_FSSQ_TO_EV
        return {
            'velocities': vel,
            'energy_kin': energy_kin,
            'n_atoms': len(vel)
        }


class SphereReflection(PrimitiveVertex):
    """
    Checks whether each atom in a structure is within a cutoff radius of its reference position; if not, reverts
        the positions and velocities of *all atoms* to an earlier state and reverses those earlier velocities.
        Forces from the current and previous time step can be provided optionally.
    Input attributes:
        reference_positions (numpy.ndarray): The reference positions to check the distances from.
        positions (numpy.ndarray): The positions to check.
        velocities (numpy.ndarray): The velocities corresponding to the positions at this time.
        previous_positions (numpy.ndarray): The previous positions to revert to.
        previous_velocities (numpy.ndarray): The previous velocities to revert to and reverse.
        structure (Atoms): The reference structure.
        cutoff_distance (float): The cutoff distance from the reference position to trigger reflection.
        use_reflection (boolean): Turn on or off `SphereReflection`
        total_steps (int): The total number of times `SphereReflection` is called so far.
    Output attributes:
        positions (numpy.ndarray): The (possibly reverted) positions.
        velocities (numpy.ndarray): The (possibly reverted and reversed) velocities.
        reflected_mask (bool): Whether or not reflection occurred.
        total_steps (int): The total number of times `SphereReflection` is called.
    """

    def __init__(self, name=None):
        super(SphereReflection, self).__init__(name=name)
        id_ = self.input.default
        id_.use_reflection = True
        id_.total_steps = 0

    def command(self, reference_positions, cutoff_distance, positions, velocities, previous_positions,
                previous_velocities, structure, use_reflection, total_steps):
        total_steps += 1
        distance = np.linalg.norm(structure.find_mic(reference_positions - positions), axis=-1)
        is_at_home = (distance < cutoff_distance)[:, np.newaxis]
        if np.all(is_at_home) or use_reflection is False:
            return {
                'positions': positions,
                'velocities': velocities,
                'reflected': False,
                'total_steps': total_steps
            }
        else:
            return {
                'positions': previous_positions,
                'velocities': -previous_velocities,
                'reflected': True,
                'total_steps': total_steps
            }


class SphereReflectionPerAtom(PrimitiveVertex):
    """
    Checks whether each atom in a structure is within a cutoff radius of its reference position; if not, reverts
        the positions and velocities of *the violating atoms* to an earlier state and reverses those earlier
        velocities. The remaining atoms are unaffected.
    Input attributes:
        reference_positions (numpy.ndarray): The reference positions to check the distances from.
        positions (numpy.ndarray): The positions to check.
        velocities (numpy.ndarray): The velocities corresponding to the positions at this time.
        previous_positions (numpy.ndarray): The previous positions to revert to.
        previous_velocities (numpy.ndarray): The previous velocities to revert to and reverse.
        structure (Atoms): The reference structure.
        cutoff_distance (float): The cutoff distance from the reference position to trigger reflection.
        use_reflection (boolean): Turn on or off `SphereReflectionPerAtom`
        total_steps (int): The total number of times `SphereReflectionPerAtom` is called so far.
    Output attributes:
        positions (numpy.ndarray): The (possibly reverted) positions.
        velocities (numpy.ndarray): The (possibly reverted and reversed) velocities.
        reflected_mask (numpy.ndarray): A boolean mask that is true for each atom who was reflected.
        total_steps (int): The total number of times `SphereReflectionPerAtom` is called.
    """

    def __init__(self, name=None):
        super(SphereReflectionPerAtom, self).__init__(name=name)
        id_ = self.input.default
        id_.use_reflection = True
        id_.total_steps = 0

    def command(self, reference_positions, cutoff_distance, positions, velocities, previous_positions,
                previous_velocities, structure, use_reflection, total_steps):
        total_steps += 1
        if use_reflection:
            distance = np.linalg.norm(structure.find_mic(reference_positions - positions), axis=-1)
            is_at_home = (distance < cutoff_distance)[:, np.newaxis]
            is_away = 1 - is_at_home
        else:
            is_at_home = np.ones(len(reference_positions))
            is_away = 1 - is_at_home

        return {
            'positions': is_at_home * positions + is_away * previous_positions,
            'velocities': is_at_home * velocities + is_away * -previous_velocities,
            'reflected': is_away.astype(bool).flatten(),
            'total_steps': total_steps
        }


class CutoffDistance(PrimitiveVertex):
    """
    Compute the cutoff distance for SphereReflection.
    Input attributes:
        structure (Atoms): The reference structure.
        cutoff_factor (float): The cutoff is obtained by taking the first nearest neighbor distance and multiplying
            it by the cutoff factor. A default value of 0.4 is chosen, because taking a cutoff factor of ~0.5
            sometimes let certain reflections off the hook, and we do not want that to happen.
    Output attributes:
        cutoff_distance (float): The cutoff distance.
    """

    def command(self, structure, cutoff_factor=0.5):
        try:
            nn_list = structure.get_neighbors(num_neighbors=1)
        except ValueError:
            nn_list = structure.get_neighbors(num_neighbors=1)

        cutoff_distance = nn_list.distances[0] * cutoff_factor

        return {
            'cutoff_distance': cutoff_distance[-1]
        }


class Slice(PrimitiveVertex):
    """
    Slices off the masked positions from the input vector.
    Input attributes:
        vector (numpy.ndarray): The input vector.
        mask (numpy.ndarray): The integer ids shared by both the old and new structure.
        ensure_interable mask (bool):
    Output attributes:
        sliced (numpy.ndarray): The sliced vector.
    """

    def __init__(self, name=None):
        super(Slice, self).__init__(name=name)
        self.input.default.ensure_iterable_mask = False

    def command(self, vector, mask, ensure_iterable_mask):
        if ensure_iterable_mask:
            mask = ensure_iterable(mask)
        return {
            'sliced': vector[mask]
        }


class Transpose(PrimitiveVertex):
    """
    Transposes the input matrix.
    Input attributes:
        matrix (numpy.ndarray): The input matrix.
    Output attributes:
        matrix_transpose (numpy.ndarray): The transposed matrix.
    """

    def command(self, matrix):
        transpose = list(map(list, zip(*matrix)))
        return {
            'matrix_transpose': transpose
        }


class VerletParent(PrimitiveVertex, ABC):
    """
    A parent class for holding code which is shared between both position and velocity updates in two-step
        Velocity Verlet.
    Input attributes:
        time_step (float): MD time step in fs. (Default is 1 fs.)
        temperature (float): The target temperature. (Default is None, no thermostat is used.)
        damping_timescale (float): Damping timescale in fs. (Default is None, no thermostat is used.)
    TODO: VerletVelocityUpdate should *always* have its velocity input wired to the velocity outupt of
          VerletPositionUpdate. This implies to me that we need some structure *other* than two fully independent
          nodes. It would also be nice to syncronize, e.g. the thermostat and timestep input which is also the same
          for both. However, I haven't figured out how to do that in the confines of the current graph traversal
          and hdf5 setup.
    """

    def __init__(self, name=None):
        super(VerletParent, self).__init__(name=name)
        id_ = self.input.default
        id_.time_step = 1.
        id_.temperature = None
        id_.temperature_damping_timescale = None

    @abstractmethod
    def command(self, *arg, **kwargs):
        pass

    @staticmethod
    def reshape_masses(masses):
        if len(np.array(masses).shape) == 1:
            masses = np.array(masses)[:, np.newaxis]
        return masses

    @staticmethod
    def convert_to_acceleration(forces, masses):
        return forces * EV_TO_U_ANGSQ_PER_FSSQ / masses

    @staticmethod
    def langevin_delta_v(temperature, time_step, masses, damping_timescale, velocities):
        """
        Velocity changes due to the Langevin thermostat.
        Args:
            temperature (float): The target temperature in K.
            time_step (float): The MD time step in fs.
            masses (numpy.ndarray): Per-atom masses in u with a shape (N_atoms, 1).
            damping_timescale (float): The characteristic timescale of the thermostat in fs.
            velocities (numpy.ndarray): Per-atom velocities in angstrom/fs.
        Returns:
            (numpy.ndarray): Per atom accelerations to use for changing velocities.
        """
        drag = -0.5 * time_step * velocities / damping_timescale
        np.random.seed()
        noise = np.sqrt(EV_TO_U_ANGSQ_PER_FSSQ * KB * temperature * time_step / (masses * damping_timescale)) \
                * np.random.randn(*velocities.shape)
        noise -= np.mean(noise, axis=0)
        return drag + noise


class VerletPositionUpdate(VerletParent):
    """
    First half of Velocity Verlet, where positions are updated and velocities are set to their half-step value.
    Input attributes:
        positions (numpy.ndarray): The per-atom positions in angstroms.
        velocities (numpy.ndarray): The per-atom velocities in angstroms/fs.
        forces (numpy.ndarray): The per-atom forces in eV/angstroms.
        masses (numpy.ndarray): The per-atom masses in atomic mass units.
        time_step (float): MD time step in fs. (Default is 1 fs.)
        temperature (float): The target temperature. (Default is None, no thermostat is used.)
        damping_timescale (float): Damping timescale in fs. (Default is None, no thermostat is used.)
    Output attributes:
        positions (numpy.ndarray): The new positions on time step in the future.
        velocities (numpy.ndarray): The new velocities *half* a time step in the future.
    """

    def command(self, positions, velocities, forces, masses, time_step, temperature, temperature_damping_timescale):
        masses = self.reshape_masses(masses)
        acceleration = self.convert_to_acceleration(forces, masses)
        vel_half = velocities + 0.5 * acceleration * time_step
        if temperature_damping_timescale is not None:
            vel_half += self.langevin_delta_v(
                temperature,
                time_step,
                masses,
                temperature_damping_timescale,
                velocities
            )
        pos_step = positions + vel_half * time_step

        return {
            'positions': pos_step,
            'velocities': vel_half
        }


class VerletVelocityUpdate(VerletParent):
    """
    Second half of Velocity Verlet, where velocities are updated. Forces should be updated between the position
        and velocity updates
    Input attributes:
        velocities (numpy.ndarray): The per-atom velocities in angstroms/fs. These should be the half-step
            velocities output by `VerletPositionUpdate`.
        forces (numpy.ndarray): The per-atom forces in eV/angstroms. These should be updated since the last call
            of `VerletPositionUpdate`.
        masses (numpy.ndarray): The per-atom masses in atomic mass units.
        time_step (float): MD time step in fs. (Default is 1 fs.)
        temperature (float): The target temperature. (Default is None, no thermostat is used.)
        damping_timescale (float): Damping timescale in fs. (Default is None, no thermostat is used.)
    Output attributes:
        velocities (numpy.ndarray): The new velocities *half* a time step in the future.
        energy_kin (float): The total kinetic energy of the system in eV.
        instant_temperature (float): The instantaneous temperature, obtained from the total kinetic energy.
    """

    def command(self, velocities, forces, masses, time_step, temperature, temperature_damping_timescale):
        masses = self.reshape_masses(masses)
        acceleration = self.convert_to_acceleration(forces, masses)

        vel_step = velocities + 0.5 * acceleration * time_step
        if temperature_damping_timescale is not None:
            vel_step += self.langevin_delta_v(
                temperature,
                time_step,
                masses,
                temperature_damping_timescale,
                velocities
            )
        kinetic_energy = 0.5 * np.sum(masses * vel_step * vel_step) / EV_TO_U_ANGSQ_PER_FSSQ
        instant_temperature = (kinetic_energy * 2) / (3 * KB * len(velocities))

        return {
            'velocities': vel_step,
            'energy_kin': kinetic_energy,
            'instant_temperature': instant_temperature
        }


class VoronoiReflection(PrimitiveVertex):
    """
    Checks whether each atom in a structure is closest to its own reference site; if not, reverts the positions
        and velocities to an earlier state and reverses those earlier velocities.
    Input attributes:
        reference_positions (numpy.ndarray): The reference positions to check the distances from.
        positions (numpy.ndarray): The positions to check.
        velocities (numpy.ndarray): The velocities to check.
        previous_positions (numpy.ndarray): The previous positions to revert to.
        previous_velocities (numpy.ndarray): The previous velocities to revert to and reverse.
        pbc (numpy.ndarray/list): Three booleans declaring which dimensions have periodic boundary conditions for
            finding the minimum distance convention.
        cell (numpy.ndarray): The 3x3 cell vectors for pbcs.
    Output attributes:
        positions (numpy.ndarray): The (possibly reverted) positions.
        velocities (numpy.ndarray): The (possibly reverted and reversed) velocities.
        reflected_mask (numpy.ndarray): A boolean mask that is true for each atom who was reflected.
    WARNING: Outdated.
    """

    def __init__(self, name=None):
        super(VoronoiReflection, self).__init__(name=name)

    def command(self, reference_positions, positions, velocities, previous_positions, previous_velocities, pbc, cell):
        _, distance_matrix = get_distances(p1=reference_positions, p2=positions, cell=cell, pbc=pbc)
        closest_reference = np.argmin(distance_matrix, axis=0)
        is_at_home = (closest_reference == np.arange(len(positions)))[:, np.newaxis]
        is_away = 1 - is_at_home
        return {
            'positions': is_at_home * positions + is_away * previous_positions,
            'velocities': is_at_home * velocities + is_away * -previous_velocities,
            'reflected': is_away.astype(bool).flatten()
        }


class WeightedSum(PrimitiveVertex):
    """
    Given a list of vectors of with the same shape, calculates a weighted sum of the vectors. By default the
        weights are the inverse of the number of elements and the sum is just the mean.
    Input attributes:
        vectors (list/numpy.ndarray): The vectors to sum. (Masked) vectors must all be of the same length. If the
            the vectors are already in a numpy array, the 0th index should determine the vector.
        weights (list/numpy.ndarray): A 1D list of coefficients (floats) with the same length as `vectors`.
            (Default is None, which gives the simple mean.)
        masks (list): If not None, a mask must be passed for each vector to extract a sub-vector. The resulting
            collection of vectors and sub-vectors must all have the same length. If the mask for a given vector
            is `None`, all elements are retained. Otherwise the mask must be integer-like or boolean. (Default is
            None, do not mask any of the vectors.)
    Output attributes:
        weighted_sum (numpy.ndarray): The weighted sum, having the same shape as a (masked) input vector.
    """

    def __init__(self, name=None):
        super(WeightedSum, self).__init__(name=name)
        self.input.default.weights = None
        self.input.default.masks = None

    def command(self, vectors, weights, masks):
        # Handle weight defaults
        if weights is None:
            n = len(vectors)
            weights = np.ones() / n
        elif len(weights) != len(vectors):
            raise ValueError('The length of the weights and vectors must be comensurate, but were {} and {}'.format(
                len(weights), len(vectors)))

        # Mask vectors
        if masks is not None:
            for n, mask in enumerate(masks):
                if mask is not None:
                    if isinstance(mask[0], bool) or isinstance(mask[0], np.bool_):
                        mask = np.array(mask, dtype=bool)
                    else:
                        mask = np.array(mask, dtype=int)
                    vectors[n] = vectors[n][mask]

        # Prepare vectors for numpy operation
        vectors = np.array(vectors)

        # Dot vectors
        weighted_sum = np.tensordot(weights, vectors, axes=1)

        # If dot reduces to a single value, recast
        if len(weighted_sum.shape) == 0:
            weighted_sum = float(weighted_sum)

        return {
            'weighted_sum': weighted_sum
        }


class WelfordOnline(PrimitiveVertex):
    """
    Computes the cumulative mean and standard deviation.
    Note: The standard deviation calculated is for the population (ddof=0). For the sample (ddof=1) it would
        to be extended.
    Input attributes:
        sample (float/numpy.ndarray): The new sample. (Default is None.)
        mean (float/numpy.ndarray): The mean so far. (Default is None.)
        std (float/numpy.ndarray): The standard deviation so far. (Default is None.)
        n_samples (int): How many samples were used to calculate the existing `mean` and `std`.
    Output attributes:
        mean (float/numpy.ndarray): The new mean.
        std (float/numpy.ndarray): The new standard deviation.
        n_samples (int): The new number of samples.
    """

    def __init__(self, name=None):
        super(WelfordOnline, self).__init__(name=name)
        id_ = self.input.default
        id_.mean = None
        id_.std = None
        id_.n_samples = None

        op = Pointer(self.output)
        self.input.mean = op.mean[-1]
        self.input.std = op.std[-1]
        self.input.n_samples = op.n_samples[-1]

    def command(self, sample, mean, std, n_samples):
        if n_samples is None:
            new_mean = sample
            new_std = 0
            n_samples = 0
        else:
            new_mean, new_std = welford_online(sample, mean, std, n_samples)
        return {
            'mean': new_mean,
            'std': new_std,
            'n_samples': n_samples + 1
        }


class FEPExponential(PrimitiveVertex):
    """
    Compute the free energy perturbation exponential difference.
    Input attributes:
        u_diff (float): The energy difference between system B and system A.
        delta_lambda (float): The delta for the lambdas for the two systems A and B.
        temperature (float): The instantaneous temperature.
    Output attributes:
        exponential_difference (float): The exponential difference.
    """

    def command(self, u_diff, delta_lambda, temperature):
        return {
            'exponential_difference': np.exp(-u_diff * delta_lambda / (KB * temperature))
        }


class TILDPostProcess(PrimitiveVertex):
    """
    Post processing for the Harmonic and Vacancy TILD protocols, to remove the necessity to load interactive
        jobs after they have been closed, once the protocol has been executed.
    Input attributes:
        lambda_pairs (numpy.ndarray): The (`n_lambdas`, 2)-shaped array of mixing pairs.
        tild_mean (list): The mean of the computed integration points.
        tild_std (list): The standard deviation of the computed integration points.
        fep_exp_mean (list): The mean of the free energy perturbation exponential differences.
        fep_exp_mean (list): The standard deviation of the free energy perturbation exponential differences.
        temperature (float): The simulated temperature in K.
        n_samples (int): Number of samples used to calculate the means.
    Output attributes:
        tild_free_energy_mean (float): The mean calculated via thermodynamic integration.
        tild_free_energy_std (float): The standard deviation calculated via thermodynamic integration.
        tild_free_energy_se (float): The standard error calculated via thermodynamic integration.
        fep_free_energy_mean (float): The mean calculated via free energy perturbation.
        fep_free_energy_std (float): The standard deviation calculated via free energy perturbation.
        fep_free_energy_se (float): The standard error calculated via free energy perturbation.
    """

    def command(self, lambda_pairs, tild_mean, tild_std, fep_exp_mean, fep_exp_std, temperature, n_samples):

        tild_fe_mean, tild_fe_std, tild_fe_se = self.get_tild_free_energy(lambda_pairs, tild_mean, tild_std,
                                                                          n_samples)
        fep_fe_mean, fep_fe_std, fep_fe_se = self.get_fep_free_energy(fep_exp_mean, fep_exp_std, n_samples,
                                                                      temperature)
        return {
            'tild_free_energy_mean': tild_fe_mean,
            'tild_free_energy_std': tild_fe_std,
            'tild_free_energy_se': tild_fe_se,
            'fep_free_energy_mean': fep_fe_mean,
            'fep_free_energy_std': fep_fe_std,
            'fep_free_energy_se': fep_fe_se

        }

    @staticmethod
    def get_tild_free_energy(lambda_pairs, tild_mean, tild_std, n_samples):
        y = unumpy.uarray(tild_mean, tild_std)
        integral = simps(x=lambda_pairs[:, 0], y=y)
        mean = unumpy.nominal_values(integral)
        std = unumpy.std_devs(integral)
        tild_se = tild_std / np.sqrt(n_samples)
        y_se = unumpy.uarray(tild_mean, tild_se)
        integral_se = simps(x=lambda_pairs[:, 0], y=y_se)
        se = unumpy.std_devs(integral_se)

        return float(mean), float(std), float(se)

    @staticmethod
    def get_fep_free_energy(fep_exp_mean, fep_exp_std, n_samples, temperature):
        fep_exp_se = fep_exp_std / np.sqrt(n_samples)
        y = unumpy.uarray(fep_exp_mean, fep_exp_std)
        y_se = unumpy.uarray(fep_exp_mean, fep_exp_se)
        free_energy = 0
        free_energy_se = 0
        for (val, val_se) in zip(y, y_se):
            free_energy += -KB * temperature * unumpy.log(val)
            free_energy_se += -KB * temperature * unumpy.log(val_se)
        mean = unumpy.nominal_values(free_energy)
        std = unumpy.std_devs(free_energy)
        se = unumpy.std_devs(free_energy_se)

        return float(mean), float(std), float(se)


class BerendsenBarostat(PrimitiveVertex):
    """
    The Berendsen barostat which can be used for pressure control as elaborated in
    https://doi.org/10.1063/1.448118
    NOTE: Always use in conjunction with a thermostat, otherwise time integration will not be performed.
    The barostat only modifies the cell of the input structure, and scales the positions. Positions
    and velocities will only be updated by the thermostat.
    Input attributes:
        pressure (float): The pressure in GPa to be simulated (Default is None GPa)
        temperature (float): The temperature in K (Default is 0. K)
        box_pressures (numpy.ndarray): The pressure tensor in GPa generated per step by Lammps
        energy_kin (float): The kinetic energy of the system in eV (Default is None)
        time_step (float): MD time step in fs. (Default is 1 fs.)
        pressure_damping_timescale (float): Damping timescale in fs. (Default is None, no barostat is used.)
        compressibility (float): The compressibility of water in bar-1. More information here:
            http://www.sklogwiki.org/SklogWiki/index.php/Berendsen_barostat (Default is 4.57e-5 bar-1)
        structure (Atoms): The structure whose cell and positions are to be scaled.
        positions (numpy.ndarray): The updated positions from `VerletPositionUpdate`.
        previous_volume (float): The volume of the cell from the previous step in Ang3 (Default is None)
        pressure_style (string): 'isotorpic' or 'anisotropic'. (Default is 'anisotropic')
    Output attributes:
        pressure (float): The isotropic pressure in GPa
        volume (float): The volume of the cell in Ang3
        structure (Atoms): The scaled structure, corresponding to the simulated volume
        positions (numpy.ndarray): The scaled positions
    """

    def __init__(self, name=None):
        super(BerendsenBarostat, self).__init__(name=name)
        id_ = self.input.default
        id_.pressure = 0.
        id_.temperature = 0.
        id_.pressure_damping_timescale = 1000.
        id_.time_step = 1.
        id_.compressibility = 4.57e-5  # compressibility of water in bar^-1
        id_.pressure_style = 'isotropic'

    def command(self, pressure, temperature, box_pressure, energy_kin, time_step, positions,
                pressure_damping_timescale, compressibility, structure, previous_volume, pressure_style):

        if pressure_style != 'isotropic' and pressure_style != 'anisotropic':
            raise TypeError('style can only be \'isotropic\' or \'anisotropic\'')

        n_atoms = len(structure.positions)

        if previous_volume is None:
            previous_volume = structure.cell.diagonal().prod()

        if energy_kin is None:
            energy_kin = (3 / 2) * n_atoms * KB * temperature

        isotropic_pressure = np.trace(box_pressure) / 3  # pyiron stores pressure in GPa

        if pressure is None:
            new_structure = structure.copy()
            total_pressure = isotropic_pressure
        elif pressure is not None and pressure_style == 'isotropic':
            new_structure = structure.copy()
            new_structure.positions = positions
            first_term = ((2 * energy_kin) / (3 * previous_volume)) * EV_PER_ANGCUB_TO_GPA
            tau = (time_step / pressure_damping_timescale) * (compressibility / 3)
            total_pressure = first_term + isotropic_pressure  # GPa
            eta = 1 - (tau * (pressure - total_pressure) * GPA_TO_BAR)
            new_cell = new_structure.cell * eta
            new_structure.set_cell(new_cell, scale_atoms=True)
        elif pressure is not None and pressure_style == 'anisotropic':
            new_structure = structure.copy()
            new_structure.positions = positions
            first_term = ((2 * energy_kin) / (3 * previous_volume)) * EV_PER_ANGCUB_TO_GPA
            tau = (time_step / pressure_damping_timescale) * (compressibility / 3)
            total_pressure_x = first_term + box_pressure[0, 0]  # GPa
            eta_x = 1 - (tau * (pressure - total_pressure_x) * GPA_TO_BAR)
            total_pressure_y = first_term + box_pressure[1, 1]  # GPa
            eta_y = 1 - (tau * (pressure - total_pressure_y) * GPA_TO_BAR)
            total_pressure_z = first_term + box_pressure[2, 2]  # GPa
            eta_z = 1 - (tau * (pressure - total_pressure_z) * GPA_TO_BAR)

            old_cell = new_structure.cell
            new_cell = np.array([eta_x * old_cell[0],
                                 eta_y * old_cell[1],
                                 eta_z * old_cell[2]])

            new_structure.set_cell(new_cell, scale_atoms=True)
            total_pressure = np.mean([total_pressure_x, total_pressure_y, total_pressure_z])
        else:
            raise TypeError('Invalid value for pressure')

        return {
            'pressure': total_pressure,
            'structure': new_structure,
            'positions': new_structure.positions
        }


class ComputeFormationEnergy(PrimitiveVertex):
    """
    """

    def command(self, n_atoms, eq_energy, harm_to_inter_mean, harm_to_inter_std, harm_to_inter_se, inter_to_vac_mean,
                inter_to_vac_std, inter_to_vac_se):

        harm_to_inter_fe = unumpy.uarray(harm_to_inter_mean, harm_to_inter_std)
        harm_to_inter_fe_se = unumpy.uarray(harm_to_inter_mean, harm_to_inter_se)
        inter_to_vac_fe = unumpy.uarray(inter_to_vac_mean, inter_to_vac_std)
        inter_to_vac_fe_se = unumpy.uarray(inter_to_vac_mean, inter_to_vac_se)

        fe = ((eq_energy + harm_to_inter_fe) / n_atoms) + inter_to_vac_fe
        fe_se = ((eq_energy + harm_to_inter_fe_se) / n_atoms) + inter_to_vac_fe_se

        return {
            'formation_energy_mean': float(unumpy.nominal_values(fe)),
            'formation_energy_std': float(unumpy.std_devs(fe)),
            'formation_energy_se': float(unumpy.std_devs(fe_se))
        }