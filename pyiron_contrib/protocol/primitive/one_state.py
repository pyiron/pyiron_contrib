# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from pyiron_contrib.protocol.generic import PrimitiveVertex
from pyiron_contrib.protocol.utils import Pointer
import numpy as np
from pyiron.atomistics.job.interactive import GenericInteractive
from pyiron.lammps.lammps import LammpsInteractive
from scipy.constants import physical_constants
from ase.geometry import find_mic, get_distances  # TODO: Wrap things using pyiron functionality
from pyiron import Project
from pyiron_contrib.protocol.utils import ensure_iterable
from os.path import split
from abc import ABC, abstractmethod
from pyiron_contrib.protocol.math import welford_online

KB = physical_constants['Boltzmann constant in eV/K'][0]
EV_TO_U_ANGSQ_PER_FSSQ = 0.00964853322  # https://www.wolframalpha.com/input/?i=1+eV+in+u+*+%28angstrom%2Ffs%29%5E2
U_ANGSQ_PER_FSSQ_TO_EV = 1. / EV_TO_U_ANGSQ_PER_FSSQ

"""
Primitive vertices which have only one outbound execution edge.

TODO: Rely on pyiron units instead of hard-coding in scipy's Boltzmann constant in eV/K.
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
    Builds an array of mixing parameters [lambda, (1-lambda)].

    Input attributes:
        n_lambdas (int): How many mixing pairs to create.
        lambdas (list/numpy.ndarray): The individual lambda values to use for the first member of each pair.

    Note:
        Exactly one of `n_lambdas` or `lambdas` must be provided.

    Output attributes:
        lambda_pairs (numpy.ndarray): The (`n_lambdas`, 2)-shaped array of mixing pairs.
    """
    def __init__(self, name=None):
        super(BuildMixingPairs, self).__init__(name=name)
        self.input.default.n_lambdas = None
        self.input.default.custom_lambdas = None

    def command(self, n_lambdas, custom_lambdas):
        if custom_lambdas is not None:
            lambdas = np.array(custom_lambdas)
        else:
            lambdas = np.linspace(0, 1, num=n_lambdas)
        return {
            'lambda_pairs': np.array([lambdas, 1 - lambdas]).T
        }


class Counter(PrimitiveVertex):
    """
    Increments by one at each execution.

    Output attributes:
        n_counts (int): How many executions have passed. (Default is 0.)
    """
    def __init__(self, name=None):
        super(Counter, self).__init__(name=name)
        self.output.n_counts = [0]

    def command(self):
        return {
             'n_counts': self.output.n_counts[-1] + 1
         }


class Compute(PrimitiveVertex):

    def __init__(self, name=None):
        super(Compute, self).__init__(name=name)

    def command(self, *args, **kwargs):
        function_ = self.input.function
        return function_(*args, **kwargs)


class DeleteAtom(PrimitiveVertex):
    """
    Given a structure, deletes one of the atoms.

    Input attributes:
        structure (Atoms): The structure to delete an atom of.
        id (int): Which atom to delete.

    Output attributes:
        structure (Atoms): The new, modified structure.
        mask (numpy.ndarray): The integer ids shared by both the old and new structure.
    """
    def command(self, structure, id):
        vacancy_structure = structure.copy()
        vacancy_structure.pop(id)
        mask = np.delete(np.arange(len(structure)).astype(int), id)
        # mask = np.arange(len(structure)).astype(int)
        return {
            'structure': vacancy_structure,
            'mask': mask
        }


class ExternalHamiltonian(PrimitiveVertex):
    """
    Manages calls to an external interpreter (e.g. Lammps, Vasp, Sphinx...) to produce energies, forces, and possibly
    other properties.

    The collected output can be expanded beyond forces and energies (e.g. to magnetic properties or whatever else the
    interpreting code produces) by modifying the `interesting_keys` in the input. The property must have a corresponding
    interactive getter for this property.

    Input attributes:
        ref_job_full_path (string): The full path to the hdf5 file of the job to use as a reference template.
        structure (Atoms): Overwrites the reference job structure when provided. (Default is None, the reference job
            needs to have its structure set.)
        interesting_keys (list[str]): String codes for output properties of the underlying job to collect. (Default is
            ['forces', 'energy_pot'].)
        positions (numpy.ndarray): New positions to evaluate. (Not set by default, only necessary if positions are being
            updated.)
    """

    def __init__(self, name=None):
        super(ExternalHamiltonian, self).__init__(name=name)
        self.input.default.structure = None
        self.input.default.interesting_keys = ['forces', 'energy_pot']
        self.input.default.positions = None

        self._fast_lammps_mode = True  # Set to false only to intentionally be slow for comparison purposes
        self._job = None

    def command(self, ref_job_full_path, structure, interesting_keys, positions):
        if self._job is None:
            self._initialize(ref_job_full_path, structure)
        elif self._job.status == 'finished':
            # The interactive library needs to be reopened
            self._job.interactive_open()
            self._job.interactive_initialize_interface()

        if isinstance(self._job, LammpsInteractive) and self._fast_lammps_mode:
            # Run Lammps 'efficiently'
            if positions is not None:
                self._job.interactive_positions_setter(positions)
            self._job._interactive_lib_command(self._job._interactive_run_command)
        elif isinstance(self._job, GenericInteractive):
            # DFT codes are slow enough that we can run them the regular way and not care
            # Also we might intentionally run Lammps slowly for comparison purposes
            if positions is not None:
                self._job.structure.positions = positions
            self._job.calc_static()
            self._job.run()
        else:
            raise TypeError('Job of class {} is not compatible.'.format(self._job.__class__))

        return {key: self.get_interactive_value(key) for key in interesting_keys}

    def _initialize(self, ref_job_full_path, structure):
        loc = self.get_graph_location()
        name = 'job'
        project_path, ref_job_path = split(ref_job_full_path)
        pr = Project(path=project_path)
        sub_pr = pr.create_group(loc)
        # sub directory is necessary so jobs don't fight for the same `rewrite_hdf` space
        ref_job = pr.load(ref_job_path)
        job = ref_job.copy_template(project=sub_pr, new_job_name=name)
        if structure is not None:
            job.structure = structure

        if isinstance(job, GenericInteractive):
            job.interactive_open()

            if isinstance(job, LammpsInteractive) and self._fast_lammps_mode:
                # Note: This might be done by default at some point in LammpsInteractive, and could then be removed here
                job.interactive_flush_frequency = 10**10
                job.interactive_write_frequency = 10**10
                self._disable_lmp_output = True

            job.calc_static()
            job.run(run_again=True)
            # job.interactive_initialize_interface()
            # TODO: Running is fine for Lammps, but wasteful for DFT codes! Get the much cheaper interface
            #  initialization working -- right now it throws a (passive) TypeError due to database issues
        else:
            raise TypeError('Job of class {} is not compatible.'.format(ref_job.__class__))
        self._job = job
        self._job_name = name

    def get_interactive_value(self, key):
        if key == 'positions':
            val = np.array(self._job.interactive_positions_getter())
        elif key == 'forces':
            val = np.array(self._job.interactive_forces_getter())
        elif key == 'energy_pot':
            val = self._job.interactive_energy_pot_getter()
            print(val)
        elif key == 'cells':
            val = np.array(self._job.interactive_cells_getter())
        else:
            raise NotImplementedError
        return val

    def finish(self):
        super(ExternalHamiltonian, self).finish()
        if self._job is not None:
            self._job.interactive_close()

    def parallel_setup(self):
        super(ExternalHamiltonian, self).parallel_setup()
        if self._job is None:
            self._initialize(self.input.ref_job_full_path, self.input.structure)
        elif self._job.status == 'finished':
            # The interactive library needs to be reopened
            self._job.interactive_open()
            self._job.interactive_initialize_interface()

    def to_hdf(self, hdf=None, group_name=None):
        super(ExternalHamiltonian, self).to_hdf(hdf=hdf, group_name=group_name)
        hdf[group_name]["fastlammpsmode"] = self._fast_lammps_mode

    def from_hdf(self, hdf=None, group_name=None):
        super(ExternalHamiltonian, self).from_hdf(hdf=hdf, group_name=group_name)
        self._fast_lammps_mode = hdf[group_name]["fastlammpsmode"]


class GradientDescent(PrimitiveVertex):
    """
    Simple gradient descent update for positions in `flex_output` and structure.

    Input attributes:
        gamma0 (float): Initial step size as a multiple of the force. (Default is 0.1.)
        fix_com (bool): Whether the center of mass motion should be subtracted off of the position update. (Default is
            True)
        use_adagrad (bool): Whether to have the step size decay according to adagrad. (Default is False)
        output_displacements (bool): Whether to return the per-atom displacement vector in the output dictionary.

    TODO:
        Fix adagrad bug when GradientDescent is passed as a Serial vertex
    """

    def __init__(self, name=None):
        super(GradientDescent, self).__init__(name=name)
        self.input.default.gamma0 = 0.1
        self.input.default.fix_com = True
        self.input.default.use_adagrad = False
        self._accumulated_force = 0

    def command(self, positions, forces, gamma0, use_adagrad, fix_com, mask=None, masses=None,
                output_displacements=True):
        positions = np.array(positions)
        forces = np.array(forces)

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

    def to_hdf(self, hdf=None, group_name=None):
        super(GradientDescent, self).to_hdf(hdf=hdf, group_name=group_name)
        hdf[group_name]["accumulatedforce"] = self._accumulated_force

    def from_hdf(self, hdf=None, group_name=None):
        super(GradientDescent, self).from_hdf(hdf=hdf, group_name=group_name)
        self._accumulated_force = hdf[group_name]["accumulatedforce"]


class HarmonicHamiltonian(PrimitiveVertex):
    """

    """

    def command(self, positions, home_positions, cell, pbc, spring_constant):
        dr = find_mic(positions - home_positions, cell, pbc)[0]
        force = -spring_constant * dr
        energy = 0.5 * np.sum(spring_constant * dr * dr)
        return {
            'forces': force,
            'energy_pot': energy
        }


class InterpolatePositions(PrimitiveVertex):
    """
    Creates interpolated positions between the positions of an initial and final structure.
    Returns only interpolated positions and not structures.

    Input attributes:
        structure_initial (Atoms): The initial structure
        structure_initial (Atoms): The final structure
        n_images (int): Number of structures to interpolate

    Output attributes:
        interpolated_positions (list/numpy.ndarray): A list of (n_images) positions interpolated between
            the positions of the initial and final structures.
    """

    def __init__(self, name=None):
        super(InterpolatePositions, self).__init__(name=name)

    def command(self, structure_initial, structure_final, n_images):
        pos_i = structure_initial.positions
        pos_f = structure_final.positions
        cell = structure_initial.cell
        pbc = structure_initial.pbc
        displacement = find_mic(pos_f - pos_i, cell, pbc)[0]

        interpolated_positions = []
        for n, mix in enumerate(np.linspace(0, 1, n_images)):
            interpolated_positions += [pos_i + (mix * displacement)]

        return {
            'interpolated_positions': interpolated_positions
        }


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
        id = self.input.default
        id.temperature = 0.
        id.damping_timescale = 100.
        id.time_step = 1.
        id.fix_com = True

    def command(self, velocities, masses,
                temperature, damping_timescale, time_step, fix_com):

        # Ensure that masses are a commensurate shape
        masses = np.array(masses)[:, np.newaxis]
        gamma = masses / damping_timescale

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
    Numpy's amax (except no `out` option).

    Docstrings directly copied from the `numpy docs`_. The `initial` field is renamed `initial_val` to not conflict with
    the input dictionary.

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

    Note: This misses the new argument ``initial`` in the latest version of Numpy.
    """
    def __init__(self, name=None):
        super(Max, self).__init__(name=name)
        self.input.default.axis = None
        self.input.default.keepdims = False
        self.input.default.initial_val = None

    def command(self, a, axis, keepdims, initial_val):
        return {
            'amax': np.amax(a, axis=axis, keepdims=keepdims)
        }


class NEBForces(PrimitiveVertex):
    """
    Given a list of positions, forces, and energies for each image along some transition, calculates the tangent
    direction along the transition path for each image and returns new forces which have their original value
    perpendicular to this tangent, and 'spring' forces parallel to the tangent. Endpoints forces are set to zero, so
    start with relaxed endpoint structures.

    Note:
        All images must use the same cell and contain the same number of atoms.

    Input attributes:
        positions_list (list/numpy.ndarray): The positions of the images. Each element should have the same shape, and
            the order of the images is relevant!
        energies (list/numpy.ndarray): The potential energies associated with each set of positions. (Not always
            needed.)
        force_list (list/numpy.ndarray): The forces from the underlying energy landscape on which the NEB is being run.
            Must have the same shape as `positions_list`. (Not always needed.)
        cell (numpy.ndarray): The cell matrix the positions live in. All positions must share the same cell.
        pbc (list/numpy.ndarray): Three bools declaring the presence of periodic boundary conditions along the three
            vectors of the cell.
        spring_constant (float): Spring force between atoms in adjacent images. (Default is 1.0 eV/angstrom^2.)
        tangent_style ('plain'/'improved'/'upwinding'): How to calculate the image tangent in 3N-dimensional space.
            (Default is 'upwinding', which requires energies to be set.)
        use_climbing_image (bool): Whether to replace the force with one that climbs along the tangent direction for
            the job with the highest energy. (Default is True, which requires energies and a list of forces to be set.)
        smoothing (float): Strength of the smoothing spring when consecutive images form an angle. (Default is None, do
            not apply such a force.)

    Output attributes:
        forces_list (list): The forces after applying nudged elastic band method.

    TODO:
        Add references to papers (Jonsson and others)
        Implement Sheppard's equations for variable cell shape
    """

    def __init__(self, name=None):
        super(NEBForces, self).__init__(name=name)
        self.input.default.spring_constant = 1.
        self.input.default.tangent_style = 'upwinding'
        self.input.default.use_climbing_image = True
        self.input.default.smoothing = None

    def command(self, positions_list, energies, forces_list, cell, pbc,
                spring_constant, tangent_style, smoothing, use_climbing_image):
        if use_climbing_image:
            climbing_image_index = np.argmax(energies)
        else:
            climbing_image_index = None

        n_images = len(positions_list)
        neb_forces = []
        for i, pos in enumerate(positions_list):
            if i == 0 or i == n_images - 1:
                neb_forces.append(np.zeros(pos.shape))
                continue

            # Otherwise calculate the spring forces
            pos_left = positions_list[i - 1]
            pos_right = positions_list[i + 1]

            # Get displacement to adjacent images
            dr_left = find_mic(pos - pos_left, cell, pbc)[0]
            dr_right = find_mic(pos_right - pos, cell, pbc)[0]

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
            input_force = forces_list[i]
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
            'forces_list': neb_forces
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
    Numpy's linalg norm.

    Docstrings directly copied from the `numpy docs`_.

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
        self.input.default.ord = 2
        self.input.default.axis = None
        self.input.default.keepdims = False

    def command(self, x, ord, axis, keepdims):
        return {
            'n': np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        }


class Overwrite(PrimitiveVertex):
    """
    Overwrite particular entries of an array with new values
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
    """

    def __init__(self, name=None):
        super(RandomVelocity, self).__init__(name=name)
        self.input.default.overheat_fraction = 2.

    def command(self, temperature, masses, overheat_fraction):

        masses = np.array(masses)[:, np.newaxis]
        vel_scale = np.sqrt(EV_TO_U_ANGSQ_PER_FSSQ * KB * temperature / masses) * np.sqrt(overheat_fraction)
        vel_dir = np.random.randn(len(masses), 3)
        vel = vel_scale * vel_dir
        vel -= np.mean(vel, axis=0)
        energy_kin = 0.5 * np.sum(masses * vel * vel) * U_ANGSQ_PER_FSSQ_TO_EV
        return {
            'velocities': vel,
            'energy_kin': energy_kin
        }


class SphereReflection(PrimitiveVertex):
    """
    Checks whether each atom in a structure is within a cutoff radius of its reference position; if not, reverts the
    positions and velocities of *all atoms* to an earlier state and reverses those earlier velocities. Forces from the
    current and previous time step can be provided optionally.

    Input attributes:
        reference_positions (numpy.ndarray): The reference positions to check the distances from.
        cutoff (float): The cutoff distance from the reference position to trigger reflection.
        positions (numpy.ndarray): The positions to check.
        velocities (numpy.ndarray): The velocities corresponding to the positions at this time.
        previous_positions (numpy.ndarray): The previous positions to revert to.
        previous_velocities (numpy.ndarray): The previous velocities to revert to and reverse.
        pbc (numpy.ndarray/list): Three booleans declaring which dimensions have periodic boundary conditions for
            finding the minimum distance convention.
        cell (numpy.ndarray): The 3x3 cell vectors for pbcs.
        forces (numpy.ndarray): The forces corresponding to the positions at this time. (Default is None.)
        previous_forces (numpy.ndarray): The forces at the `previous_positions` to revert to. (Default is None.)

    Output attributes:
        positions (numpy.ndarray): The (possibly reverted) positions.
        velocities (numpy.ndarray): The (possibly reverted and reversed) velocities.
        forces (numpy.ndarray/None): The (possibly reverted) forces, if force input was provided, else None.
        reflected_mask (bool): Whether or not reflection occurred.
    """

    def __init__(self, name=None):
        super(SphereReflection, self).__init__(name=name)
        self.input.default.forces = None
        self.input.default.previous_forces = None
        # self.input.default.previous_velocities = Pointer(self.input.velocities)

    def command(self, reference_positions, cutoff_distance, positions, velocities, previous_positions,
                previous_velocities, pbc, cell, forces, previous_forces):
        distance = np.linalg.norm(find_mic(reference_positions - positions, cell=cell, pbc=pbc)[0], axis=-1)
        is_at_home = (distance < cutoff_distance)[:, np.newaxis]

        if np.all(is_at_home):
            return {
                'positions': positions,
                'velocities': velocities,
                'forces': forces,
                'reflected': False
            }
        else:
            return {
                'positions': previous_positions,
                'velocities': -previous_velocities,
                'forces': previous_forces,
                'reflected': True
            }


class SphereReflectionPeratom(PrimitiveVertex):
    """
    Checks whether each atom in a structure is within a cutoff radius of its reference position; if not, reverts the
    positions and velocities of *the violating atoms* to an earlier state and reverses those earlier velocities. The
    remaining atoms are unaffected.

    Input attributes:
        reference_positions (numpy.ndarray): The reference positions to check the distances from.
        cutoff (float): The cutoff distance from the reference position to trigger reflection.
        positions (numpy.ndarray): The positions to check.
        velocities (numpy.ndarray): The velocities corresponding to the positions at this time.
        previous_positions (numpy.ndarray): The previous positions to revert to.
        previous_velocities (numpy.ndarray): The previous velocities to revert to and reverse.
        pbc (numpy.ndarray/list): Three booleans declaring which dimensions have periodic boundary conditions for
            finding the minimum distance convention.
        cell (numpy.ndarray): The 3x3 cell vectors for pbcs.

    Output attributes:
        positions (numpy.ndarray): The (possibly reverted) positions.
        velocities (numpy.ndarray): The (possibly reverted and reversed) velocities.
        reflected_mask (numpy.ndarray): A boolean mask that is true for each atom who was reflected.
    """

    def __init__(self, name=None):
        super(SphereReflectionPeratom, self).__init__(name=name)
        # self.input.default.previous_velocities = Pointer(self.input.velocities)

    def command(self, reference_positions, cutoff_distance, positions, velocities, previous_positions,
                previous_velocities, pbc, cell):
        distance = np.linalg.norm(find_mic(reference_positions - positions, cell=cell, pbc=pbc)[0], axis=-1)
        is_at_home = (distance < cutoff_distance)[:, np.newaxis]
        is_away = 1 - is_at_home

        return {
            'positions': is_at_home * positions + is_away * previous_positions,
            'velocities': is_at_home * velocities + is_away * -previous_velocities,
            'reflected': is_away.astype(bool).flatten()
        }


class Slice(PrimitiveVertex):
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
    """

    def command(self, matrix):
        transpose = list(map(list, zip(*matrix)))
        return {
            'matrix_transpose': transpose
        }


class VerletParent(PrimitiveVertex, ABC):
    """
    A parent class for holding code which is shared between both position and velocity updates in two-step Velocity
    Verlet.

    TODO: VerletVelocityUpdate should *always* have its velocity input wired to the velocity outupt of
          VerletPositionUpdate. This implies to me that we need some structure *other* than two fully independent
          nodes. It would also be nice to syncronize, e.g. the thermostat and timestep input which is also the same for
          both. However, I haven't figured out how to do that in the confines of the current graph traversal and hdf5
          setup.
    """

    def __init__(self, name=None):
        super(VerletParent, self).__init__(name=name)
        id = self.input.default
        id.time_step = 1.
        id.temperature = None
        id.temperature_damping_timescale = None

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
        noise = np.sqrt(EV_TO_U_ANGSQ_PER_FSSQ * KB * temperature * time_step / (masses * damping_timescale)) \
            * np.random.randn(*velocities.shape)
        noise -= np.mean(noise, axis=0)
        return drag + noise


class VerletPositionUpdate(VerletParent):
    """
    First half of Velocity Verlet, where positions are updated and velocities are set to their half-step value.

    Input dictionary:
        positions (numpy.ndarray): The per-atom positions in angstroms.
        velocities (numpy.ndarray): The per-atom velocities in angstroms/fs.
        forces (numpy.ndarray): The per-atom forces in eV/angstroms.
        masses (numpy.ndarray): The per-atom masses in atomic mass units.
        time_step (float): MD time step in fs. (Default is 1 fs.)
        temperature (float): The target temperature. (Default is None, no thermostat is used.)
        damping_timescale (float): Damping timescale in fs. (Default is None, no thermostat is used.)

    Output dictionary:
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
    Second half of Velocity Verlet, where velocities are updated. Forces should be updated between the position and
    velocity updates

    Input dictionary:
        velocities (numpy.ndarray): The per-atom velocities in angstroms/fs. These should be the half-step velocities
            output by `VerletPositionUpdate`.
        forces (numpy.ndarray): The per-atom forces in eV/angstroms. These should be updated since the last call of
            `VerletPositionUpdate`.
        masses (numpy.ndarray): The per-atom masses in atomic mass units.
        time_step (float): MD time step in fs. (Default is 1 fs.)
        temperature (float): The target temperature. (Default is None, no thermostat is used.)
        damping_timescale (float): Damping timescale in fs. (Default is None, no thermostat is used.)

    Output dictionary:
        velocities (numpy.ndarray): The new velocities *half* a time step in the future.
        energy_kin (float): The total kinetic energy of the system in eV
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

        return {
            'velocities': vel_step,
            'energy_kin': kinetic_energy
        }


class VoronoiReflection(PrimitiveVertex):
    """
    Checks whether each atom in a structure is closest to its own reference site; if not, reverts the positions and
    velocities to an earlier state and reverses those earlier velocities.

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
    """

    def __init__(self, name=None):
        super(VoronoiReflection, self).__init__(name=name)
        # self.input.default.previous_velocities = Pointer(self.input.velocities)

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
    Given a list of vectors of with the same shape, calculates a weighted sum of the vectors. By default the weights are
    the inverse of the number of elements and the sum is just the mean.

    Input attributes:
        vectors (list/numpy.ndarray): The vectors to sum. (Masked) vectors must all be of the same length. If the
            the vectors are already in a numpy array, the 0th index should determine the vector.
        weights (list/numpy.ndarray): A 1D list of coefficients (floats) with the same length as ``vectors``. (Default
            is None, which gives the simple mean.)
        masks (list): If not None, a mask must be passed for each vector to extract a sub-vector. The resulting
            collection of vectors and sub-vectors must all have the same length. If the mask for a given vector is
            `None`, all elements are retained. Otherwise the mask must be integer-like or boolean. (Default is None,
            do not mask any of the vectors.)

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
    def __init__(self, name=None):
        super(WelfordOnline, self).__init__(name=name)
        self.input.default.mean = None
        self.input.default.std = None
        self.input.default.n_samples = None

        op = Pointer(self.output)
        self.input.mean = op.mean[-1]
        self.input.std = op.std[-1]
        self.input.n_samples = op.n_samples[-1]

    def command(self, sample, mean, std, n_samples):
        if mean is None:
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
