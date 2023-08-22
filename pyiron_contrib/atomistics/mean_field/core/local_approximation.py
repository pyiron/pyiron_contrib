# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit

from pyiron_contrib.atomistics.mean_field.core.bond_analysis import StaticBondAnalysis

class GenerateBondingPotential():
    """
    !!! This class is currently in an experimental state and undergoes frequent updates. 
        Please be cautious when using it. -rads !!!
    
    
    Generates the 'local anharmonic' (LA) potential and force functions for a NN shell.
    
    This is done by displacing an atom i along 1 longitudinal and 2 transversal vectors, 
    and collecting the forces on atom j projected along these vectors. The longitudinal and transversal vectors are 
    the basis vectors of the atom j. 
    
    Takes as inputs all necessary data for the shell (generated using StaticBondAnalysis) and returns cubic spline 
    fits for the LA potential and force along the basis vectors.
    
    Parameters:
    - ref_job (object): The reference job object.
    - b_0 (float): The bond length for the NN shell.
    - basis (list): The basis vectors of atom j.
    - rotations (list): The rotation matrices which transform atom j to the equivalent atoms in the same shell.
    - jth_atom_id (int): The atom id of atom j. The forces on this atom will be used to generate the LA potential/forces.
    - ith_atom_id (int): The atom id of atom i. This atom will be displaced.
    - long_disp_low (float): This value will be subtracted off of b_0 to set the lower limit of the longitudinal 
                             displacement. Default is -0.5.
    - long_disp_hi (float): This value will be added to b_0 to set the upper limit of the longitudinal displacement. Default is 1.5.
    - t_disp_low (float): This value will be subtracted off of 0 to set the lower limit of the transversal 
                          displacement. Default is 0.
    - t_disp_hi (float): This value will be added to 0 to set the upper limit of the transversal displacement. Default is 1.
    - n_long (int): The number of samples for longitudinal displacement. Default is 10 samples.
    - n_t (int): The number of samples for transversal displacement. Default is 10 samples.
    - energy_offset (float): The energy offset that is added to the LA bonding potential. Default is 0.
    - deg (int): If deg is not None, then perform a polynomial fit of degree 'deg' for the transveral components.
    """
    
    def __init__(self, ref_job, b_0, basis, rotations, jth_atom_id, ith_atom_id=0, long_disp_low=-0.5, long_disp_hi=1.5, 
                 t_disp_low=0., t_disp_hi=1., n_long=10, n_t=10, energy_offset=0., deg=None, eta=0.15, uneven=False):
        self.ref_job = ref_job
        self.structure = ref_job.structure.copy()
        self.b_0 = b_0
        self.basis = np.array(basis)
        self.rotations = np.array(rotations)
        self.jth_atom_id = jth_atom_id
        self.ith_atom_id = ith_atom_id
        self.long_disp_low = long_disp_low
        self.long_disp_hi = long_disp_hi
        self.t_disp_low = t_disp_low
        self.t_disp_hi = t_disp_hi
        self.n_long = n_long
        self.n_t = n_t
        self.energy_offset = energy_offset
        self.deg = deg
        self.uneven = uneven
        
        self._project = self.ref_job.project.create_group('base_' + str(np.round(self.b_0, decimals=5)))
        self._long_hat = basis[0]
        self._t1_hat = basis[1]
        self._t2_hat = basis[2]
        self._long_pos = None
        self._t1_pos = None
        self._t2_pos = None
        self._long_force_on_j = None
        self._t1_force_on_j = None
        self._t2_force_on_j = None
        self._data = None
        self.eta = eta
        
    @staticmethod
    def uneven_linspace(lb, ub, steps, spacing=1.1, endpoint=True):
        span = (ub-lb)
        dx = 1.0/(steps-1)
        if not endpoint:
            dx = 1.0/(steps)
        return np.array([lb+(i*dx)**spacing*span for i in range(steps)])
    
    def displace_atom(self, position_of_the_atom, direction, disp_low, disp_hi, n_samples, tag='long'):
        """
        Displaces an atom along a given direction (unit vector) to generate a set of atom positions.
        
        Parameters:
        - position_of_the_atom (array-like): The initial position of the ith atom.
        - direction (array-like): The displacement direction (unit vector).
        - disp_low (float): Magnitude of the displacement towards atom j, if direction is along the line joining atoms i and j.
        - disp_hi (float): Magnitude of the displacement away from atom j, if direction is along the line joining atoms i and j.
        - n_samples (int): The number of samples for the displacement.
        - tag (str): 'long', 't1', or 't2'.
        
        Returns:
        - array-like: An array of atom positions generated by displacing the atom along the given direction.
        """
        if disp_low < 0.:
            # if the disp_low is < 0 (which would always be the case for the longitudinal direction and not always for the transversal
            # direction, we make sure to include 'b_0' (for longitudinal and 0. for transversal) in the samples. In this case, the 
            # samples are not evenly spaced!
            half_samples = int(n_samples/2)
            if self.uneven:
                low_samples = self.uneven_linspace(disp_low, 0., half_samples, spacing=0.7, endpoint=False)
                hi_samples = self.uneven_linspace(0., disp_hi, n_samples-half_samples, spacing=1.6)
            else:
                low_samples = np.linspace(disp_low, 0., half_samples, endpoint=False)
                hi_samples = np.linspace(0., disp_hi, n_samples-half_samples)
            samples = np.concatenate((low_samples, hi_samples))
        else:
            if self.uneven:
                samples = self.uneven_linspace(disp_low, disp_hi, n_samples, spacing=1.6)
            else:
                samples = np.linspace(disp_low, disp_hi, n_samples)
        if tag == 'long':
            return np.array([position_of_the_atom-direction*s for s in samples])
        else:
            return np.array([position_of_the_atom-direction*s for s in samples])
    
    def generate_atom_positions(self):
        """
        Generates and stores the 0th atom positions along the longitudinal and 2 transversal directions.
        """
        self._long_pos = self.displace_atom(self.structure.positions[self.ith_atom_id], self._long_hat, disp_low=self.long_disp_low,
                                            disp_hi=self.long_disp_hi, n_samples=self.n_long, tag='long')
        self._t1_pos = self.displace_atom(self.structure.positions[self.ith_atom_id], self._t1_hat, disp_low=self.t_disp_low, 
                                          disp_hi=self.t_disp_hi, n_samples=self.n_t, tag='t1')
        self._t2_pos = self.displace_atom(self.structure.positions[self.ith_atom_id], self._t2_hat, disp_low=self.t_disp_low, 
                                          disp_hi=self.t_disp_hi, n_samples=self.n_t, tag='t2')
    
    def _run_jobs(self, positions, tag, delete_existing_jobs=False):
        """
        Runs a static calculation for a structure with the 0th atom position replaced by each entry in positions, for the 'tag' direction.
        
        Parameters:
        - positions (array-like): The positions of the 0th atom.
        - tag (str): 'long', 't1' or 't2'.
        """
        def run_job(position):
            job = self.ref_job.copy_template(project=pr_tag, new_job_name=job_name)
            job.structure.positions[self.ith_atom_id] = position
            job.calc_static()
            job.run()
            
        pr_tag = self._project.create_group(tag)
        job_list = pr_tag.job_table().job.to_list()
        job_status = pr_tag.job_table().status.to_list()
        for i, pos in enumerate(positions):
            job_name = tag + '_' + str(i)
            if job_name not in job_list:
                run_job(position=pos)
            elif job_status[i] != 'finished' or delete_existing_jobs:
                pr_tag.remove_job(job_name)
                run_job(position=pos) 

    def run_jobs(self, delete_existing_jobs=False):
        """
        Runs static calculation jobs for the longitudinal and 2 transversal directions.
        """
        self.generate_atom_positions()
        self._run_jobs(positions=self._long_pos, tag='long', delete_existing_jobs=delete_existing_jobs)
        self._run_jobs(positions=self._t1_pos, tag='t1', delete_existing_jobs=delete_existing_jobs)
        self._run_jobs(positions=self._t2_pos, tag='t2', delete_existing_jobs=delete_existing_jobs)
        
    def _load_jobs(self, tag, n_samples):
        """
        Loads the static calculation jobs for the 'tag' direction.
        
        Parameters:
        - tag (str): 'long', 't1' or 't2'.
        - n_samples (int): The number of samples for the jobs.
        
        Returns:
        - list: A list of loaded jobs.
        """
        pr_tag = self._project.create_group(tag)
        jobs = []
        for i in range(n_samples):
            jobs.append(pr_tag.inspect(tag + '_' + str(i)))
        return jobs
    
    def load_jobs(self):
        """
        Loads static calculation jobs for the longitudinal and 2 transversal directions.
        
        Returns:
        - tuple: A tuple containing the lists of loaded jobs for longitudinal, t1, and t2 directions.
        """
        self.generate_atom_positions()
        long_jobs = self._load_jobs(tag='long', n_samples=self.n_long)
        t1_jobs = self._load_jobs(tag='t1', n_samples=self.n_t)
        t2_jobs = self._load_jobs(tag='t2', n_samples=self.n_t)
        return long_jobs, t1_jobs, t2_jobs
    
    def _force_on_j(self, jobs):
        """
        Returns the forces on the jth atom.

        Parameters:
        - jobs (list): A list of job objects.

        Returns:
        - array-like: An array of forces on the jth atom.
        """
        return np.array([job['output/generic/forces'][-1][self.jth_atom_id] for job in jobs])
    
    def get_force_on_j(self):
        """
        Retrieves and stores the forces on the jth atom along longitudinal and 2 transversal directions.
        """
        long_jobs, t1_jobs, t2_jobs = self.load_jobs()
        self.long_force_on_j = self._force_on_j(long_jobs)
        self.t1_force_on_j = self._force_on_j(t1_jobs)
        self.t2_force_on_j = self._force_on_j(t2_jobs)
    
    def find_mic(self, vectors):
        """
        Applies minimum image convention (MIC) to a set of vectors to account for periodic boundary conditions (PBC).

        Parameters:
        - vectors (array-like): An array of vectors.

        Returns:
        - array-like: An array of vectors with MIC applied.
        """
        cell=self.structure.cell
        pbc=self.structure.pbc
        vecs = np.asarray(vectors).reshape(-1, 3)
        if any(pbc):
            vecs = np.einsum('ji,nj->ni', np.linalg.inv(cell), vecs)
            vecs[:, pbc] -= np.rint(vecs)[:, pbc]
            vecs = np.einsum('ji,nj->ni', cell, vecs)
        return vecs.reshape(np.asarray(vectors).shape)
    
    @staticmethod
    def orthogonalize(matrix):
        """
        Orthogonalizes a matrix using the Gram-Schmidt process.

        Parameters:
        - matrix (array-like): The matrix to be orthogonalized.

        Returns:
        - array-like: The orthogonalized matrix.
        """
        orth_matrix = matrix.copy()
        orth_matrix[1] = matrix[1] - (orth_matrix[0]@matrix[1])/(orth_matrix[0]@orth_matrix[0])*orth_matrix[0]
        orth_matrix[2] = matrix[2] - (orth_matrix[0]@matrix[2])/(orth_matrix[0]@orth_matrix[0])*orth_matrix[0] \
                                   - (orth_matrix[1]@matrix[2])/(orth_matrix[1]@orth_matrix[1])*orth_matrix[1]
        return orth_matrix/np.linalg.norm(orth_matrix, axis=-1)[:, np.newaxis]
    
    def get_ij_bond_force(self, i_atom_positions, j_atom_forces):
        """
        Computes the bond forces along the line connecting atoms i and j (which is r). Corresponds to the 'central' forces. 

        Parameters:
        - i_atom_positions (array-like): Positions of the ith atom.
        - j_atom_forces (array-like): Forces on the jth atom.

        Returns:
        - tuple: A tuple containing the magnitudes of displacements, displacement directions, and bond forces.
        """
        j_position = self.structure.positions[self.jth_atom_id]
        ij = self.find_mic(j_position-i_atom_positions)
        r = np.linalg.norm(ij, axis=-1)
        ij_direcs = ij/r[:, np.newaxis]
        F_ij = (j_atom_forces*ij_direcs).sum(axis=-1)
        return r, ij_direcs, F_ij
    
    def get_t_bond_force(self, tag='t2'):
        """
        Computes the bond forces along a transversal direction.
        
        For a displacement 'u' along a transversal direction, one has to remember that there is also a displacement r = sqrt(u**2+b_0**2)
        along the ij or 'r' or 'central' direction that needs to accounted for. The force on atom j for a displacement u along a 
        transversal direction is then F_j = F_t(u) + F_ij(r), where we take ij to be the longitudinal direction l with 0 transversal
        displacement. The true transversal force F_t(u) is then F_j - F_ij(r).
        
        Further, for each transversal displacment u != 0, ij is not perpendicular to the original transversal direction anymore. By using 
        the Gram-Schmit method, we can orthogonalize the transversal direction to ij and parameterize the forces along this new 
        transversal direction 't_prime' as 'F_t_prime'. While the effect of this orthogonalization is minimial, we by default will still 
        consider this for the parameterization.

        Parameters:
        - tag (str): 't1' or 't2'.

        Returns:
        - tuple: A tuple containing the displacements u and bond forces F_t_prime along the 'tag' direction.
        """
        if tag=='t1':
            positions = self._t1_pos
            force_on_j = self.t1_force_on_j
            v_id = 1
        elif tag=='t2':
            positions = self._t2_pos
            force_on_j = self.t2_force_on_j
            v_id = 2
        else:
            raise ValueError
            
        r, ij_direcs, F_ij_mags = self.get_ij_bond_force(i_atom_positions=positions, j_atom_forces=force_on_j)
        u = ij_direcs*r[:, np.newaxis]@self.basis[v_id]
        F_t = force_on_j - ij_direcs*F_ij_mags[:, np.newaxis]  
        
        F_t_prime = []
        for ij_dir, f_t in zip(ij_direcs, F_t):
            basis_copy = self.basis.copy()
            basis_copy[0] = ij_dir
            # t_prime = self.orthogonalize(basis_copy)[v_id]
            t_prime = basis_copy[v_id]
            F_t_prime.append(f_t@t_prime)
        F_t_prime = np.array(F_t_prime)
        
        # take F_t_prime == F_t 
        # F_t_prime = np.linalg.norm(F_t, axis=-1)

        if self.t_disp_low == 0.:
            u = np.concatenate((-np.flip(u)[:-1], u))
            F_t_prime = np.concatenate((-np.flip(F_t_prime)[:-1], F_t_prime))
            
        return u, F_t_prime
    
    def _get_potential(self, bonds, force, tag='long'):
        """
        Computes the potential energy from the bond length and bond force. Also subtracts the energy offset, if any.

        Parameters:
        - bonds (array-like): The bond lengths.
        - force (array-like): The bond forces.
        - tag (str): 'long', 't1', or 't2'.

        Returns:
        - array-like: The potential energy.
        """
        if tag == 'long':
            arg_b_0 = np.argmin(abs(bonds-self.b_0))
        elif tag in ['t1', 't2']:
            arg_b_0 = np.argmin(abs(bonds))
        else:
            raise ValueError
        up = cumtrapz(y=-force[arg_b_0:], x=bonds[arg_b_0:], initial=0.)
        down = np.flip(cumtrapz(y=-np.flip(force[:arg_b_0+1]), x=np.flip(bonds[:arg_b_0+1])))
        potential = np.concatenate((down, up))
        if tag == 'long':
            return potential
        return potential
    
    def get_all_bond_force_pot(self):
        """
        Computes bond forces and potentials for the longitudinal and 2 transversal directions.

        Returns:
        - list: A list of lists containing the bond lengths, bond forces, and potentials for each direction.
        """
        # long
        long_bonds, _, long_force = self.get_ij_bond_force(i_atom_positions=self._long_pos, j_atom_forces=self.long_force_on_j)
        long_force -= long_force[np.argmin(abs(long_bonds-self.b_0))]
        long_pot = self._get_potential(long_bonds, long_force, tag='long')
        # t1
        t1_bonds, t1_force = self.get_t_bond_force(tag='t1')
        t1_force -= t1_force[np.argmin(abs(t1_bonds))]
        t1_pot = self._get_potential(t1_bonds, t1_force, tag='t1')
        # t2
        t2_bonds, t2_force = self.get_t_bond_force(tag='t2')
        t2_force -= t2_force[np.argmin(abs(t2_bonds))]
        t2_pot = self._get_potential(t2_bonds, t2_force, tag='t2')
        self._data = [[long_bonds, -long_force, long_pot],
                      [t1_bonds, -t1_force, t1_pot],
                      [t2_bonds, -t2_force, t2_pot]]
    
    def get_spline_fit(self, data):
        """
        Performs a cubic spline fit to the given data.

        Parameters:
        - data (list): A list of bond lengths, bond forces, and potentials.

        Returns:
        - tuple: A tuple containing the cubic spline functions for bond force and potential.
        """
        force = CubicSpline(data[0], data[1])
        potential = CubicSpline(data[0], data[2]+self.energy_offset)
        return force, potential
    
    def get_poly_fit(self, data, deg=4):
        """
        Performs a polynomial fit to the given data.

        Parameters:
        - data (list): A list of bond lengths, bond forces, and potentials.
        - deg (int): The degree of the polynomial fit (default is 4).

        Returns:
        - tuple: A tuple containing the polynomial functions for bond force and potential.
        """
        force = np.poly1d(np.polyfit(data[0], data[1], deg=deg))
        potential = np.poly1d(np.polyfit(data[0], data[2]+self.energy_offset, deg=deg))
        return force, potential
    
    def morse(self, r, D, alpha):
        """
        Computes the Morse potential energy as a function of the distance.

        Parameters:
        - r (float): Distance.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Morse potential energy.
        """
        return D*(1.0+np.exp(-2.0*alpha*(r-self.b_0))-2.0*np.exp(-alpha*(r-self.b_0)))+self.energy_offset

    def dmorse(self, r, D, alpha):
        """
        Computes the derivative of the Morse potential energy with respect to the distance.

        Parameters:
        - r (float): Distance.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Derivative of the Morse potential energy with respect to the distance.
        """
        return -2.0*alpha*D*(np.exp(-2.0*alpha*(r-self.b_0))-np.exp(-alpha*(r-self.b_0)))

    @staticmethod
    def harm(r, kappa, D, alpha):
        """
        Computes the harmonic potential energy as a function of the distance.

        Parameters:
        - r (float): Distance.
        - kappa (float): Harmonic potential parameter.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Harmonic potential energy.
        """
        return kappa*D*alpha**2*r**2

    @staticmethod
    def dharm(r, kappa, D, alpha):
        """
        Computes the derivative of the harmonic potential energy with respect to the distance.

        Parameters:
        - r (float): Distance.
        - kappa (float): Harmonic potential parameter.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Derivative of the harmonic potential energy with respect to the distance.
        """
        return 2.0*kappa*D*alpha**2*r

    def apply_lindemann_criteria(self, r, force, tag='long'):
        """
        Filters the distance and force arrays based on the Lindemann criteria.

        Parameters:
        - r (numpy.ndarray): Array of distances.
        - force (numpy.ndarray): Array of forces.
        - tag (str, optional): Tag to specify the criteria ('long' for longitudinal, 't1' or 't2' for transversal).

        Returns:
        - tuple: A tuple containing the filtered distance and force arrays.
        """
        if tag == 'long':
            mask = (self.b_0*(1-self.eta) <= r) & (r < self.b_0*(1+self.eta))
        else:
            mask = (-self.b_0*self.eta <= r) & (r < self.b_0*self.eta)
        return r[mask], force[mask]

    def get_potential_parameters(self):
        """
        Estimates the potential Morse and harmonic parameters based on the force and distance data.

        Parameters:
        - data_array (numpy.ndarray): An array containing distance and force data for longitudinal, t1, and t2 directions.

        Returns:
        - numpy.ndarray: An array of estimated potential parameters.
        """
        # parameterize longitudinal to morse
        long, long_force = self.apply_lindemann_criteria(self._data[0][0], self._data[0][1], tag='long')
        popt_long, _ = curve_fit(f=self.dmorse, xdata=long, ydata=long_force, p0=(0.1, 1.5))
        # parameterize transversal to harmonic
        # t1
        def t_eqn(r, kappa):
            return self.dharm(r, kappa, D=popt_long[0], alpha=popt_long[1])
        t1, t1_force = self.apply_lindemann_criteria(self._data[1][0], self._data[1][1], tag='t1')
        popt_t1, pcov_t1 = curve_fit(f=t_eqn, xdata=t1, ydata=t1_force, p0=0.1)
        # t2
        t2, t2_force = self.apply_lindemann_criteria(self._data[2][0], self._data[2][1], tag='t2')
        popt_t2, pcov_t2 = curve_fit(f=t_eqn, xdata=t2, ydata=t2_force, p0=0.1)
        return np.array([popt_long[0], popt_long[1], popt_t1[0], popt_t2[0]])
    
    def get_parameterized_functions(self, D, alpha, kappa_1, kappa_2):
        """
        Computes the parameterized force and potential energy functions for the longitudinal and two transverse directions.

        Parameters:
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.
        - kappa_1 (float): Hamronic potential parameter for the t1 direction.
        - kappa_2 (float): Hamronic potential parameter for the t2 direction.

        Returns:
        - tuple: A tuple containing the parameterized potential energy functions for the longitudinal, t1, and t2 directions,
                 and the force functions for each direction.
        """
        def long_force(r):
            return self.dmorse(r, D=D, alpha=alpha)
        def long_potential(r):
            return self.morse(r, D=D, alpha=alpha)
        def t1_force(r):
            return self.dharm(r, D=D, alpha=alpha, kappa=kappa_1)
        def t1_potential(r):
            return self.harm(r, D=D, alpha=alpha, kappa=kappa_1)
        def t2_force(r):
            return self.dharm(r, D=D, alpha=alpha, kappa=kappa_2)
        def t2_potential(r):
            return self.harm(r, D=D, alpha=alpha, kappa=kappa_2)
        return long_potential, t1_potential, t2_potential, long_force, t1_force, t2_force

    def get_bonding_forces_potential(self, parameterized=False, deg=None):
        """
        Computes the bonding forces and potentials for the longitudinal and two transverse directions.

        Parameters:
        - parameterized (bool, optional): Flag to indicate whether to use parameterized potentials.
        - deg (int, optional): Degree of the polynomial fit (used if parameterized is False).

        Returns:
        - tuple: A tuple containing the longitudinal, t1, and t2 bonding forces and potential functions.
        """
        self.get_force_on_j()
        data = self.get_all_bond_force_pot()
        if parameterized:
            parameters = self.get_potential_parameters()
            long_potential, t1_potential, t2_potential, long_force, t1_force, t2_force = self.get_parameterized_functions(*parameters)
        else:
            long_force, long_potential = self.get_spline_fit(data=self._data[0])
            if self.deg is not None:
                t1_force, t1_potential = self.get_poly_fit(data=self._data[1], deg=self.deg)
                t2_force, t2_potential = self.get_poly_fit(data=self._data[2], deg=self.deg)
            else:
                t1_force, t1_potential = self.get_spline_fit(data=self._data[1])
                t2_force, t2_potential = self.get_spline_fit(data=self._data[2])
        return long_potential, t1_potential, t2_potential, long_force, t1_force, t2_force


class GenerateLAPotentialv2():
    """
    Generates the 'local anharmonic' (LA) potential and force functions for the 1st NN shell.
    
    This is done by displacing an atom i along the line that connects it to NN atom j (longitudinal direction between the 2), 
    and collecting the forces on atom j. From these forces, the central/radial 'bonding potential' between the 2 atoms can be 
    parameterized. Because of symmetry in the FCC crystal, forces on the NN atoms along the 2 transversal directions (perpendicular 
    to the longitudinal) can be also collected to parameterize the 2 transversal bonding potentials from the same longitudinal
    displacements.
    
    Parameters:
    - project_name (str): Name of the pyiron project for storing the jobs.
    - ref_job (pyiron Job): Reference job containing an appropriate FCC structure and an interatomic potential.
    - ith_atom_id (int, optional): Index of the atom that will be displaced (default is 0).
    - disp (float, optional): Maximum displacement for atom perturbations in lattice units (default is 1.0).
    - n_disps (int, optional): Number of displacement steps (default is 5).
    - uneven (bool, optional): Flag to use uneven spacing for displacement samples (default is False).
    - delete_existing_jobs (bool, optional): Flag to delete all existing jobs within the project. Cannot be used for individual jobs!
      (default is False)
    """
    
    def __init__(self, project_name, ref_job, ith_atom_id=0, disp=1., n_disps=5, uneven=False, delete_existing_jobs=False):
        
        self.project_name = project_name
        self.ref_job = ref_job
        self.structure = ref_job.structure.copy()
        self.ith_atom_id = ith_atom_id
        self.disp = disp
        self.n_disps = n_disps
        self.uneven = uneven
        self.delete_existing_jobs = delete_existing_jobs
        
        self._project = self.ref_job.project.create_group(self.project_name)
        self._stat_ba = None
        self._nn_bond_vecs = None
        self._nn_atom_ids = None
        self._disp_pos = None
        self._disp_jobs = None
        self._plane_nn = None
        self._force_on_j_list = None
        self._data = None
        
    def run_static_analysis(self):
        """
        Run static bond analysis on the reference job.
        """
        self._stat_ba = self._project.create_job(StaticBondAnalysis, 'stat_ba', 
                                                 delete_existing_job=self.delete_existing_jobs)
        self._stat_ba.input.structure = self.structure.copy()
        self._stat_ba.input.n_shells = 2
        self._stat_ba.run()

        self._b_0 = np.linalg.norm(self._stat_ba.output.per_shell_irreducible_bond_vectors, axis=-1)[0]
        self._long_hat, self._t1_hat, self._t2_hat = self._stat_ba.output.per_shell_transformation_matrices[0][0] 
        
    def get_nn_bond_info(self, output=False):
        """
        Retrieve neighbor bond information.
        """
        rotations = self._stat_ba.output.per_shell_0K_rotations
        basis = [self._stat_ba.output.per_shell_transformation_matrices[i][0] for i in range(2)]
        self._nn_bond_vecs = np.concatenate([basis[i][0]@rotations[i] for i in range(2)])
        self._nn_atom_ids = np.concatenate([self._stat_ba.output.per_shell_bond_indexed_neighbor_list[i][self.ith_atom_id]
                                            for i in range(2)])
        if output:
            return self._nn_bond_vecs, self._nn_atom_ids
        
    @staticmethod
    def uneven_linspace(lb, ub, steps, spacing=1.1, endpoint=True):
        """
        Generate unevenly spaced samples using a power-law distribution with a specified spacing factor. 
        The power-law distribution allows for denser sampling near the lower bound if spacing > 1, and denser sampling towards 
        the upper bound if spacing < 1.

        Parameters:
        - lb (float): Lower bound of the range for generating samples.
        - ub (float): Upper bound of the range for generating samples.
        - steps (int): Number of samples to generate.
        - spacing (float, optional): Spacing factor controlling the distribution (default is 1.1).
        - endpoint (bool, optional): Whether to include the upper bound in the samples (default is True).

        Returns:
        - numpy.ndarray: An array of unevenly spaced samples.
        """
        span = (ub-lb)
        dx = 1.0/(steps-1)
        if not endpoint:
            dx = 1.0/(steps)
        return np.array([lb+(i*dx)**spacing*span for i in range(steps)])
    
    def generate_atom_positions(self, output=False):
        """
        Generate atom positions corresponding to the displacements from equilibrium.
        """
        if self.uneven:
            samples = self.uneven_linspace(0., self.disp, self.n_disps, spacing=self.uneven)
        else:
            samples = np.linspace(0., self.disp, self.n_disps)
        
        ith_atom_pos = self.structure.positions[self.ith_atom_id]
        self._disp_pos = np.array([ith_atom_pos-self._long_hat*s for s in samples])
        if output:
            return self._disp_pos
    
    def run_disp_jobs(self):
        """
        Run displacement jobs for the perturbed atom positions.
        """
        if self._stat_ba is None:
            self.run_static_analysis()
            self.get_nn_bond_info()
        if self._disp_pos is None:
            self.generate_atom_positions()
        
        def _run_job(position):
            job = self.ref_job.copy_template(project=self._project, new_job_name=job_name)
            job.structure.positions[self.ith_atom_id] = position
            job.calc_static()
            job.run()
            
        job_list = self._project.job_table().job.to_list()
        job_status = self._project.job_table().status.to_list()
        for i, pos in enumerate(self._disp_pos):
            job_name = 'disp_' + str(i)
            if job_name not in job_list:
                _run_job(position=pos)
            elif job_status[i] != 'finished' or self.delete_existing_jobs:
                self._project.remove_job(job_name)
                _run_job(position=pos) 
        
    def load_disp_jobs(self):
        """
        Load displacement jobs from the project.
        """
        if self._stat_ba is None:
            self.run_static_analysis()
            self.get_nn_bond_info()
        if self._disp_pos is None:
            self.generate_atom_positions()
        self._disp_jobs = [self._project.inspect('disp_' + str(i)) for i in range(self.n_disps)]
    
    def get_plane_neighbors(self, output=False):
        """
        Retrieve jth atom indices along the longitudinal and 2 transversal directions.
        """
        nn_bond_vecs = self._nn_bond_vecs
        anti_long_id = np.argwhere(np.isclose(a=nn_bond_vecs@self._long_hat, b=-1., atol=1e-10))[-1][-1]
        t1_id = np.argwhere(np.all(np.isclose(a=nn_bond_vecs, b=self._t1_hat, atol=1e-10), axis=-1))[-1][-1]
        anti_t1_id = np.argwhere(np.isclose(a=nn_bond_vecs@self._t1_hat, b=-1., atol=1e-10))[-1][-1]
        t2_id = np.argwhere(np.all(np.isclose(a=nn_bond_vecs, b=self._t2_hat, atol=1e-10), axis=-1))[-1][-1]
        anti_t2_id = np.argwhere(np.isclose(a=nn_bond_vecs@self._t2_hat, b=-1., atol=1e-10))[-1][-1]
        self._plane_nn = np.array([self._nn_atom_ids[i] for i in [self.ith_atom_id, anti_long_id,  
                                                                  t1_id, anti_t1_id, t2_id, anti_t2_id]])
        if output:
            return self._plane_nn
    
    def _force_on_j(self, atom_id):
        return np.array([job['output/generic/forces'][-1][atom_id] for job in self._disp_jobs])
    
    def get_force_on_j(self, output=False):
        """
        Retrieve forces on the jth atoms at each dislacement.
        """
        if self._force_on_j_list is None:
            self.load_disp_jobs()
        if self._plane_nn is None:
            self.get_plane_neighbors()  
        self._force_on_j_list = np.array([self._force_on_j(i) for i in self._plane_nn])
        if output:
            return self._force_on_j_list
    
    @staticmethod
    def find_mic(structure, vectors):
        """
        Apply minimum image convention (MIC) to a set of vectors to account for periodic boundary conditions (PBC).

        Parameters:
        - vectors (array-like): An array of vectors.

        Returns:
        - array-like: An array of vectors with MIC applied.
        """
        cell = structure.cell
        pbc = structure.pbc
        vecs = np.asarray(vectors).reshape(-1, 3)
        if any(pbc):
            vecs = np.einsum('ji,nj->ni', np.linalg.inv(cell), vecs)
            vecs[:, pbc] -= np.rint(vecs)[:, pbc]
            vecs = np.einsum('ji,nj->ni', cell, vecs)
        return vecs.reshape(np.asarray(vectors).shape)
    
    @staticmethod
    def orthogonalize(matrix):
        """
        Orthogonalize a matrix using the Gram-Schmidt process.

        Parameters:
        - matrix (array-like): The matrix to be orthogonalized.

        Returns:
        - array-like: The orthogonalized matrix.
        """
        orth_matrix = matrix.copy()
        orth_matrix[1] = matrix[1]-(orth_matrix[0]@matrix[1])/(orth_matrix[0]@orth_matrix[0])*orth_matrix[0]
        orth_matrix[2] = matrix[2]-(orth_matrix[0]@matrix[2])/(orth_matrix[0]@orth_matrix[0])*orth_matrix[0] \
                                  -(orth_matrix[1]@matrix[2])/(orth_matrix[1]@orth_matrix[1])*orth_matrix[1]
        return orth_matrix/np.linalg.norm(orth_matrix, axis=-1)[:, np.newaxis]
    
    def get_ij_bond_force(self, jth_atom_id, jth_atom_forces):
        """
        Compute the bonding forces along the line connecting atoms i and j (which is r). Corresponds to the 'central' force. 

        Parameters:
        - jth_atom_id (array-like): id of the jth atom.
        - j_atom_forces (array-like): Forces on the jth atom.

        Returns:
        - tuple: A tuple containing the magnitudes of displacements, bond forces, and displacement directions.
        """
        j_position = self.structure.positions[jth_atom_id]
        ij = self.find_mic(self.structure, j_position-self._disp_pos)
        r = np.linalg.norm(ij, axis=-1)
        ij_direcs = ij/r[:, np.newaxis]
        F_ij = (jth_atom_forces*ij_direcs).sum(axis=-1)
        return r, F_ij, ij_direcs
    
    def get_t_bond_force(self, jth_atom_id, jth_atom_forces, tag='t1'):
        """
        Compute the bonding forces along a transversal direction.
        
        For a displacement 'u' along a transversal direction, there is also a displacement r = sqrt(u**2+b_0**2) along ij or 'r' 
        or the 'central' direction. The force on atom j for a displacement u along a transversal direction is then F_j = F_t(u) +
        F_ij(r), where F_t(u) is the force along the transversal direction and F_ij(r) is the force along the ij.
        
        Further, for each transversal displacment u != 0, ij is not perpendicular to the original transversal direction. By using 
        the Gram-Schmit process, we can orthogonalize the transversal direction to ij and parameterize the forces along the new 
        transversal direction 't_prime' as 'F_t_prime'. While the effect of this orthogonalization is minimial, we by default will 
        consider this for the parameterization.

        Parameters:
        - jth_atom_id (array-like): id of the jth atom.
        - j_atom_forces (array-like): Forces on the jth atom.
        - tag (str): 't1' or 't2'.

        Returns:
        - tuple: A tuple containing the displacements u and bond forces F_t_prime along the 'tag' direction.
        """
        r, F_ij, ij_direcs = self.get_ij_bond_force(jth_atom_id=jth_atom_id, jth_atom_forces=jth_atom_forces)
        u = ij_direcs*r[:, np.newaxis]@self._long_hat
        F_t = jth_atom_forces-ij_direcs*F_ij[:, np.newaxis]
        if tag == 't1':
            basis = np.array([self._t1_hat, self._long_hat, self._t2_hat])
        elif tag == 't2':
            basis = np.array([self._t2_hat, self._long_hat, self._t1_hat])
        F_t_prime = []
        for ij_dir, f_t in zip(ij_direcs, F_t):
            basis[0] = ij_dir
            t_prime = self.orthogonalize(basis.copy())[1]
            F_t_prime.append(f_t@t_prime)
        F_t_prime = np.array(F_t_prime)
        return u, F_t_prime
    
    def _bond_force(self, tag='long'):
        if tag == 'long':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id) 
                   for forces, atom_id in zip(self._force_on_j_list[:2], self._plane_nn[:2])]
            bonds = np.concatenate((np.flip(out[1][0]), out[0][0][1:]))
            force = np.concatenate((np.flip(out[1][1]), out[0][1][1:]))
        elif tag == 't1':
            out = [self.get_t_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, tag=tag) 
                   for forces, atom_id in zip(self._force_on_j_list[2:4], self._plane_nn[2:4])]
            bonds = np.concatenate((-np.flip(out[1][0]), out[0][0][1:]))
            force = np.concatenate((-np.flip(out[1][1]), out[0][1][1:]))
        elif tag == 't2':
            out = [self.get_t_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, tag=tag) 
                   for forces, atom_id in zip(self._force_on_j_list[4:], self._plane_nn[4:])]
            bonds = np.concatenate((-np.flip(out[1][0]), out[0][0][1:]))
            force = np.concatenate((-np.flip(out[1][1]), out[0][1][1:]))
        else:
            raise ValueError
        return bonds, force
    
    def _bond_force_pot(self, tag='long'):
        bonds, force = self._bond_force(tag=tag)
    
        if tag == 'long':
            arg_b_0 = np.argmin(abs(bonds-self._b_0))
        elif tag in ['t1', 't2']:
            arg_b_0 = np.argmin(abs(bonds))
        else:
            raise ValueError
            
        up = cumtrapz(y=-force[arg_b_0:], x=bonds[arg_b_0:], initial=0.)
        down = np.flip(cumtrapz(y=-np.flip(force[:arg_b_0+1]), x=np.flip(bonds[:arg_b_0+1])))
        potential = np.concatenate((down, up))
        return bonds, force, potential
    
    def get_all_bond_force_pot(self, output=False):
        """
        Compute bonding forces and potentials along the longitudinal and 2 transversal directions.

        Returns:
        - list: A list of lists containing the bond lengths, bond forces, and potentials for each direction.
        """
        long_bonds, long_forces, long_potential = self._bond_force_pot(tag='long')
        t1_bonds, t1_forces, t1_potential = self._bond_force_pot(tag='t1')
        t2_bonds, t2_forces, t2_potential = self._bond_force_pot(tag='t2')
        self._data = [[long_bonds, -long_forces, long_potential],
                      [t1_bonds, -t1_forces, t1_potential],
                      [t2_bonds, -t2_forces, t2_potential]]
        if output:
            return self._data
    
    def get_spline_fit(self, data):
        """
        Perform a cubic spline fit to the given data.

        Parameters:
        - data (list): A list of bond lengths, bond forces, and potentials.

        Returns:
        - tuple: A tuple containing the cubic spline functions for bond force and potential.
        """
        force = CubicSpline(data[0], data[1])
        potential = CubicSpline(data[0], data[2])
        return force, potential
    
    def get_poly_fit(self, data, deg=4):
        """
        Perform a polynomial fit to the given data.

        Parameters:
        - data (list): A list of bond lengths, bond forces, and potentials.
        - deg (int): The degree of the polynomial fit (default is 4).

        Returns:
        - tuple: A tuple containing the polynomial functions for bond force and potential.
        """
        force = np.poly1d(np.polyfit(data[0], data[1], deg=deg))
        potential = np.poly1d(np.polyfit(data[0], data[2], deg=deg))
        return force, potential
    
    def morse(self, r, D, alpha):
        """
        Compute the Morse potential energy as a function of radial distance.

        Parameters:
        - r (float): Radial distance.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Morse potential energy.
        """
        return D*(1.0+np.exp(-2.0*alpha*(r-self._b_0))-2.0*np.exp(-alpha*(r-self._b_0)))

    def dmorse(self, r, D, alpha):
        """
        Computes the derivative of the Morse potential energy with respect to the radial distance.

        Parameters:
        - r (float): Radial distance.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Derivative of the Morse potential energy with respect to the distance.
        """
        return -2.0*alpha*D*(np.exp(-2.0*alpha*(r-self._b_0))-np.exp(-alpha*(r-self._b_0)))

    @staticmethod
    def harm(r, kappa, D, alpha):
        """
        Compute the harmonic potential energy as a function of distance.

        Parameters:
        - r (float): Distance.
        - kappa (float): Harmonic potential parameter.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Harmonic potential energy.
        """
        return kappa*D*alpha**2*r**2

    @staticmethod
    def dharm(r, kappa, D, alpha):
        """
        Computes the derivative of the harmonic potential energy with respect to the distance.

        Parameters:
        - r (float): Distance.
        - kappa (float): Harmonic potential parameter.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Derivative of the harmonic potential energy with respect to the distance.
        """
        return 2.0*kappa*D*alpha**2*r

    def apply_lindemann_criteria(self, r, force, eta, tag='long'):
        """
        Filter the distance and force arrays based on the Lindemann criteria.

        Parameters:
        - r (numpy.ndarray): Array of distances.
        - force (numpy.ndarray): Array of forces.
        - eta (float, optional): The tolerance for Lindemann criteria filtering.
        - tag (str, optional): Tag to specify the criteria ('long' for longitudinal, 't1' or 't2' for transversal).

        Returns:
        - tuple: A tuple containing the filtered distance and force arrays.
        """
        if tag == 'long':
            mask = (self._b_0*(1-eta) <= r) & (r < self._b_0*(1+eta))
        else:
            mask = (-self._b_0*eta <= r) & (r < self._b_0*eta)
        return r[mask], force[mask]

    def get_potential_parameters(self, eta):
        """
        Estimate the Morse and harmonic potential parameters based on the force and distance data.

        Parameters:
        - eta (float, optional): The tolerance for Lindemann criteria filtering.

        Returns:
        - numpy.ndarray: An array of estimated potential parameters.
        """
        # parameterize longitudinal to morse
        long, long_force = self.apply_lindemann_criteria(self._data[0][0], self._data[0][1], eta, tag='long')
        popt_long, _ = curve_fit(f=self.dmorse, xdata=long, ydata=long_force, p0=(0.1, 1.5))
        # parameterize transversal to harmonic
        # t1
        def t_eqn(r, kappa):
            return self.dharm(r, kappa, D=popt_long[0], alpha=popt_long[1])
        t1, t1_force = self.apply_lindemann_criteria(self._data[1][0], self._data[1][1], eta, tag='t1')
        popt_t1, pcov_t1 = curve_fit(f=t_eqn, xdata=t1, ydata=t1_force, p0=0.1)
        # t2
        t2, t2_force = self.apply_lindemann_criteria(self._data[2][0], self._data[2][1], eta, tag='t2')
        popt_t2, pcov_t2 = curve_fit(f=t_eqn, xdata=t2, ydata=t2_force, p0=0.1)
        return np.array([popt_long[0], popt_long[1], popt_t1[0], popt_t2[0]])
    
    def get_parameterized_functions(self, D, alpha, kappa_1, kappa_2):
        """
        Compute the Morse and harmonic parameterized force and potential energy functions along the longitudinal and 2 
        transversal directions.

        Parameters:
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.
        - kappa_1 (float): Hamronic potential parameter for the t1 direction.
        - kappa_2 (float): Hamronic potential parameter for the t2 direction.

        Returns:
        - tuple: A tuple containing the parameterized potential energy functions for the longitudinal, t1, and t2 directions,
                 and the force functions for each direction.
        """
        def long_force(r):
            return self.dmorse(r, D=D, alpha=alpha)
        def long_potential(r):
            return self.morse(r, D=D, alpha=alpha)
        def t1_force(r):
            return self.dharm(r, D=D, alpha=alpha, kappa=kappa_1)
        def t1_potential(r):
            return self.harm(r, D=D, alpha=alpha, kappa=kappa_1)
        def t2_force(r):
            return self.dharm(r, D=D, alpha=alpha, kappa=kappa_2)
        def t2_potential(r):
            return self.harm(r, D=D, alpha=alpha, kappa=kappa_2)
        return long_potential, t1_potential, t2_potential, long_force, t1_force, t2_force

    def get_bonding_equations(self, eta=None, deg=None):
        """
        Retrieve functions for the bonding forces and potentials along the longitudinal and 2 transversal directions.

        Parameters:
        - eta (float, optional): If eta is not None, it specifies the tolerance for the Lindemann criteria filtering, and returns Morse 
        and harmonic fit functions. Otherwise, returns CubicSpline functions.
        - deg (int, optional): Degree of the polynomial fit (if eta is None) for the transversal functions.

        Returns:
        - tuple: A tuple containing the longitudinal, t1, and t2 bonding forces and potential functions.
        """
        self.get_force_on_j()
        data = self.get_all_bond_force_pot()
        if eta:
            parameters = self.get_potential_parameters(eta=eta)
            long_potential, t1_potential, t2_potential, long_force, t1_force, t2_force = self.get_parameterized_functions(*parameters)
        else:
            long_force, long_potential = self.get_spline_fit(data=self._data[0])
            if deg is not None:
                t1_force, t1_potential = self.get_poly_fit(data=self._data[1], deg=self.deg)
                t2_force, t2_potential = self.get_poly_fit(data=self._data[2], deg=self.deg)
            else:
                t1_force, t1_potential = self.get_spline_fit(data=self._data[1])
                t2_force, t2_potential = self.get_spline_fit(data=self._data[2])
        return long_potential, t1_potential, t2_potential, long_force, t1_force, t2_force
    
class GenerateLAPotential():
    """
    Generates the 'local anharmonic' (LA) potential and force functions for the 1st NN shell.
    
    This is done by displacing an atom i along the line that connects it to NN atom j (longitudinal direction between the 2), 
    and collecting the forces on atom j. From these forces, the central/radial 'bonding potential' between the 2 atoms can be 
    parameterized. Because of symmetry in the FCC crystal, forces on the NN atoms along the 2 transversal directions (perpendicular 
    to the longitudinal) can be also collected to parameterize the 2 transversal bonding potentials from the same longitudinal
    displacements.
    
    Parameters:
    - project_name (str): Name of the pyiron project for storing the jobs.
    - ref_job (pyiron Job): Reference job containing an appropriate FCC structure and an interatomic potential.
    - ith_atom_id (int, optional): Index of the atom that will be displaced (default is 0).
    - disp (float, optional): Maximum displacement for atom perturbations in lattice units (default is 1.0).
    - n_disps (int, optional): Number of displacement steps (default is 5).
    - uneven (bool, optional): Flag to use uneven spacing for displacement samples (default is False).
    - delete_existing_jobs (bool, optional): Flag to delete all existing jobs within the project. Cannot be used for individual jobs!
      (default is False)
    """
    
    def __init__(self, project_name, ref_job, ith_atom_id=0, disp=1., n_disps=5, uneven=False, delete_existing_jobs=False):
        
        self.project_name = project_name
        self.ref_job = ref_job
        self.structure = ref_job.structure.copy()
        self.ith_atom_id = ith_atom_id
        self.disp = disp
        self.n_disps = n_disps
        self.uneven = uneven
        self.delete_existing_jobs = delete_existing_jobs
        
        self._project = self.ref_job.project.create_group(self.project_name)
        self._stat_ba = None
        self._nn_bond_vecs = None
        self._nn_atom_ids = None
        self._long_pos = None
        self._t2_pos = None
        self._long_jobs = None
        self._t2_jobs = None
        self._plane_nn = None
        self._force_on_j_list = None
        self._data = None
        self._tags = ['long', 't2']
        
    def run_static_analysis(self):
        """
        Run static bond analysis on the reference job.
        """
        self._stat_ba = self._project.create_job(StaticBondAnalysis, 'stat_ba', 
                                                 delete_existing_job=self.delete_existing_jobs)
        self._stat_ba.input.structure = self.structure.copy()
        self._stat_ba.input.n_shells = 1
        self._stat_ba.run()

        self._b_0 = np.linalg.norm(self._stat_ba.output.per_shell_irreducible_bond_vectors, axis=-1)[0]
        self._long_hat, self._t1_hat, self._t2_hat = self._stat_ba.output.per_shell_transformation_matrices[0][0]
        self._nn_bond_vecs = self._long_hat@self._stat_ba.output.per_shell_0K_rotations[0]
        self._nn_atom_ids = self._stat_ba.output.per_shell_bond_indexed_neighbor_list[0][self.ith_atom_id]
        
    def get_plane_neighbors(self, output=False):
        """
        Retrieve jth atom indices along the longitudinal and t1 directions.
        """
        nn_bond_vecs = self._nn_bond_vecs
        anti_long_id = np.argwhere(np.isclose(a=nn_bond_vecs@self._long_hat, b=-1., atol=1e-10))[-1][-1]
        t1_id = np.argwhere(np.all(np.isclose(a=nn_bond_vecs, b=self._t1_hat, atol=1e-10), axis=-1))[-1][-1]
        self._plane_nn = np.array([self._nn_atom_ids[i] for i in [anti_long_id, 0, t1_id]])
        if output:
            return self._plane_nn
        
    @staticmethod
    def uneven_linspace(lb, ub, steps, spacing=1.1, endpoint=True):
        """
        Generate unevenly spaced samples using a power-law distribution with a specified spacing factor. 
        The power-law distribution allows for denser sampling near the lower bound if spacing > 1, and denser sampling towards 
        the upper bound if spacing < 1.

        Parameters:
        - lb (float): Lower bound of the range for generating samples.
        - ub (float): Upper bound of the range for generating samples.
        - steps (int): Number of samples to generate.
        - spacing (float, optional): Spacing factor controlling the distribution (default is 1.1).
        - endpoint (bool, optional): Whether to include the upper bound in the samples (default is True).

        Returns:
        - numpy.ndarray: An array of unevenly spaced samples.
        """
        span = (ub-lb)
        dx = 1.0/(steps-1)
        if not endpoint:
            dx = 1.0/(steps)
        return np.array([lb+(i*dx)**spacing*span for i in range(steps)])
    
    def generate_atom_positions(self, direction):
        """
        Generate atom positions corresponding to the displacements from equilibrium.
        """
        if self.uneven:
            samples = self.uneven_linspace(0., self.disp, self.n_disps, spacing=self.uneven)
        else:
            samples = np.linspace(0., self.disp, self.n_disps)
        
        ith_atom_pos = self.structure.positions[self.ith_atom_id]
        positions = np.array([ith_atom_pos+direction*s for s in samples])
        return positions
        
    def _validate_ready_to_run(self):
        if self._stat_ba is None:
            self.run_static_analysis()
        if self._long_pos is None:
            self._long_pos = self.generate_atom_positions(direction=self._long_hat)
        if self._t2_pos is None:
            self._t2_pos = self.generate_atom_positions(direction=self._t2_hat)
            
    def _run_job(self, project, job_name, position):
        job = self.ref_job.copy_template(project=project, new_job_name=job_name)
        job.structure.positions[self.ith_atom_id] = position
        job.calc_static()
        job.run()
    
    def run_disp_jobs(self):
        """
        Run displacement jobs for the perturbed atom positions.
        """
        self._validate_ready_to_run()
        for tag in self._tags:
            pr_tag = self._project.create_group(tag)
            job_list = pr_tag.job_table().job.to_list()
            job_status = pr_tag.job_table().status.to_list()
            
            if tag == 'long':
                positions = self._long_pos
            elif tag == 't2':
                positions = self._t2_pos
            else:
                raise ValueError
            
            for i, pos in enumerate(positions):
                job_name = tag + '_' + str(i)
                if job_name not in job_list:
                    self._run_job(project=pr_tag, job_name=job_name, position=pos)
                elif job_status[i] != 'finished' or self.delete_existing_jobs:
                    pr_tag.remove_job(job_name)
                    self._run_job(project=pr_tag, job_name=job_name, position=pos) 
        
    def load_disp_jobs(self):
        """
        Load displacement jobs from the project.
        """
        self._validate_ready_to_run()
        all_jobs = []
        for tag in self._tags:
            pr_tag = self._project.create_group(tag)
            all_jobs.append([pr_tag.inspect(tag + '_' + str(i)) for i in range(self.n_disps)])
        self._long_jobs, self._t2_jobs = all_jobs
    
    def _force_on_j(self, atom_id, tag='long'):
        if tag == 'long':
            return np.array([job['output/generic/forces'][-1][atom_id] for job in self._long_jobs])
        elif tag == 't2':
            return np.array([job['output/generic/forces'][-1][atom_id] for job in self._t2_jobs])
        else:
            raise ValueError
    
    def get_force_on_j(self, output=False):
        """
        Retrieve forces on the jth atoms at each dislacement.
        """
        if self._force_on_j_list is None:
            self.load_disp_jobs()
        if self._plane_nn is None:
            self.get_plane_neighbors()  
        force_on_j_list = [self._force_on_j(i) for i in self._plane_nn]
        force_on_j_list.append(self._force_on_j(self._plane_nn[1], tag='t2'))
        self._force_on_j_list = np.array(force_on_j_list)
        if output:
            return self._force_on_j_list
    
    @staticmethod
    def find_mic(structure, vectors):
        """
        Apply minimum image convention (MIC) to a set of vectors to account for periodic boundary conditions (PBC).

        Parameters:
        - vectors (array-like): An array of vectors.

        Returns:
        - array-like: An array of vectors with MIC applied.
        """
        cell = structure.cell
        pbc = structure.pbc
        vecs = np.asarray(vectors).reshape(-1, 3)
        if any(pbc):
            vecs = np.einsum('ji,nj->ni', np.linalg.inv(cell), vecs)
            vecs[:, pbc] -= np.rint(vecs)[:, pbc]
            vecs = np.einsum('ji,nj->ni', cell, vecs)
        return vecs.reshape(np.asarray(vectors).shape)
    
    @staticmethod
    def orthogonalize(matrix):
        """
        Orthogonalize a matrix using the Gram-Schmidt process.

        Parameters:
        - matrix (array-like): The matrix to be orthogonalized.

        Returns:
        - array-like: The orthogonalized matrix.
        """
        orth_matrix = matrix.copy()
        orth_matrix[1] = matrix[1]-(orth_matrix[0]@matrix[1])/(orth_matrix[0]@orth_matrix[0])*orth_matrix[0]
        orth_matrix[2] = matrix[2]-(orth_matrix[0]@matrix[2])/(orth_matrix[0]@orth_matrix[0])*orth_matrix[0] \
                                  -(orth_matrix[1]@matrix[2])/(orth_matrix[1]@orth_matrix[1])*orth_matrix[1]
        return orth_matrix/np.linalg.norm(orth_matrix, axis=-1)[:, np.newaxis]
    
    def get_ij_bond_force(self, jth_atom_id, jth_atom_forces, ith_atom_positions=None):
        """
        Compute the bonding forces along the line connecting atoms i and j (which is r). 
        Corresponds to the 'central' force. 

        Parameters:
        - i_atom_positions (array-like): Positions of the ith atom.
        - jth_atom_id (array-like): id of the jth atom.
        - j_atom_forces (array-like): Forces on the jth atom.

        Returns:
        - tuple: A tuple containing the magnitudes of displacements, bond forces, and displacement directions.
        """
        if ith_atom_positions is None:
            ith_atom_positions = self._long_pos
        j_position = self.structure.positions[jth_atom_id]
        ij = self.find_mic(self.structure, j_position-ith_atom_positions)
        r = np.linalg.norm(ij, axis=-1)
        ij_direcs = ij/r[:, np.newaxis]
        F_ij = (jth_atom_forces*ij_direcs).sum(axis=-1)
        return r, F_ij, ij_direcs
    
    def get_t_bond_force(self, jth_atom_id, jth_atom_forces, tag='t1'):
        """
        Compute the bonding forces along a transversal direction.
        
        For a displacement 'u' along a transversal direction, there is also a displacement r = sqrt(u**2+b_0**2) along ij or 'r' 
        or the 'central' direction. The force on atom j for a displacement u along a transversal direction is then F_j = F_t(u) +
        F_ij(r), where F_t(u) is the force along the transversal direction and F_ij(r) is the force along the ij.
        
        Further, for each transversal displacment u != 0, ij is not perpendicular to the original transversal direction. By using 
        the Gram-Schmit process, we can orthogonalize the transversal direction to ij and parameterize the forces along the new 
        transversal direction 't_prime' as 'F_t_prime'. While the effect of this orthogonalization is minimial, we by default will 
        consider this for the parameterization.

        Parameters:
        - jth_atom_id (array-like): id of the jth atom.
        - j_atom_forces (array-like): Forces on the jth atom.
        - tag (str): 't1' or 't2'.

        Returns:
        - tuple: A tuple containing the displacements u and bond forces F_t_prime along the 'tag' direction.
        """
        if tag == 't1':
            r, F_ij, ij_direcs = self.get_ij_bond_force(jth_atom_id, jth_atom_forces, self._long_pos)
            u = ij_direcs*r[:, np.newaxis]@self._long_hat
            basis = np.array([self._t1_hat, self._long_hat, self._t2_hat])
        elif tag == 't2':
            r, F_ij, ij_direcs = self.get_ij_bond_force(jth_atom_id, jth_atom_forces, self._t2_pos)
            u = ij_direcs*r[:, np.newaxis]@self._t2_hat
            basis = np.array([self._long_hat, self._t2_hat, self._t1_hat])
        else:
            raise ValueError
        F_t = jth_atom_forces-ij_direcs*F_ij[:, np.newaxis]
        F_t_prime = []
        for ij_dir, f_t in zip(ij_direcs, F_t):
            basis[0] = ij_dir
            t_prime = self.orthogonalize(basis.copy())[1]
            F_t_prime.append(f_t@t_prime)
        F_t_prime = np.array(F_t_prime)
        return u, F_t_prime
    
    def _bond_force(self, tag='long'):
        if tag == 'long':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id) 
                   for forces, atom_id in zip(self._force_on_j_list[:2], self._plane_nn[:2])]
            bonds = np.concatenate((np.flip(out[1][0]), out[0][0][1:]))
            force = np.concatenate((np.flip(out[1][1]), out[0][1][1:]))
        elif tag == 't1':
            out = self.get_t_bond_force(jth_atom_id=self._plane_nn[2], jth_atom_forces=self._force_on_j_list[2], tag=tag)
            bonds = np.concatenate((np.flip(out[0]), -out[0][1:]))
            force = np.concatenate((np.flip(out[1]), -out[1][1:]))
        elif tag == 't2':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[3], jth_atom_id=self._plane_nn[1], tag=tag) 
            bonds = np.concatenate((np.flip(out[0]), -out[0][1:]))
            force = np.concatenate((np.flip(out[1]), -out[1][1:]))
        else:
            raise ValueError
        return bonds, force
    
    def _bond_force_pot(self, tag='long'):
        bonds, force = self._bond_force(tag=tag)
    
        if tag == 'long':
            arg_b_0 = np.argmin(abs(bonds-self._b_0))
        elif tag in ['t1', 't2']:
            arg_b_0 = np.argmin(abs(bonds))
        else:
            raise ValueError
            
        up = cumtrapz(y=-force[arg_b_0:], x=bonds[arg_b_0:], initial=0.)
        down = np.flip(cumtrapz(y=-np.flip(force[:arg_b_0+1]), x=np.flip(bonds[:arg_b_0+1])))
        potential = np.concatenate((down, up))
        return bonds, force, potential
    
    def get_all_bond_force_pot(self, output=False):
        """
        Compute bonding forces and potentials along the longitudinal and 2 transversal directions.

        Returns:
        - list: A list of lists containing the bond lengths, bond forces, and potentials for each direction.
        """
        long_bonds, long_forces, long_potential = self._bond_force_pot(tag='long')
        t1_bonds, t1_forces, t1_potential = self._bond_force_pot(tag='t1')
        t2_bonds, t2_forces, t2_potential = self._bond_force_pot(tag='t2')
        self._data = [[long_bonds, -long_forces, long_potential],
                      [t1_bonds, -t1_forces, t1_potential],
                      [t2_bonds, -t2_forces, t2_potential]]
        if output:
            return self._data
    
    def get_spline_fit(self, data):
        """
        Perform a cubic spline fit to the given data.

        Parameters:
        - data (list): A list of bond lengths, bond forces, and potentials.

        Returns:
        - tuple: A tuple containing the cubic spline functions for bond force and potential.
        """
        force = CubicSpline(data[0], data[1])
        potential = CubicSpline(data[0], data[2])
        return force, potential
    
    def get_poly_fit(self, data, deg=4):
        """
        Perform a polynomial fit to the given data.

        Parameters:
        - data (list): A list of bond lengths, bond forces, and potentials.
        - deg (int): The degree of the polynomial fit (default is 4).

        Returns:
        - tuple: A tuple containing the polynomial functions for bond force and potential.
        """
        force = np.poly1d(np.polyfit(data[0], data[1], deg=deg))
        potential = np.poly1d(np.polyfit(data[0], data[2], deg=deg))
        return force, potential
    
    def morse(self, r, D, alpha):
        """
        Compute the Morse potential energy as a function of radial distance.

        Parameters:
        - r (float): Radial distance.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Morse potential energy.
        """
        return D*(1.0+np.exp(-2.0*alpha*(r-self._b_0))-2.0*np.exp(-alpha*(r-self._b_0)))

    def dmorse(self, r, D, alpha):
        """
        Computes the derivative of the Morse potential energy with respect to the radial distance.

        Parameters:
        - r (float): Radial distance.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Derivative of the Morse potential energy with respect to the distance.
        """
        return -2.0*alpha*D*(np.exp(-2.0*alpha*(r-self._b_0))-np.exp(-alpha*(r-self._b_0)))

    @staticmethod
    def harm(r, kappa, D, alpha):
        """
        Compute the harmonic potential energy as a function of distance.

        Parameters:
        - r (float): Distance.
        - kappa (float): Harmonic potential parameter.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Harmonic potential energy.
        """
        return kappa*D*alpha**2*r**2

    @staticmethod
    def dharm(r, kappa, D, alpha):
        """
        Computes the derivative of the harmonic potential energy with respect to the distance.

        Parameters:
        - r (float): Distance.
        - kappa (float): Harmonic potential parameter.
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.

        Returns:
        - float: Derivative of the harmonic potential energy with respect to the distance.
        """
        return 2.0*kappa*D*alpha**2*r

    def apply_lindemann_criteria(self, r, force, eta, tag='long'):
        """
        Filter the distance and force arrays based on the Lindemann criteria.

        Parameters:
        - r (numpy.ndarray): Array of distances.
        - force (numpy.ndarray): Array of forces.
        - eta (float, optional): The tolerance for Lindemann criteria filtering.
        - tag (str, optional): Tag to specify the criteria ('long' for longitudinal, 't1' or 't2' for transversal).

        Returns:
        - tuple: A tuple containing the filtered distance and force arrays.
        """
        if tag == 'long':
            mask = (self._b_0*(1-eta) <= r) & (r < self._b_0*(1+eta))
        else:
            mask = (-self._b_0*eta <= r) & (r < self._b_0*eta)
        return r[mask], force[mask]

    def get_potential_parameters(self, eta):
        """
        Estimate the Morse and harmonic potential parameters based on the force and distance data.

        Parameters:
        - eta (float, optional): The tolerance for Lindemann criteria filtering.

        Returns:
        - numpy.ndarray: An array of estimated potential parameters.
        """
        # parameterize longitudinal to morse
        long, long_force = self.apply_lindemann_criteria(self._data[0][0], self._data[0][1], eta, tag='long')
        popt_long, _ = curve_fit(f=self.dmorse, xdata=long, ydata=long_force, p0=(0.1, 1.5))
        # parameterize transversal to harmonic
        # t1
        def t_eqn(r, kappa):
            return self.dharm(r, kappa, D=popt_long[0], alpha=popt_long[1])
        t1, t1_force = self.apply_lindemann_criteria(self._data[1][0], self._data[1][1], eta, tag='t1')
        popt_t1, pcov_t1 = curve_fit(f=t_eqn, xdata=t1, ydata=t1_force, p0=0.1)
        # t2
        t2, t2_force = self.apply_lindemann_criteria(self._data[2][0], self._data[2][1], eta, tag='t2')
        popt_t2, pcov_t2 = curve_fit(f=t_eqn, xdata=t2, ydata=t2_force, p0=0.1)
        return np.array([popt_long[0], popt_long[1], popt_t1[0], popt_t2[0]])
    
    def get_parameterized_functions(self, D, alpha, kappa_1, kappa_2):
        """
        Compute the Morse and harmonic parameterized force and potential energy functions along the longitudinal and 2 
        transversal directions.

        Parameters:
        - D (float): Morse potential parameter.
        - alpha (float): Morse potential parameter.
        - kappa_1 (float): Hamronic potential parameter for the t1 direction.
        - kappa_2 (float): Hamronic potential parameter for the t2 direction.

        Returns:
        - tuple: A tuple containing the parameterized potential energy functions for the longitudinal, t1, and t2 directions,
                 and the force functions for each direction.
        """
        def long_force(r):
            return self.dmorse(r, D=D, alpha=alpha)
        def long_potential(r):
            return self.morse(r, D=D, alpha=alpha)
        def t1_force(r):
            return self.dharm(r, D=D, alpha=alpha, kappa=kappa_1)
        def t1_potential(r):
            return self.harm(r, D=D, alpha=alpha, kappa=kappa_1)
        def t2_force(r):
            return self.dharm(r, D=D, alpha=alpha, kappa=kappa_2)
        def t2_potential(r):
            return self.harm(r, D=D, alpha=alpha, kappa=kappa_2)
        return long_potential, t1_potential, t2_potential, long_force, t1_force, t2_force

    def get_bonding_equations(self, eta=None, deg=None):
        """
        Retrieve functions for the bonding forces and potentials along the longitudinal and 2 transversal directions.

        Parameters:
        - eta (float, optional): If eta is not None, it specifies the tolerance for the Lindemann criteria filtering, and returns Morse 
        and harmonic fit functions. Otherwise, returns CubicSpline functions.
        - deg (int, optional): Degree of the polynomial fit (if eta is None) for the transversal functions.

        Returns:
        - tuple: A tuple containing the longitudinal, t1, and t2 bonding forces and potential functions.
        """
        self.get_force_on_j()
        data = self.get_all_bond_force_pot()
        if eta:
            parameters = self.get_potential_parameters(eta=eta)
            long_potential, t1_potential, t2_potential, long_force, t1_force, t2_force = self.get_parameterized_functions(*parameters)
        else:
            long_force, long_potential = self.get_spline_fit(data=self._data[0])
            if deg is not None:
                t1_force, t1_potential = self.get_poly_fit(data=self._data[1], deg=self.deg)
                t2_force, t2_potential = self.get_poly_fit(data=self._data[2], deg=self.deg)
            else:
                t1_force, t1_potential = self.get_spline_fit(data=self._data[1])
                t2_force, t2_potential = self.get_spline_fit(data=self._data[2])
        return long_potential, t1_potential, t2_potential, long_force, t1_force, t2_force
