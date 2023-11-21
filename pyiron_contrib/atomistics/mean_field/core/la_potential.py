# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz

from pyiron_contrib.atomistics.mean_field.core.bond_analysis import StaticBondAnalysis
    
class GenerateFccLAPotential():
    """
    Generates the 'local anharmonic' (LA) potential and force functions for the 1st and 2nd NN shells of an FCC crystal.

    4 potential and 4 force functions are generated (l0_1nn, t1_1nn, t2_1nn, l0_2nn).
    
    An atom i is displaced along the line that connects it to a 1NN atom j, and the forces on atom j are collected. From these forces, the longitudinal 
    'bonding potential' between the 2 atoms can be parameterized. Because of the symmetry in FCC crystals, the potential on another 1NN atom on the same 
    plane, but along the transversal direction (t1) can also be parameterized. The NN atom perpendicular to this plane is a 2NN atom. Atom i is once again 
    displaced along the normal to the plane, now towards a 2NN atom, to parameterize the out-of-plane transversal potential (t2). From the same displacements, 
    the longitudinal potential along the line connecting atom i to atom j (now, a 2NN neighbor) is parameterized.
    
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
        self._b0_1nn = None
        self._b0_2nn = None
        self._b0_3nn = None
        self._rotations_1nn = None
        self._rotations_2nn = None
        self._rotations_3nn = None
        self._basis_1nn = None
        self._basis_2nn = None
        self._basis_3nn = None
        self._nn_bond_vecs = None
        self._nn_atom_ids = None
        self._pos_1nn = None
        self._pos_2nn = None
        self._pos_3nn = None
        self._jobs_1nn = None
        self._jobs_2nn = None
        self._jobs_3nn = None
        self._in_plane_nn = None
        self._out_plane_nn = None
        self._out_plane_nn_2 = None
        self._force_on_j_list = None
        self._data = None
        self._tags = ['1', '2', '3']

    @staticmethod
    def _get_rotation_matrix(vec1, vec2):
        a, b = (vec1/np.linalg.norm(vec1)).reshape(3), (vec2/np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        if any(v): # if not all zeros then 
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        elif np.allclose(vec1, -vec2): # if vectors are in opposite direction
            return -np.eye(3)
        else:
            return np.eye(3) # if vectors are in the same direction
        
    def run_static_analysis(self):
        """
        Run static bond analysis on the reference job and assign hidden variables.
        """
        self._stat_ba = self._project.create_job(StaticBondAnalysis, 'static_ba', delete_existing_job=self.delete_existing_jobs)
        self._stat_ba.input.structure = self.structure.copy()
        self._stat_ba.input.n_shells = 3
        self._stat_ba.run()

        self._b0_1nn, self._b0_2nn, self._b0_3nn = np.linalg.norm(self._stat_ba.output.per_shell_irreducible_bond_vectors, axis=-1)
        self._rotations_1nn, rotations_2nn, self._rotations_3nn = self._stat_ba.output.per_shell_0K_rotations
        atom_ids_1nn = self._stat_ba.output.per_shell_bond_indexed_neighbor_list[0][self.ith_atom_id]
        atom_ids_2nn = self._stat_ba.output.per_shell_bond_indexed_neighbor_list[1][self.ith_atom_id]
        atom_ids_3nn = self._stat_ba.output.per_shell_bond_indexed_neighbor_list[2][self.ith_atom_id]

        # 1nn basis, strightforward
        self._basis_1nn = self._stat_ba.output.per_shell_transformation_matrices[0][0]

        # 2nn basis, needs rearranging such that t2_1nn == l_2nn
        basis_2nn = self._stat_ba.output.per_shell_transformation_matrices[1][0]
        roll_arg = np.argwhere(np.all(np.isclose(a=(basis_2nn@rotations_2nn)[:, 0], b=self._basis_1nn[2], atol=1e-10), axis=-1))[-1][-1]
        roll_id = int(len(rotations_2nn) - roll_arg)
        self._basis_2nn = self._stat_ba.output.per_shell_transformation_matrices[1][roll_arg]
        rolled_vecs = np.roll(basis_2nn[0]@rotations_2nn, roll_id, axis=0)
        self._rotations_2nn = np.array([self._get_rotation_matrix(rolled_vecs[0], rolled_vecs[i]) for i in range(len(rotations_2nn))])

        # 3nn basis, strightforward
        self._basis_3nn = self._stat_ba.output.per_shell_transformation_matrices[2][0]

        self._nn_atom_ids = [atom_ids_1nn, np.roll(atom_ids_2nn, roll_id), atom_ids_3nn]
        self._nn_bond_vecs = [self._basis_1nn[0]@self._rotations_1nn, self._basis_2nn[0]@self._rotations_2nn, 
                              self._basis_3nn[0]@self._rotations_3nn]
        
        
    def get_plane_neighbors(self, output=False):
        """
        Retrieve atom indices (jth atom) for in-plane and out-of-plane NNs. 
        """
        anti_l_id_1nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[0]@self._basis_1nn[0], b=-1., atol=1e-10))[-1][-1]
        t1_id_1nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[0], b=self._basis_1nn[1], atol=1e-10), axis=-1))[-1][-1]
        anti_l_id_2nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[1]@self._basis_2nn[0], b=-1., atol=1e-10))[-1][-1]
        t1_id_2nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[1], b=self._basis_2nn[1], atol=1e-10), axis=-1))[-1][-1]
        t2_id_2nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[1], b=self._basis_2nn[2], atol=1e-10), axis=-1))[-1][-1]
        anti_l_id_3nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[2]@self._basis_3nn[0], b=-1., atol=1e-10))[-1][-1]
        self._in_plane_nn = np.array([self._nn_atom_ids[0][i] for i in [anti_l_id_1nn, 0, t1_id_1nn]])
        self._out_plane_nn = np.array([self._nn_atom_ids[1][i] for i in [anti_l_id_2nn, 0, t1_id_2nn, t2_id_2nn]])
        self._out_plane_nn_2 = np.array([self._nn_atom_ids[2][i] for i in [anti_l_id_3nn, 0]])
        if output:
            return self._in_plane_nn, self._out_plane_nn, self._out_plane_nn_2
        
    @staticmethod
    def uneven_linspace(lb, ub, steps, spacing=1.1, endpoint=True):
        """
        Generate unevenly spaced samples using a power-law distribution with a specified spacing factor. 
        The power-law distribution allows for denser sampling near the lower bound if spacing > 1, and denser sampling towards the upper bound if spacing < 1.

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
        if self._pos_1nn is None:
            self._pos_1nn = self.generate_atom_positions(direction=self._basis_1nn[0])
        if self._pos_2nn is None:
            self._pos_2nn = self.generate_atom_positions(direction=self._basis_2nn[0])
        if self._pos_3nn is None:
            self._pos_3nn = self.generate_atom_positions(direction=self._basis_3nn[0])
            
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
            pr_tag = self._project.create_group('NN' + tag)
            job_list = pr_tag.job_table().job.to_list()
            job_status = pr_tag.job_table().status.to_list()
            
            if tag == '1':
                positions = self._pos_1nn
            elif tag == '2':
                positions = self._pos_2nn
            elif tag == '3':
                positions = self._pos_3nn
            else:
                raise ValueError
            
            for i, pos in enumerate(positions):
                job_name = 'NN' + tag + '_' + str(i)
                if job_name not in job_list:
                    self._run_job(project=pr_tag, job_name=job_name, position=pos)
                elif job_status[i] not in ['finished', 'warning', 'running'] or self.delete_existing_jobs:
                    pr_tag.remove_job(job_name)
                    self._run_job(project=pr_tag, job_name=job_name, position=pos) 
        
    def load_disp_jobs(self):
        """
        Load displacement jobs from the project.
        """
        self._validate_ready_to_run()
        all_jobs = []
        for tag in self._tags:
            pr_tag = self._project.create_group('NN' + tag)
            all_jobs.append([pr_tag.inspect('NN' + tag + '_' + str(i)) for i in range(self.n_disps)])
        self._jobs_1nn, self._jobs_2nn, self._jobs_3nn = all_jobs
    
    def _force_on_j(self, atom_id, tag='1'):
        if tag == '1':
            return np.array([job['output/generic/forces'][-1][atom_id] for job in self._jobs_1nn])
        elif tag == '2':
            return np.array([job['output/generic/forces'][-1][atom_id] for job in self._jobs_2nn])
        elif tag == '3':
            return np.array([job['output/generic/forces'][-1][atom_id] for job in self._jobs_3nn])
        else:
            raise ValueError
    
    def get_force_on_j(self, output=False):
        """
        Retrieve forces on the jth atoms at each dislacement.
        """
        if self._force_on_j_list is None:
            self.load_disp_jobs()
        if self._in_plane_nn is None:
            self.get_plane_neighbors()  
        force_on_j_list = [self._force_on_j(i, tag='1') for i in self._in_plane_nn]
        force_on_j_list.append(self._force_on_j(self._in_plane_nn[1], tag='2'))
        for i in self._out_plane_nn:
            force_on_j_list.append(self._force_on_j(i, tag='2'))
        for i in self._out_plane_nn_2:
            force_on_j_list.append(self._force_on_j(i, tag='3'))
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
    
    def get_ij_bond_force(self, jth_atom_id, jth_atom_forces, ith_atom_positions):
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
        j_position = self.structure.positions[jth_atom_id]
        ij = self.find_mic(self.structure, j_position-ith_atom_positions)
        r = np.linalg.norm(ij, axis=-1)
        ij_direcs = ij/r[:, np.newaxis]
        F_ij = (jth_atom_forces*ij_direcs).sum(axis=-1)
        return r, F_ij, ij_direcs
    
    def get_t_bond_force(self, jth_atom_id, jth_atom_forces, tag='1', nn='1', return_prime=False):
        """
        Compute the bonding forces along a transversal direction.
        
        For a displacement 'u' along a transversal direction, there is also a displacement r = sqrt(u**2+b0**2) along ij or 'r' 
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
        - return_prime (bool): Whether to return the t_prime directions, for debugging. 

        Returns:
        - tuple: A tuple containing the displacements u and bond forces F_t_prime along the 'tag' direction.
        """
        if nn == '1':
            l_hat, t1_hat, t2_hat = self._basis_1nn
            pos_t1, pos_t2 = self._pos_1nn, self._pos_2nn
        elif nn == '2':
            l_hat, t1_hat, t2_hat = self._basis_2nn
            pos_t1, pos_t2 = self._pos_2nn, self._pos_2nn
        else:
            raise ValueError
        if tag == '1':
            r, F_ij, ij_direcs = self.get_ij_bond_force(jth_atom_id, jth_atom_forces, pos_t1)
            u = ij_direcs*r[:, np.newaxis]@l_hat
            basis = np.array([t1_hat, l_hat, t2_hat])
        elif tag == '2':
            r, F_ij, ij_direcs = self.get_ij_bond_force(jth_atom_id, jth_atom_forces, pos_t2)
            if nn == '1':
                u = ij_direcs*r[:, np.newaxis]@t2_hat
                basis = np.array([l_hat, t2_hat, t1_hat])
            elif nn == '2':
                u = ij_direcs*r[:, np.newaxis]@l_hat
                basis = np.array([t2_hat, l_hat, t1_hat])
            else:
                raise ValueError
        else:
            raise ValueError
        F_t = jth_atom_forces-ij_direcs*F_ij[:, np.newaxis]
        F_t_prime = []
        prime = []
        for ij_dir, f_t in zip(ij_direcs, F_t):
            new_basis = basis.copy()
            new_basis[0] = ij_dir
            t_prime = self.orthogonalize(new_basis)[1]
            F_t_prime.append(f_t@t_prime)
            prime.append(t_prime)
        if return_prime:
            return u, np.array(F_t_prime), np.array(prime)
        return u, np.array(F_t_prime)
    
    def _bond_force(self, tag='l_1nn'):
        if tag == 'l_1nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos_1nn) 
                   for forces, atom_id in zip(self._force_on_j_list[:2], self._in_plane_nn[:2])]
            bonds = np.concatenate((np.flip(out[1][0]), out[0][0][1:]))
            force = np.concatenate((np.flip(out[1][1]), out[0][1][1:]))
        elif tag == 't1_1nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[2], jth_atom_id=self._in_plane_nn[2], tag='1', nn='1')
            bonds = np.concatenate((np.flip(out[0]), -out[0][1:]))
            force = np.concatenate((np.flip(out[1]), -out[1][1:]))
        elif tag == 't2_1nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[3], jth_atom_id=self._in_plane_nn[1], tag='2', nn='1') 
            bonds = np.concatenate((np.flip(out[0]), -out[0][1:]))
            force = np.concatenate((np.flip(out[1]), -out[1][1:]))
        elif tag == 'l_2nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos_2nn) 
                   for forces, atom_id in zip(self._force_on_j_list[4:6], self._out_plane_nn[:2])]
            bonds = np.concatenate((np.flip(out[1][0]), out[0][0][1:]))
            force = np.concatenate((np.flip(out[1][1]), out[0][1][1:]))
        elif tag == 't1_2nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[6], jth_atom_id=self._out_plane_nn[2], tag='1', nn='2')
            bonds = np.concatenate((np.flip(out[0]), -out[0][1:]))
            force = np.concatenate((np.flip(out[1]), -out[1][1:]))
        elif tag == 't2_2nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[7], jth_atom_id=self._out_plane_nn[3], tag='2', nn='2') 
            bonds = np.concatenate((np.flip(out[0]), -out[0][1:]))
            force = np.concatenate((np.flip(out[1]), -out[1][1:]))
        elif tag == 'l_3nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos_3nn) 
                   for forces, atom_id in zip(self._force_on_j_list[8:10], self._out_plane_nn_2[:2])]
            bonds = np.concatenate((np.flip(out[1][0]), out[0][0][1:]))
            force = np.concatenate((np.flip(out[1][1]), out[0][1][1:]))
        else:
            raise ValueError
        return bonds, force
    
    def _bond_force_pot(self, tag):
        bonds, force = self._bond_force(tag=tag)
    
        if tag == 'l_1nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0_1nn))
        elif tag in ['t1_1nn', 't2_1nn', 't1_2nn', 't2_2nn']:
            arg_b_0 = np.argmin(abs(bonds))
        elif tag == 'l_2nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0_2nn))
        elif tag == 'l_3nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0_3nn))
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
        l_1nn_bonds, l_1nn_forces, l_1nn_potential = self._bond_force_pot(tag='l_1nn')
        t1_1nn_bonds, t1_1nn_forces, t1_1nn_potential = self._bond_force_pot(tag='t1_1nn')
        t2_1nn_bonds, t2_1nn_forces, t2_1nn_potential = self._bond_force_pot(tag='t2_1nn')
        l_2nn_bonds, l_2nn_forces, l_2nn_potential = self._bond_force_pot(tag='l_2nn')
        t1_2nn_bonds, t1_2nn_forces, t1_2nn_potential = self._bond_force_pot(tag='t1_2nn')
        t2_2nn_bonds, t2_2nn_forces, t2_2nn_potential = self._bond_force_pot(tag='t2_2nn')
        l_3nn_bonds, l_3nn_forces, l_3nn_potential = self._bond_force_pot(tag='l_3nn')
        self._data = [[l_1nn_bonds , -l_1nn_forces , l_1nn_potential ],
                      [t1_1nn_bonds, -t1_1nn_forces, t1_1nn_potential],
                      [t2_1nn_bonds, -t2_1nn_forces, t2_1nn_potential],
                      [l_2nn_bonds , -l_2nn_forces , l_2nn_potential ],
                      [t1_2nn_bonds, -t1_2nn_forces, t1_2nn_potential],
                      [t2_2nn_bonds, -t2_2nn_forces, t2_2nn_potential],
                      [l_3nn_bonds , -l_3nn_forces , l_3nn_potential ]]
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

    def get_bonding_functions(self, deg=None):
        """
        Retrieve functions for the bonding forces and potentials along the longitudinal and 2 transversal directions.

        Parameters:
        - deg (int, optional): Degree of the polynomial fit (if eta is None) for the transversal functions. Otherwise, returns CubicSpline functions.

        Returns:
        - tuple: A tuple containing the longitudinal, t1, and t2 bonding forces and potential functions.
        """
        self.get_force_on_j()
        data = self.get_all_bond_force_pot()
        l_1nn_force, l_1nn_potential = self.get_spline_fit(data=self._data[0])
        l_2nn_force, l_2nn_potential = self.get_spline_fit(data=self._data[3])
        l_3nn_force, l_3nn_potential = self.get_spline_fit(data=self._data[6])
        if deg is not None:
            t_forces, t_potentials = zip(*[self.get_poly_fit(data=self._data[i], deg=self.deg) for i in [1, 2, 4, 5]])
        else:
            t_forces, t_potentials = zip(*[self.get_spline_fit(data=self._data[i]) for i in [1, 2, 4, 5]])

        return [[l_1nn_potential, *t_potentials[:2], l_2nn_potential, *t_potentials[2:4], l_3nn_potential],
                [l_1nn_force,     *t_forces[:2],    l_2nn_force,     *t_forces[2:4], l_3nn_force]]
