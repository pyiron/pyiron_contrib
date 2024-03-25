# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz

from pyiron_contrib.atomistics.mean_field.core.bond_analysis import StaticBondAnalysis

### TODO: Update documentation!

class GenerateLAPotential():
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
    
    def __init__(self, project_name, ref_job, potential, ith_atom_id=0, shells=1, disp=1., n_disps=5, uneven=False, 
                 delete_existing_jobs=False, delete_static_job=False):
        
        self.project_name = project_name
        self.potential = potential
        self.ref_job = ref_job
        self.ith_atom_id = ith_atom_id
        self.shells = shells
        self.disp = disp
        self.n_disps = n_disps
        self.uneven = uneven
        self.delete_existing_jobs = delete_existing_jobs
        self.delete_static_job = delete_static_job
        self.structure = ref_job.structure.copy()
        self.disp_dir = None
        
        self._project = self.ref_job.project.create_group(self.project_name)
        self._stat_ba = None
        self._b0 = None
        self._rotations = None
        self._basis = None
        self._nn_bond_vecs = None
        self._nn_atom_ids = None
        self._pos = None
        self._jobs = None
        self._plane_nn = None
        self._force_on_j_list = None
        self._data = None
        self._tags = None

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
        
    @staticmethod
    def _find_rotation_matrix(A, B):
        """
        Find the rotation matrix that rotates matrix A to matrix B.
        """
        # Compute the covariance matrix
        H = np.dot(B.T, A)
        # Perform Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(H)
        # Compute the rotation matrix
        R = np.dot(Vt.T, U.T)
        return R
        
    def run_static_analysis(self):
        pass
        
    def get_plane_neighbors(self):
        pass
        
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
    
    def generate_atom_positions(self, direction, low_disp=0, spacing=None, asymmetric=False):
        """
        Generate atom positions corresponding to the displacements from equilibrium.
        """
        if spacing:
            if asymmetric:
                right = self.uneven_linspace(low_disp, self.disp, self.n_disps, spacing=spacing)
                samples = np.concatenate((np.flip(-right), right[1:]))
            else:
                samples = self.uneven_linspace(low_disp, self.disp, self.n_disps, spacing=spacing)
        else:
            samples = np.linspace(low_disp, self.disp, self.n_disps)
            
        # make sure there is always a 0 displacement
        if not np.any(np.isclose(samples, 0., atol=1e-10)):
            samples = np.insert(samples, np.argmin(abs(samples)), 0.)
            samples = np.delete(samples, -1)

        ith_atom_pos = self.structure.positions[self.ith_atom_id]
        positions = np.array([ith_atom_pos+direction*s for s in samples])
        return positions
        
    def _validate_ready_to_run(self):
        pass
            
    def _run_job(self, project, job_name, position):
        job = self.ref_job.copy_template(project=project, new_job_name=job_name)
        job.structure.positions[self.ith_atom_id] = position
        if self.potential is not None:
            job.potential = self.potential
        job.calc_static()
        job.run()
    
    def run_disp_jobs(self):
        """
        Run displacement jobs for the perturbed atom positions.
        """
        self._validate_ready_to_run()
        for tag in self._tags:
            pr_tag = self._project.create_group('disp_dir_' + tag)
            job_list = pr_tag.job_table().job.to_list()
            positions = self._pos[int(tag)]

            for i, pos in enumerate(positions):
                job_name = 'disp_dir_' + tag + '_' + str(i)
                if job_name not in job_list:
                    self._run_job(project=pr_tag, job_name=job_name, position=pos)
                else:
                    job = pr_tag.inspect(job_name)
                    if job.status in ['aborted'] or self.delete_existing_jobs:
                        pr_tag.remove_job(job_name)
                        self._run_job(project=pr_tag, job_name=job_name, position=pos) 
        
    def load_disp_jobs(self):
        """
        Load displacement jobs from the project.
        """
        self._validate_ready_to_run()
        jobs = []
        for tag in self._tags:
            pr_tag = self._project.create_group('disp_dir_' + tag)
            jobs.append([pr_tag.inspect('disp_dir_' + tag + '_' + str(i)) for i in range(len(self._pos[int(tag)]))])
        self._jobs = jobs
    
    def _force_on_j(self, atom_id, tag='0'):
        if tag not in self._tags:
            raise ValueError
        return np.array([job['output/generic/forces'][-1][atom_id] for job in self._jobs[int(tag)]])
    
    def get_force_on_j(self):
        pass
    
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
        jth_atom_position = self.structure.positions[jth_atom_id]
        ij = self.find_mic(self.structure, jth_atom_position-ith_atom_positions)
        r = np.linalg.norm(ij, axis=-1)
        ij_direcs = ij/r[:, np.newaxis]
        F_ij = (jth_atom_forces*ij_direcs).sum(axis=-1)
        return r, F_ij, ij_direcs
    
    def get_t_bond_force(self):
       pass
    
    def _bond_force(self):
        pass
    
    def _bond_force_pot(self, tag):
        bonds, force = self._bond_force(tag=tag)
    
        if tag == 'l_1nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0[0]))
        elif tag == 'l_2nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0[1]))
        elif tag == 'l_3nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0[2]))
        elif tag == 'l_4nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0[3]))
        elif tag == 'l_5nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0[4]))
        elif tag == 'l_6nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0[5]))
        elif tag == 'l_7nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0[6]))
        elif tag == 'l_8nn':
            arg_b_0 = np.argmin(abs(bonds-self._b0[7]))
        elif tag in ['t1_1nn', 't2_1nn', 't1_2nn', 't2_2nn', 't1_3nn', 't2_3nn', 't1_4nn', 't2_4nn', 't1_5nn', 't2_5nn', 't1_6nn', 't2_6nn', 't1_7nn', 't2_7nn', 't1_8nn', 't2_8nn']:
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
        directions = ['l', 't1', 't2']
        self._data = []
        
        for shell in range(1, self.shells+1):
            shell_data = []
            
            for direction in directions:
                tag = f'{direction}_{shell}nn'
                bonds, forces, potential = self._bond_force_pot(tag=tag)
                shell_data.append([bonds, -forces, potential])
            
            self._data.extend(shell_data)
        
        return self._data if output else None
    
    def get_spline_fit(self, data):
        """
        Perform a cubic spline fit to the given data.

        Parameters:
        - data (list): A list of bond lengths, bond forces, and potentials.

        Returns:
        - tuple: A tuple containing the cubic spline functions for bond force and potential.
        """
        force = CubicSpline(data[0], data[1], bc_type='natural')
        potential = CubicSpline(data[0], data[2], bc_type='natural')
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
        self.get_all_bond_force_pot()

        potential_func = [[], [], []]
        force_func = [[], [], []]

        for i in range(self.shells):
            l_nn_force, l_nn_potential = self.get_spline_fit(data=self._data[i*3])
            t_nn_forces, t_nn_potentials = zip(*[self.get_spline_fit(data=self._data[i*3+j]) for j in range(1, 3)])

            potential_func[0].append(l_nn_potential)
            potential_func[1].append(t_nn_potentials[0])
            potential_func[2].append(t_nn_potentials[1])
            force_func[0].append(l_nn_force)
            force_func[1].append(t_nn_forces[0])
            force_func[2].append(t_nn_forces[1])

        return potential_func, force_func
    
    @staticmethod
    def _concatenate_t(out):
        return np.concatenate((np.flip(out[0]), -out[0][1:])), np.concatenate((np.flip(out[1]), -out[1][1:]))
    
    @staticmethod
    def _concatenate_t_special(out):
        return -out[0], -out[1]
    
    @staticmethod
    def _concatenate_l(out):
        return np.concatenate((np.flip(out[1][0]), out[0][0][1:])), np.concatenate((np.flip(out[1][1]), out[0][1][1:]))
    
class FccLAPotential(GenerateLAPotential):
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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run_static_analysis(self):
        """
        Run static bond analysis on the reference job and assign hidden variables.
        """
        if self.delete_existing_jobs:
            self.delete_static_job = True
        self._stat_ba = self._project.create_job(StaticBondAnalysis, 'static_ba', delete_existing_job=self.delete_static_job)
        self._stat_ba.input.structure = self.structure.copy()
        self._stat_ba.input.n_shells = self.shells
        self._stat_ba.run()

        self._b0 = np.linalg.norm(self._stat_ba.output.per_shell_irreducible_bond_vectors, axis=-1)
        self._rotations = self._stat_ba.output.per_shell_0K_rotations
        atom_ids = [self._stat_ba.output.per_shell_bond_indexed_neighbor_list[i][self.ith_atom_id]
                    for i in range(self.shells)]

        # 1nn basis, strightforward
        self._basis = [self._stat_ba.output.per_shell_transformation_matrices[0][0]]
        self._nn_atom_ids = (atom_ids[0]).tolist()
        self._nn_bond_vecs = (self._basis[0][0]@self._rotations[0]).tolist()
        self.disp_dir = [self._basis[0][0]]

        def update_basis_rotations(s, basis, atom_ids, a, b):
            """
            Update the basis rotations.

            Args:
                s (int): Index of the shell.
                basis (numpy.ndarray): Basis array.
                atom_ids (list): List of atom IDs.
                a (int): Index of the column in the basis array.
                b (float): Value to compare with.

            Returns:
                None
            """
            roll_arg = np.argwhere(np.all(np.isclose(a=(basis@self._rotations[s])[:, a], b=b, atol=1e-10), axis=-1))[0][-1]
            roll_id = int(len(self._rotations[s]) - roll_arg)
            self._basis.append((basis@self._rotations[s])[roll_arg])
            rolled_vecs = np.roll(basis[0]@self._rotations[s], roll_id, axis=0)
            rolled_basis = np.roll(basis@self._rotations[s], roll_id, axis=0)
            self._rotations[s] = np.array([self._find_rotation_matrix(rolled_basis[0], rb) for rb in rolled_basis])
            self._nn_atom_ids += np.roll(atom_ids[s], roll_id).tolist()
            self._nn_bond_vecs += (rolled_vecs).tolist()

        if self.shells >= 2:
            s = 1
            basis = self._stat_ba.output.per_shell_transformation_matrices[s][0]
            update_basis_rotations(s, basis, atom_ids, a=0, b=self._basis[0][2])
            self.disp_dir.append(self._basis[s][0])

        if self.shells >= 3:
            s = 2
            basis = self._stat_ba.output.per_shell_transformation_matrices[s][0]
            update_basis_rotations(s, basis, atom_ids, a=1, b=self._basis[0][0])
            self.disp_dir.extend([self._basis[s][0], self._basis[s][2]])

        if self.shells >= 4:
            s = 3
            self._basis.append(self._stat_ba.output.per_shell_transformation_matrices[s][0])
            self._nn_atom_ids += (atom_ids[s]).tolist()
            self._nn_bond_vecs += (self._basis[s][0]@self._rotations[s]).tolist()

        if self.shells >= 5:
            s = 4
            basis = self._stat_ba.output.per_shell_transformation_matrices[s][0]
            update_basis_rotations(s, basis, atom_ids, a=2, b=self._basis[0][2])
            self.disp_dir.append(self._basis[s][0])

        if self.shells == 6:
            s = 5
            basis = self._stat_ba.output.per_shell_transformation_matrices[s][0]
            update_basis_rotations(s, basis, atom_ids, a=0, b=self._basis[2][2])
            
        self._nn_bond_vecs = np.array(self._nn_bond_vecs)
        self._nn_atom_ids = np.array(self._nn_atom_ids)
        
    def get_plane_neighbors(self, output=False):
        """
        Retrieve atom indices (jth atom) for in-plane and out-of-plane NNs. 
        """
        anti_l_id_1nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:12]@self._basis[0][0], b=-1., atol=1e-10))[-1][-1]
        t1_id_1nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[:12], b=self._basis[0][1], atol=1e-10), axis=-1))[-1][-1]
        self._plane_nn = [[self._nn_atom_ids[i] for i in [anti_l_id_1nn, 0, t1_id_1nn]]]
        
        if self.shells >= 2:
            anti_l_id_2nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:18]@self._basis[1][0], b=-1., atol=1e-10))[-1][-1]
            l_id_2nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:18]@self._basis[1][0], b=1., atol=1e-10))[-1][-1]
            t1_id_2nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[:18], b=self._basis[1][1], atol=1e-10), axis=-1))[-1][-1]
            t2_id_2nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[:18], b=self._basis[1][2], atol=1e-10), axis=-1))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i] for i in [anti_l_id_2nn, l_id_2nn, t1_id_2nn, t2_id_2nn]])
        
        if self.shells >= 3:
            anti_l_id_3nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:42]@self._basis[2][0], b=-1., atol=1e-10))[-1][-1]
            l_id_3nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:42]@self._basis[2][0], b=1., atol=1e-10))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i] for i in [anti_l_id_3nn, l_id_3nn]])

        if self.shells >= 4:
            anti_l_id_4nn = np.argwhere(np.isclose(a=self._nn_bond_vecs@self._basis[3][0], b=-1., atol=1e-10))[-1][-1]
            l_id_4nn = np.argwhere(np.isclose(a=self._nn_bond_vecs@self._basis[3][0], b=1., atol=1e-10))[-1][-1]
            t1_id_4nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs, b=self._basis[3][1], atol=1e-10), axis=-1))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i] for i in [anti_l_id_4nn, l_id_4nn, t1_id_4nn]])

        if self.shells >= 5:
            anti_l_id_5nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[54:]@self._basis[4][0], b=-1., atol=1e-10))[-1][-1]
            l_id_5nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[54:]@self._basis[4][0], b=1., atol=1e-10))[-1][-1]
            t1_id_5nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[54:], b=self._basis[4][1], atol=1e-10), axis=-1))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i+54] for i in [anti_l_id_5nn, l_id_5nn, t1_id_5nn]])

        if self.shells == 6:
            anti_l_id_6nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[54:]@self._basis[5][0], b=-1., atol=1e-10))[-1][-1]
            l_id_6nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[54:]@self._basis[5][0], b=1., atol=1e-10))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i+54] for i in [anti_l_id_6nn, l_id_6nn]])
        
        if output:
            return self._plane_nn
        
    def _validate_ready_to_run(self):
        if self.shells == 1:
            raise ValueError('At least 2 shells are required, in order to parameterize t2_1nn.')
        if self._stat_ba is None:
            self.run_static_analysis()
        if self._pos is None:
            self._pos = [self.generate_atom_positions(direction=self.disp_dir[0], spacing=self.uneven)]
            
            if self.shells >= 2:
                self._pos.append(self.generate_atom_positions(direction=self.disp_dir[1], spacing=self.uneven))
                
            if self.shells >= 3:
                self._pos.append(self.generate_atom_positions(direction=self.disp_dir[2], spacing=self.uneven))
                self._pos.append(self.generate_atom_positions(direction=self.disp_dir[3], spacing=self.uneven, asymmetric=True))
                # self._pos.extend([self.generate_atom_positions(direction=self.disp_dir[i], spacing=self.uneven) for i in range(2, 4)])

            if self.shells >= 5:
                self._pos.append(self.generate_atom_positions(direction=self.disp_dir[4], spacing=self.uneven))

        # set the tags to correspond to the directions along which static calcuations are actually performed
        self._tags = np.arange(len(self._pos)).astype(str).tolist()
    
    def get_force_on_j(self, output=False):
        """
        Retrieve forces on the jth atoms at each dislacement.
        """
        if self._force_on_j_list is None:
            self.load_disp_jobs()
        if self._plane_nn is None:
            self.get_plane_neighbors()  
        
        self._force_on_j_list = [self._force_on_j(i, tag='0') for i in self._plane_nn[0]]
        self._force_on_j_list.append(self._force_on_j(self._plane_nn[0][1], tag='1'))
        
        if self.shells >= 2:
            for i in self._plane_nn[1]:
                self._force_on_j_list.append(self._force_on_j(i, tag='1'))
                
        if self.shells >= 3:
            for i in self._plane_nn[2]:
                self._force_on_j_list.append(self._force_on_j(i, tag='2'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[2][1], tag='0'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[2][1], tag='3'))

        if self.shells >= 4:
            for i in self._plane_nn[3]:
                self._force_on_j_list.append(self._force_on_j(i, tag='0'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[3][1], tag='1'))

        if self.shells >= 5:
            for i in self._plane_nn[4]:
                self._force_on_j_list.append(self._force_on_j(i, tag='4'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[4][1], tag='1'))

        if self.shells == 6:
            for i in self._plane_nn[5]:
                self._force_on_j_list.append(self._force_on_j(i, tag='3'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[5][1], tag='0'))
            self._force_on_j_list.append(-self._force_on_j(self._plane_nn[5][1], tag='2'))  # the displacements for 6nn_t2 point away from 3nn_l
        
        if output:
            return self._force_on_j_list
    
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
        l_hat, t1_hat, t2_hat = self._basis[int(nn)-1]
        if nn == '1':
            pos_t1, pos_t2 = self._pos[0], self._pos[1]
        elif nn == '2':
            pos_t1, pos_t2 = self._pos[1], self._pos[1]
        elif nn == '3':
            pos_t1, pos_t2 = self._pos[0], self._pos[3]
        elif nn == '4':
            pos_t1, pos_t2 = self._pos[0], self._pos[1]
        elif nn == '5':
            pos_t1, pos_t2 = self._pos[4], self._pos[1]
        elif nn == '6':
            pos_t1, pos_t2 = self._pos[0], -self._pos[2]  # the displacements for 6nn_t2 point away from 3nn_l
        else:
            raise ValueError
        
        if tag == '1':
            r, F_ij, ij_direcs = self.get_ij_bond_force(jth_atom_id, jth_atom_forces, pos_t1)
            if nn in ['1', '2', '4', '5']:
                u = ij_direcs*r[:, np.newaxis]@l_hat
                basis = np.array([t1_hat, l_hat, t2_hat])
            elif nn in ['3', '6']:
                u = ij_direcs*r[:, np.newaxis]@t1_hat
                basis = np.array([l_hat, t1_hat, t2_hat])
        elif tag == '2':
            r, F_ij, ij_direcs = self.get_ij_bond_force(jth_atom_id, jth_atom_forces, pos_t2)
            if nn in ['1', '3', '4', '5', '6']:
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
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[0]) 
                   for forces, atom_id in zip(self._force_on_j_list[:2], self._plane_nn[0][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_1nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[2], jth_atom_id=self._plane_nn[0][2], tag='1', nn='1')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_1nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[3], jth_atom_id=self._plane_nn[0][1], tag='2', nn='1') 
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 'l_2nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[1]) 
                   for forces, atom_id in zip(self._force_on_j_list[4:6], self._plane_nn[1][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_2nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[6], jth_atom_id=self._plane_nn[1][2], tag='1', nn='2')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_2nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[7], jth_atom_id=self._plane_nn[1][3], tag='2', nn='2') 
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 'l_3nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[2]) 
                   for forces, atom_id in zip(self._force_on_j_list[8:10], self._plane_nn[2][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_3nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[10], jth_atom_id=self._plane_nn[2][1], tag='1', nn='3')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_3nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[11], jth_atom_id=self._plane_nn[2][1], tag='2', nn='3')
            bonds, force = self._concatenate_t_special(out=out)
            # bonds, force = self._concatenate_t(out=out) 
        elif tag == 'l_4nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[0]) 
                   for forces, atom_id in zip(self._force_on_j_list[12:14], self._plane_nn[3][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_4nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[14], jth_atom_id=self._plane_nn[3][2], tag='1', nn='4')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_4nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[15], jth_atom_id=self._plane_nn[3][1], tag='2', nn='4')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 'l_5nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[4]) 
                   for forces, atom_id in zip(self._force_on_j_list[16:18], self._plane_nn[4][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_5nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[18], jth_atom_id=self._plane_nn[4][2], tag='1', nn='5')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_5nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[19], jth_atom_id=self._plane_nn[4][1], tag='2', nn='5')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 'l_6nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[3]) 
                   for forces, atom_id in zip(self._force_on_j_list[20:22], self._plane_nn[5][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_6nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[22], jth_atom_id=self._plane_nn[5][1], tag='1', nn='6')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_6nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[23], jth_atom_id=self._plane_nn[5][1], tag='2', nn='6')
            bonds, force = self._concatenate_t(out=out) 
        else:
            raise ValueError
        return bonds, force
    
class BccLAPotential(GenerateLAPotential):
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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run_static_analysis(self):
        """
        Run static bond analysis on the reference job and assign hidden variables.
        """
        if self.delete_existing_jobs:
            self.delete_static_job = True
        self._stat_ba = self._project.create_job(StaticBondAnalysis, 'static_ba', delete_existing_job=self.delete_static_job)
        self._stat_ba.input.structure = self.structure.copy()
        self._stat_ba.input.n_shells = self.shells
        self._stat_ba.run()

        self._b0 = np.linalg.norm(self._stat_ba.output.per_shell_irreducible_bond_vectors, axis=-1)
        self._rotations = self._stat_ba.output.per_shell_0K_rotations
        atom_ids = [self._stat_ba.output.per_shell_bond_indexed_neighbor_list[i][self.ith_atom_id]
                    for i in range(self.shells)]

        # 1nn basis, strightforward
        self._basis = [self._stat_ba.output.per_shell_transformation_matrices[0][0]]
        self._nn_atom_ids = (atom_ids[0]).tolist()
        self._nn_bond_vecs = (self._basis[0][0]@self._rotations[0]).tolist()
        self.disp_dir = [self._basis[0][0], self._basis[0][1], self._basis[0][2]]
        
        def update_basis_rotations(s, basis, atom_ids, a, b):
            """
            Update the basis rotations.

            Args:
                s (int): Index of the shell.
                basis (numpy.ndarray): Basis array.
                atom_ids (list): List of atom IDs.
                a (int): Index of the column in the basis array.
                b (float): Value to compare with.

            Returns:
                None
            """
            roll_arg = np.argwhere(np.all(np.isclose(a=(basis@self._rotations[s])[:, a], b=b, atol=1e-10), axis=-1))[-1][-1]
            roll_id = int(len(self._rotations[s]) - roll_arg)
            self._basis.append((basis@self._rotations[s])[roll_arg])
            rolled_vecs = np.roll(basis[0]@self._rotations[s], roll_id, axis=0)
            rolled_basis = np.roll(basis@self._rotations[s], roll_id, axis=0)
            self._rotations[s] = np.array([self._find_rotation_matrix(rolled_basis[0], rb) for rb in rolled_basis])
            self._nn_atom_ids += np.roll(atom_ids[s], roll_id).tolist()
            self._nn_bond_vecs += (rolled_vecs).tolist()

        if self.shells >= 2:
            s = 1
            basis = self._stat_ba.output.per_shell_transformation_matrices[s][0]
            update_basis_rotations(s, basis, atom_ids, a=0, b=np.array([-1., 0., 0.]))
            self.disp_dir.append(self._basis[s][0])

        if self.shells >= 3:
            s = 2
            basis = self._stat_ba.output.per_shell_transformation_matrices[s][0]
            update_basis_rotations(s, basis, atom_ids, a=0, b=self._basis[0][1])

        if self.shells >= 4:
            s = 3
            basis = self._stat_ba.output.per_shell_transformation_matrices[s][0]
            update_basis_rotations(s, basis, atom_ids, a=1, b=self._basis[0][1])
            self.disp_dir.extend([self._basis[s][0], self._basis[s][2]])

        if self.shells >= 5:
            s = 4
            self._basis.append(self._stat_ba.output.per_shell_transformation_matrices[s][0])
            self._nn_atom_ids += (atom_ids[s]).tolist()
            self._nn_bond_vecs += (self._basis[s][0]@self._rotations[s]).tolist()

        if self.shells >= 6:
            s = 5
            basis = self._stat_ba.output.per_shell_transformation_matrices[s][0]
            update_basis_rotations(s, basis, atom_ids, a=0, b=np.array([-1., 0., 0.]))

        if self.shells >= 7:
            s = 6
            basis = self._stat_ba.output.per_shell_transformation_matrices[s][0]
            update_basis_rotations(s, basis, atom_ids, a=1, b=self._basis[0][1])
            self.disp_dir.extend([self._basis[s][0], self._basis[s][2]])

        if self.shells == 8:
            s = 7
            basis = self._stat_ba.output.per_shell_transformation_matrices[s][0]
            update_basis_rotations(s, basis, atom_ids, a=2, b=np.array([-1., 0., 0.]))
            self.disp_dir.append(self._basis[s][0])
            
        self._nn_bond_vecs = np.array(self._nn_bond_vecs)
        self._nn_atom_ids = np.array(self._nn_atom_ids)
        
    def get_plane_neighbors(self, output=False):
        """
        Retrieve atom indices (jth atom) for in-plane and out-of-plane NNs. 
        """
        anti_l_id_1nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:8]@self._basis[0][0], b=-1., atol=1e-10))[-1][-1]
        self._plane_nn = [[self._nn_atom_ids[i] for i in [anti_l_id_1nn, 0]]]
        
        if self.shells >= 2:
            anti_l_id_2nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:14]@self._basis[1][0], b=-1., atol=1e-10))[-1][-1]
            l_id_2nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:14]@self._basis[1][0], b=1., atol=1e-10))[-1][-1]
            t1_id_2nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[:14], b=self._basis[1][1], atol=1e-10), axis=-1))[-1][-1]
            t2_id_2nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[:14], b=self._basis[1][2], atol=1e-10), axis=-1))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i] for i in [anti_l_id_2nn, l_id_2nn, t1_id_2nn, t2_id_2nn]])
        
        if self.shells >= 3:
            anti_l_id_3nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:26]@self._basis[2][0], b=-1., atol=1e-10))[-1][-1]
            l_id_3nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:26]@self._basis[2][0], b=1., atol=1e-10))[-1][-1]
            t1_id_3nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[:26], b=self._basis[2][1], atol=1e-10), axis=-1))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i] for i in [anti_l_id_3nn, l_id_3nn, t1_id_3nn]])

        if self.shells >= 4:
            anti_l_id_4nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:50]@self._basis[3][0], b=-1., atol=1e-10))[-1][-1]
            l_id_4nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[:50]@self._basis[3][0], b=1., atol=1e-10))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i] for i in [anti_l_id_4nn, l_id_4nn]])

        if self.shells >= 5:
            anti_l_id_5nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[50:]@self._basis[4][0], b=-1., atol=1e-10))[-1][-1]
            l_id_5nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[50:]@self._basis[4][0], b=1., atol=1e-10))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i+50] for i in [anti_l_id_5nn, l_id_5nn]])

        if self.shells >= 6:
            anti_l_id_6nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[50:]@self._basis[5][0], b=-1., atol=1e-10))[-1][-1]
            l_id_6nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[50:]@self._basis[5][0], b=1., atol=1e-10))[-1][-1]
            t1_id_6nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[50:], b=self._basis[5][1], atol=1e-10), axis=-1))[-1][-1]
            t2_id_6nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[50:], b=self._basis[5][2], atol=1e-10), axis=-1))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i+50] for i in [anti_l_id_6nn, l_id_6nn, t1_id_6nn, t2_id_6nn]])

        if self.shells >= 7:
            anti_l_id_7nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[50:]@self._basis[6][0], b=-1., atol=1e-10))[-1][-1]
            l_id_7nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[50:]@self._basis[6][0], b=1., atol=1e-10))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i+50] for i in [anti_l_id_7nn, l_id_7nn]])
        
        if self.shells == 8:
            anti_l_id_8nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[50:]@self._basis[7][0], b=-1., atol=1e-10))[-1][-1]
            l_id_8nn = np.argwhere(np.isclose(a=self._nn_bond_vecs[50:]@self._basis[7][0], b=1., atol=1e-10))[-1][-1]
            t1_id_8nn = np.argwhere(np.all(np.isclose(a=self._nn_bond_vecs[50:], b=self._basis[7][1], atol=1e-10), axis=-1))[-1][-1]
            self._plane_nn.append([self._nn_atom_ids[i+50] for i in [anti_l_id_8nn, l_id_8nn, t1_id_8nn]])
        
        if output:
            return self._plane_nn
        
    def _validate_ready_to_run(self):
        if self._stat_ba is None:
            self.run_static_analysis()
        if self._pos is None:
            self._pos = [self.generate_atom_positions(direction=self.disp_dir[i], spacing=self.uneven) for i in range(2)]
            self._pos.append(self.generate_atom_positions(direction=self.disp_dir[2], spacing=self.uneven, asymmetric=True))
            # self._pos.append(self.generate_atom_positions(direction=self.disp_dir[2], spacing=self.uneven))

            if self.shells >= 2:
                self._pos.append(self.generate_atom_positions(direction=self.disp_dir[3], spacing=self.uneven))
                
            if self.shells >= 4:
                self._pos.append(self.generate_atom_positions(direction=self.disp_dir[4], spacing=self.uneven))
                self._pos.append(self.generate_atom_positions(direction=self.disp_dir[5], spacing=self.uneven, asymmetric=True))
                # self._pos.append(self.generate_atom_positions(direction=self.disp_dir[5], spacing=self.uneven))

            if self.shells >= 7:
                self._pos.append(self.generate_atom_positions(direction=self.disp_dir[6], spacing=self.uneven))
                self._pos.append(self.generate_atom_positions(direction=self.disp_dir[7], spacing=self.uneven, asymmetric=True))
                # self._pos.append(self.generate_atom_positions(direction=self.disp_dir[7], spacing=self.uneven))

            if self.shells == 8:
                self._pos.append(self.generate_atom_positions(direction=self.disp_dir[8], spacing=self.uneven))
        
        # set the tags to correspond to the directions along which static calcuations are actually performed
        self._tags = np.arange(len(self._pos)).astype(str).tolist()
    
    def get_force_on_j(self, output=False):
        """
        Retrieve forces on the jth atoms at each dislacement.
        """
        if self._force_on_j_list is None:
            self.load_disp_jobs()
        if self._plane_nn is None:
            self.get_plane_neighbors()  
        
        self._force_on_j_list = [self._force_on_j(i, tag='0') for i in self._plane_nn[0]]
        self._force_on_j_list.append(self._force_on_j(self._plane_nn[0][1], tag='1'))
        self._force_on_j_list.append(self._force_on_j(self._plane_nn[0][1], tag='2'))
        
        if self.shells >= 2:
            s = 1
            for i in self._plane_nn[s]:
                self._force_on_j_list.append(self._force_on_j(i, tag='3'))
                
        if self.shells >= 3:
            s = 2
            for i in self._plane_nn[s]:
                self._force_on_j_list.append(self._force_on_j(i, tag='1'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[s][1], tag='3'))

        if self.shells >= 4:
            s = 3
            for i in self._plane_nn[s]:
                self._force_on_j_list.append(self._force_on_j(i, tag='4'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[s][1], tag='1'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[s][1], tag='5'))

        if self.shells >= 5:
            s = 4
            for i in self._plane_nn[s]:
                self._force_on_j_list.append(self._force_on_j(i, tag='0'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[s][1], tag='1'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[s][1], tag='2'))

        if self.shells >= 6:
            s = 5
            for i in self._plane_nn[s]:
                self._force_on_j_list.append(self._force_on_j(i, tag='3'))

        if self.shells >= 7:
            s = 6
            for i in self._plane_nn[s]:
                self._force_on_j_list.append(self._force_on_j(i, tag='6'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[s][1], tag='1'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[s][1], tag='7'))

        if self.shells == 8:
            s = 7
            for i in self._plane_nn[s]:
                self._force_on_j_list.append(self._force_on_j(i, tag='8'))
            self._force_on_j_list.append(self._force_on_j(self._plane_nn[s][1], tag='3'))
        
        if output:
            return self._force_on_j_list
    
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
        l_hat, t1_hat, t2_hat = self._basis[int(nn)-1]
        if nn == '1':
            pos_t1, pos_t2 = self._pos[1], self._pos[2]
        elif nn == '2':
            pos_t1, pos_t2 = self._pos[3], self._pos[3]
        elif nn == '3':
            pos_t1, pos_t2 = self._pos[1], self._pos[3]
        elif nn == '4':
            pos_t1, pos_t2 = self._pos[1], self._pos[5]
        elif nn == '5':
            pos_t1, pos_t2 = self._pos[1], self._pos[2]
        elif nn == '6':
            pos_t1, pos_t2 = self._pos[3], self._pos[3]
        elif nn == '7':
            pos_t1, pos_t2 = self._pos[1], self._pos[7]
        elif nn == '8':
            pos_t1, pos_t2 = self._pos[8], self._pos[3]
        else:
            raise ValueError
        
        if tag == '1':
            r, F_ij, ij_direcs = self.get_ij_bond_force(jth_atom_id, jth_atom_forces, pos_t1)
            if nn in ['2', '3', '6', '8']:
                u = ij_direcs*r[:, np.newaxis]@l_hat
                basis = np.array([t1_hat, l_hat, t2_hat])
            elif nn in ['1', '4', '5', '7']:
                u = ij_direcs*r[:, np.newaxis]@t1_hat
                basis = np.array([l_hat, t1_hat, t2_hat])
        elif tag == '2':
            r, F_ij, ij_direcs = self.get_ij_bond_force(jth_atom_id, jth_atom_forces, pos_t2)
            if nn in ['1', '3', '4', '5', '7', '8']:
                u = ij_direcs*r[:, np.newaxis]@t2_hat
                basis = np.array([l_hat, t2_hat, t1_hat])
            elif nn in ['2', '6']:
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
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[0]) 
                   for forces, atom_id in zip(self._force_on_j_list[:2], self._plane_nn[0][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_1nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[2], jth_atom_id=self._plane_nn[0][1], tag='1', nn='1')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_1nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[3], jth_atom_id=self._plane_nn[0][1], tag='2', nn='1') 
            bonds, force = self._concatenate_t_special(out=out)
            # bonds, force = self._concatenate_t(out=out) 
        elif tag == 'l_2nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[3]) 
                   for forces, atom_id in zip(self._force_on_j_list[4:6], self._plane_nn[1][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_2nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[6], jth_atom_id=self._plane_nn[1][2], tag='1', nn='2')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_2nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[7], jth_atom_id=self._plane_nn[1][3], tag='2', nn='2') 
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 'l_3nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[1]) 
                   for forces, atom_id in zip(self._force_on_j_list[8:10], self._plane_nn[2][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_3nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[10], jth_atom_id=self._plane_nn[2][2], tag='1', nn='3')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_3nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[11], jth_atom_id=self._plane_nn[2][1], tag='2', nn='3') 
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 'l_4nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[4]) 
                   for forces, atom_id in zip(self._force_on_j_list[12:14], self._plane_nn[3][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_4nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[14], jth_atom_id=self._plane_nn[3][1], tag='1', nn='4')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_4nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[15], jth_atom_id=self._plane_nn[3][1], tag='2', nn='4')
            bonds, force = self._concatenate_t_special(out=out)
            # bonds, force = self._concatenate_t(out=out) 
        elif tag == 'l_5nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[0]) 
                   for forces, atom_id in zip(self._force_on_j_list[16:18], self._plane_nn[4][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_5nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[18], jth_atom_id=self._plane_nn[4][1], tag='1', nn='5')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_5nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[19], jth_atom_id=self._plane_nn[4][1], tag='2', nn='5')
            bonds, force = self._concatenate_t_special(out=out) 
            # bonds, force = self._concatenate_t(out=out)
        elif tag == 'l_6nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[3]) 
                   for forces, atom_id in zip(self._force_on_j_list[20:22], self._plane_nn[5][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_6nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[22], jth_atom_id=self._plane_nn[5][2], tag='1', nn='6')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_6nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[23], jth_atom_id=self._plane_nn[5][3], tag='2', nn='6') 
            bonds, force = self._concatenate_t(out=out)
        elif tag == 'l_7nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[6]) 
                   for forces, atom_id in zip(self._force_on_j_list[24:26], self._plane_nn[6][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_7nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[26], jth_atom_id=self._plane_nn[6][1], tag='1', nn='7')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_7nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[27], jth_atom_id=self._plane_nn[6][1], tag='2', nn='7') 
            bonds, force = self._concatenate_t_special(out=out) 
            # bonds, force = self._concatenate_t(out=out)
        elif tag == 'l_8nn':
            out = [self.get_ij_bond_force(jth_atom_forces=forces, jth_atom_id=atom_id, ith_atom_positions=self._pos[8]) 
                   for forces, atom_id in zip(self._force_on_j_list[28:30], self._plane_nn[7][:2])]
            bonds, force = self._concatenate_l(out=out) 
        elif tag == 't1_8nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[30], jth_atom_id=self._plane_nn[7][2], tag='1', nn='8')
            bonds, force = self._concatenate_t(out=out) 
        elif tag == 't2_8nn':
            out = self.get_t_bond_force(jth_atom_forces=self._force_on_j_list[31], jth_atom_id=self._plane_nn[7][1], tag='2', nn='8') 
            bonds, force = self._concatenate_t(out=out)
        else:
            raise ValueError
        return bonds, force