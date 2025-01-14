# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import numpy as np
from phonopy.structure.grid_points import get_qpoints

class GenerateHessian():
    """
    Class to generate the Hessian blocks for a reference atom in a bulk fcc or bcc crystal.
    Used to generate covariance matrices alpha_{rpl} for the mean-field model.

    Args:
        project (str): Name of the pyiron project for storing the jobs.
        ref_job (pyiron Job): Reference job containing an appropriate structure and an interatomic potential.
        potential (potential function): A valid interatomic potential.
        ref_atom (int): The id of the atom which will be displaced. Defaults to 0.
        delta_x (float): Displacement of the reference atom from its equilibrium position. Defaults to 0.01 Angstroms.
        qpoints (float): Qpoints along each direction (the mesh will be qpoints x qpoints x qpoints). Defaults to 10.
        cutoff_radius (float): The cutoff radius upto which the nearest neighbors are considered. Defaults to None, consider all atoms as neighbors.
    """
    def __init__(self, project, ref_job, potential, structure=None, ref_atom=0, delta_x=0.01, qpoints=10, cutoff_radius=None):
        self.project = project
        self.ref_job = ref_job
        self.potential = potential
        self.structure = structure
        self.ref_atom = ref_atom
        self.delta_x = delta_x
        self.qpoints = qpoints
        self.cutoff_radius = cutoff_radius

        self._jobs = None
        if self.structure is None:
            self.structure = self.ref_job.structure.copy()

    def _run_job(self, job_name, i, j):
        """
        Helper class to run the displacement job.
        """
        job = self.ref_job.copy_template(project=self.project, new_job_name=job_name)
        if self.potential is not None:
            job.potential = self.potential
        dx = np.zeros_like(job.structure.positions)
        if i%2 == 0:
            dx[self.ref_atom][j] = self.delta_x
        else:
            dx[self.ref_atom][j] = -self.delta_x
        job.structure.positions += dx
        job.calc_static()
        job.run()

    def run_jobs(self, delete_existing_jobs=False):
        """
        Run the displacement jobs.

        Parameters:
            delete_existing_jobs (boolean): Delete the existing displacement jobs.
        """
        job_list = self.project.job_table().job.to_list()
        for i in range(6):
            # make displacements for the ref atom along x, y, z in the + and - directions
            # arragned as [x+, x-, y+, y-, z+, z-]
            job_name = 'disp_job_' + str(i)
            if job_name not in job_list:
                    self._run_job(job_name=job_name, i=i, j=int((i-(i%2))/2))
            else:
                job = self.project.inspect(job_name)
                if job.status in ['aborted'] or delete_existing_jobs:
                    self.project.remove_job(job_name)
                    self._run_job(job_name=job_name, i=i, j=int((i-(i%2))/2))

    def load_jobs(self):
        """
        Load the displacement jobs.
        """
        self._jobs = [self.project.inspect('disp_job_' + str(i)) for i in range(6)]

    def get_forces(self):
        """
        Get the forces on all the atoms from each displacement job.

        Returns:
            forces (np.ndarray): 6 x forces on all the atoms.
        """
        self.load_jobs()
        return np.array([job['output/generic/forces'][-1] for job in self._jobs])
    
    def get_hessian_direct(self):
        """
        Get the Hessian blocks for the reference atom in direct space.

        Returns:
            hessian (np.ndarray): n_atoms x 3 x 3 array of hessians.
        """

        forces = self.get_forces()
        forces_xyz = np.array([(forces[0]-forces[1]).T, 
                               (forces[2]-forces[3]).T, 
                               (forces[4]-forces[5]).T])/2.
        hessian_direct = (-forces_xyz/self.delta_x).transpose(2, 0, 1)  # shape (n_atoms, 3, 3)
        return hessian_direct
    
    def get_qpoint_vectors(self):
        """
        Get qpoint vectors and their weights (without rotational symmetry).

        Returns:
            qpoint_vectors (np.ndarray): Reduced qpoints**3 x 3 vectors.
            qpoint_weights (np.array): Weights of the reduced qpoint vectors.
        """
        structure = self.structure.copy()
        sym = structure.get_symmetry()
        reciprocal_cell = sym.get_primitive_cell().cell.reciprocal()
        # n = np.arange(self.qpoints**3)
        # ix = np.array([n//self.qpoints**2, (n//self.qpoints)%self.qpoints, n%self.qpoints]).T.astype(float)
        # qpoint_vectors = (2.0*np.pi/float(self.qpoints))*(ix+0.5)@reciprocal_cell
        # qpoint_vectors -= qpoint_vectors.mean(0)
        # qpoint_weights = np.ones(len(qpoint_vectors))
        mesh = [self.qpoints, self.qpoints, self.qpoints]
        grid, qpoint_weights = get_qpoints(mesh_numbers=mesh, reciprocal_lattice=reciprocal_cell)
        qpoint_vectors = (2.0*np.pi*grid)@reciprocal_cell
        return qpoint_vectors, qpoint_weights
    
    def get_hessian_reciprocal(self, structure=None, hessian_direct=None, rewrite=False):
        """
        Get the Hessian blocks for the reference atom in reciprocal space, along with the qpoint vectors and their weights.

        Parameters:
            rewrite (boolean): Whether or not to rewrite the reciprocal hessians, qpoint vectors, and qpoint weights. Defaults to False.

        Returns:
            hessian_reciprocal (np.ndarray): Reduced qpoints**3 x 3 x 3 Hessians.
            qpoint_vectors (np.ndarray): Reduced qpoints**3 x 3 vectors.
            qpoint_weights (np.array): Weights of the reduced qpoint vectors.
        """
        if not os.path.exists(os.path.join(self.project.path, 'resources')):
            os.mkdir(os.path.join(self.project.path, 'resources'))
        exists = [os.path.exists(os.path.join(self.project.path, 'resources', file)) for file in ['hessian_reciprocal.npy', 'qpoint_vectors.npy', 'qpoint_weights.npy']]
        if np.all(exists) and (not rewrite):
            return (np.load(os.path.join(self.project.path, 'resources', 'hessian_reciprocal.npy')),
                    np.load(os.path.join(self.project.path, 'resources', 'qpoint_vectors.npy')),
                    np.load(os.path.join(self.project.path, 'resources', 'qpoint_weights.npy'))
                    )
        else:
            if hessian_direct is None:
                hessian_direct = self.get_hessian_direct()
            qpoint_vectors, qpoint_weights = self.get_qpoint_vectors()
            structure = self.structure.copy()
            X = structure.positions.copy()
            if self.cutoff_radius is not None:
                select = structure.get_neighborhood(positions=X[self.ref_atom], cutoff_radius=self.cutoff_radius, num_neighbors=None).indices
                if len(select)>structure.get_number_of_atoms():
                    select = np.ones(structure.get_number_of_atoms(), dtype=bool)
            else:
                sq_trace = np.einsum('ijk,ikj->i', hessian_direct, hessian_direct)
                select = sq_trace > 1e-3
            dX = structure.find_mic(X-X[self.ref_atom])
            q_dX = qpoint_vectors@dX[select].T
            hessian_reciprocal = np.einsum('il,ijk->ljk', np.exp(1j*q_dX.T), hessian_direct[select])
             
            np.save(os.path.join(self.project.path, 'resources', 'hessian_reciprocal.npy'), hessian_reciprocal)
            np.save(os.path.join(self.project.path, 'resources', 'qpoint_vectors.npy'), qpoint_vectors)
            np.save(os.path.join(self.project.path, 'resources', 'qpoint_weights.npy'), qpoint_weights)
            return hessian_reciprocal, qpoint_vectors, qpoint_weights
