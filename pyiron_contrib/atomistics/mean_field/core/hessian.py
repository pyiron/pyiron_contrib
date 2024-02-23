# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np

### TODO: Update documentation!

class GenerateHessian():
    """
    """
    def __init__(self, project, ref_job, potential=None, ref_atom=0, delta_x=0.01, kpoints=10, threshold=0.02):
        self.project = project
        self.ref_job = ref_job
        self.potential = potential
        self.ref_atom = ref_atom
        self.delta_x = delta_x
        self.kpoints = kpoints
        self.threshold = threshold

        self._jobs = None

    def _run_job(self, job_name, i, j):
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
        job_list = self.project.job_table().job.to_list()
        for i in range(6):
            # make displacements for the ref atom along x, y, z in the + and - directions
            # arragned as [x+, x-, y+, y-, z+, z-]
            job_name = 'hessian_job_' + str(i)
            if job_name not in job_list:
                    self._run_job(job_name=job_name, i=i, j=int((i-(i%2))/2))
            else:
                job = self.project.inspect(job_name)
                if job.status in ['aborted'] or delete_existing_jobs:
                    self.project.remove_job(job_name)
                    self._run_job(job_name=job_name, i=i, j=int((i-(i%2))/2))

    def load_jobs(self):
        self._jobs = [self.project.inspect('hessian_job_' + str(i)) for i in range(6)]

    def get_forces(self):
        self.load_jobs()
        return np.array([job['output/generic/forces'][-1] for job in self._jobs])
    
    def get_hessian_real(self):
        forces = self.get_forces()
        forces_xyz = np.array([(forces[0]-forces[1]).T, 
                               (forces[2]-forces[3]).T, 
                               (forces[4]-forces[5]).T])/2.
        hessian = -forces_xyz/self.delta_x
        return hessian.transpose(2, 0, 1)  # shape (n_atoms, 3, 3)
    
    def get_kpoint_vectors(self):
        structure = self.ref_job.structure.copy()
        box = structure.get_symmetry().info['std_lattice']
        primitive_cell = structure.get_symmetry().get_primitive_cell(standardize=False).cell.array
        primitive_cell /= box[0][0]
        brilluoin_zone = np.linalg.inv(primitive_cell)
        n = np.arange(self.kpoints**3)
        ix = np.array([n//self.kpoints**2, (n//self.kpoints)%self.kpoints, n%self.kpoints]).T.astype(float)
        kpoint_vectors = (2.0*np.pi/float(self.kpoints))*(ix+0.5)@brilluoin_zone
        kpoint_vectors -= kpoint_vectors.mean(0)
        return kpoint_vectors
    
    def get_hessian_k(self):
        hessian_real = self.get_hessian_real()
        kpoint_vectors = self.get_kpoint_vectors()
        structure = self.ref_job.structure.copy()
        n_atoms = structure.get_number_of_atoms()

        sq_trace = np.einsum('ijk,ikj->i',hessian_real,hessian_real)
        select = sq_trace > self.threshold * sq_trace.max()

        X = structure.positions
        dX = structure.find_mic(X-X[self.ref_atom])
        hessian_k = []
        for k in kpoint_vectors:
            H_k = np.zeros((3,3),complex)
            for i in np.arange(n_atoms)[select]:
                H_k += hessian_real[i]*np.exp(complex(0,1)*k@dX[i])
            hessian_k.append(H_k)
        return np.array(hessian_k), np.array(kpoint_vectors)
