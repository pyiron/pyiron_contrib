# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import pint
unit = pint.UnitRegistry()

from pyiron_base.jobs.master.generic import GenericMaster
from pyiron_base.storage.datacontainer import DataContainer

from scipy.integrate import simpson
import scipy.constants
KB = scipy.constants.physical_constants['Boltzmann constant in eV/K'][0]
HBAR = scipy.constants.physical_constants['reduced Planck constant in eV s'][0]
H = scipy.constants.physical_constants['Planck constant in eV/Hz'][0]
AMU = scipy.constants.physical_constants['atomic mass constant'][0]

class _TRInput(DataContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ref_job = None
        self._temperatures = None
        self._mesh = None
        self._ldos = False

    @property
    def ref_job(self):
        return self._ref_job

    @ref_job.setter
    def ref_job(self, job):
        self._ref_job = job
        
    @property
    def temperatures(self):
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temps):
        self._temperatures = temps

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh

    @property
    def ldos(self):
        return self._ldos

    @ldos.setter
    def ldos(self, ldos):
        self._ldos = ldos
        
class TemperatureRemapping(GenericMaster):
    def __init__(self, project, job_name):
        super(TemperatureRemapping, self).__init__(project, job_name)
        self._python_only_job = False
        self.input = _TRInput(table_name="job_input")
        self.output = DataContainer(table_name="job_output")
        self.harm_job = None
        
    @property
    def structure(self):
        return self.input.ref_job.structure

    def run_ha(self):
        if self.harm_job is None:
            self.input.ref_job.status.initialized = True
            self.harm_job = self.child_project.create.job.PhonopyJob(self.job_name+'_phonopy_job')
            self.harm_job.ref_job = self.input.ref_job
            self.harm_job.input['dos_mesh'] = self.input.mesh
            self.harm_job.input['eigenvectors'] = self.input.ldos
            self.harm_job.input['tetrahedron_method'] = False
            self.harm_job.run()
            self.harm_job.master_id = self.id

    def reshape_hessian(self, hessian):
        n_atoms = self.structure.get_number_of_atoms()
        reshaped_hessian = hessian.transpose(0, 2, 1, 3)
        reshaped_hessian = reshaped_hessian.reshape(n_atoms*3, n_atoms*3)
        return reshaped_hessian

    def get_gamma_frequencies(self):
        hessian = self.reshape_hessian(self.output.hessian)
        masses = self.structure.get_masses()
        m = np.tile(masses, (3, 1)).T.flatten()
        mass_tensor = np.sqrt(m*m[:, np.newaxis])
        eigvals, eigvecs = np.linalg.eigh(hessian/mass_tensor)
        nu_square = (eigvals * unit.electron_volt/unit.angstrom**2/unit.amu).to("THz**2").magnitude
        nus = np.sign(nu_square)*np.sqrt(np.absolute(nu_square))/(2*np.pi)
        return nus

    @staticmethod
    def get_sigma(nus):
        return (nus.max()-nus.min())/100.0

    def collect_phonon_output(self):
        self.output.dos_nu = self.harm_job['output/dos_energies']
        self.output.dos_total = self.harm_job['output/dos_total']
        self.output.hessian = self.harm_job['output/force_constants']
        self.output.gamma_frequencies = self.get_gamma_frequencies()
        self.output.sigma = self.get_sigma(self.output.gamma_frequencies)
        
    @staticmethod
    def smearing_method(x0, x, sigma=0.025):
        return np.exp(-(x0-x)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
        
    @staticmethod
    def get_eigvecs2(nus, eigvecs):
        n_atoms = nus.shape[1]//3
        i_x = np.arange(n_atoms, dtype="int")*3
        i_y = np.arange(n_atoms, dtype="int")*3+1
        i_z = np.arange(n_atoms, dtype="int")*3+2
        
        eigvecs2 = np.abs(eigvecs[:, i_x, :])**2
        eigvecs2 += np.abs(eigvecs[:, i_y, :])**2
        eigvecs2 += np.abs(eigvecs[:, i_z, :])**2
        return eigvecs2

    def get_per_atom_dos(self):
        mesh_dict = self.harm_job.phonopy.get_mesh_dict()
        weights = mesh_dict['weights']
        nus = mesh_dict['frequencies']
        eigvecs = mesh_dict['eigenvectors']
        eigvecs2 = self.get_eigvecs2(nus, eigvecs)
        
        num_pdos = eigvecs2.shape[1]
        num_nus = len(self.output.dos_nu)
        projected_dos = np.zeros((num_pdos, num_nus), dtype="double")
        weights = weights/float(np.sum(weights))
        for i, nu in enumerate(self.output.dos_nu):
            amplitudes = self.smearing_method(x0=nus, x=nu, sigma=self.output.sigma)
            for j in range(projected_dos.shape[0]):
                projected_dos[j, i] = np.dot(
                    weights, eigvecs2[:, j, :]*amplitudes
                ).sum()
        self.output.per_atom_dos_total = projected_dos

    @staticmethod
    def remap_function(nu, dos, temperature):
        return simpson(y=H*dos*nu*(0.5+(1/(np.exp(H*nu/(KB*temperature))-1))), x=nu)/KB
    
    def remap_temperatures(self, nu, dos):
        sel = nu > 0.
        nu_sel  = nu[sel]*1e12  # from THz to Hz
        dos_sel = dos[sel]
        dos_sel /= simpson(y=dos_sel, x=nu_sel)
        remapped_temperatures = np.array([self.remap_function(nu=nu_sel, dos=dos_sel, temperature=temp) for temp in self.input.temperatures])
        return remapped_temperatures

    def get_remapped_temperatures(self):
        self.output.system_remapped_temperatures = self.remap_temperatures(nu=self.output.dos_nu, dos=self.output.dos_total)
        if self.input.ldos:
            self.output.per_atom_remapped_temperatures = []
            for dos in self.output.per_atom_dos_total:
                self.output.per_atom_remapped_temperatures.append(self.remap_temperatures(nu=self.output.dos_nu, dos=dos))
            self.output.per_atom_remapped_temperatures = np.array(self.output.per_atom_remapped_temperatures)

    def get_per_atom_remapped_masses(self, temperatures):
        assert self.output.per_atom_remapped_temperatures.any(), "No per-atom remapped temperatures found!"
        assert len(self.output.per_atom_remapped_temperatures[0]) == len(temperatures), "Temperatures are not the same length!"
        masses = self.structure.get_masses().copy()
        # per_atom_mass_map = np.array([masses[i]*temperatures**2/self.output.per_atom_remapped_temperatures[i]**2 for i in range(len(masses))])
        per_atom_mass_map = np.array([masses[i]*self.output.per_atom_remapped_temperatures[i]/temperatures for i in range(len(masses))])
        return per_atom_mass_map

    def validate_ready_to_run(self):
        if not self.input.mesh:
            self.input.mesh = 10
        if not self.input.ldos:
            self.input.ldos = False

    def run_static(self):
        self.status.running = True
        self.run_ha()
        self.status.finished = True
        self.collect_phonon_output()
        if self.input.ldos:
            self.get_per_atom_dos()
        self.get_remapped_temperatures()
        self.to_hdf()
    
    def to_hdf(self, hdf=None, group_name=None):
        super(TemperatureRemapping, self).to_hdf()
        self.input.to_hdf(self.project_hdf5)
        self.output.to_hdf(self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super(TemperatureRemapping, self).from_hdf()
        self.input.from_hdf(self.project_hdf5)
        self.output.from_hdf(self.project_hdf5)
        if self.status.finished:
            self.harm_job = self.project.inspect(self.job_name+'_phonopy_job')