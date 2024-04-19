# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import pint
unit = pint.UnitRegistry()

from pyiron_base import Project
from pyiron_base.jobs.job.generic import GenericJob
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
        self._interaction_range = 10

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
    def interaction_range(self):
        return self._interaction_range

    @interaction_range.setter
    def interaction_range(self, i_range):
        self._interaction_range = i_range
        
class TemperatureRemapping(GenericJob):
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
            sub_pr = Project(self.working_directory)
            self.harm_job = sub_pr.create.job.PhonopyJob('ha_job', delete_existing_job=True)
            self.harm_job.ref_job = self.input.ref_job
            self.harm_job.run()
        return self.harm_job
    
    def get_nu_data(self):
        self.output.dos_nu = self.harm_job['output/dos_energies']
        self.output.dos_total = self.harm_job['output/dos_total']
        
        sel = self.output.dos_nu > 0.
        nu  = self.output.dos_nu[sel]
        dos = self.output.dos_total[sel]
        dos /= simpson(y=dos, x=nu)
        self.output.effective_nu = simpson(y=dos*nu, x=nu)  # in Thz
        
        hessian = self.get_hessian()
        nu_v = self.get_vibrational_frequencies(hessian, return_eigenvectors=False)
        self.output.frequencies_at_gamma = nu_v
        
    @staticmethod
    def reshape_hessian(hessian):
        n_atoms = hessian.shape[0]
        reshaped_hessian = hessian.transpose(0, 2, 1, 3)
        reshaped_hessian = reshaped_hessian.reshape(n_atoms*3, n_atoms*3)
        return reshaped_hessian

    def get_hessian(self, reshape=True):
        hessian = self.harm_job['output/force_constants']
        if reshape:
            hessian = self.reshape_hessian(hessian)
        return hessian

    def get_vibrational_frequencies(self, hessian, masses=None, return_eigenvectors=True):
        n_atoms = hessian.shape[0] // 3
        if masses is None:
            mass = self.structure.get_masses()[0]
            masses = np.array([mass]*n_atoms)
        m = np.tile(masses, (3, 1)).T.flatten()
        mass_tensor = np.sqrt(m * m[:, np.newaxis])
        eigvals, eigvecs = np.linalg.eigh(hessian / mass_tensor)
        nu_square = (eigvals * unit.electron_volt / unit.angstrom**2 / unit.amu).to("THz**2").magnitude
        nu = np.sign(nu_square) * np.sqrt(np.absolute(nu_square)) / (2 * np.pi)
        if return_eigenvectors:
            return nu, eigvecs
        return nu
    
    @staticmethod
    def delta_function(x, x0, sigma=0.025):
        return np.exp(-(x - x0)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

    @staticmethod
    def rho_i_alpha(nu, eigvectors, nu_v):
        return np.sum((eigvectors**2) * delta_function(nu, nu_v), axis=-1)
    
    def get_remapped_temperatures(self):
        sel = self.output.dos_nu > 0.
        nu  = self.output.dos_nu[sel]*1e12  # from THz to Hz
        dos = self.output.dos_total[sel]
        dos /= simpson(y=dos, x=nu)
        self.output.remapped_temperatures = np.array([simpson(y=H*dos*nu*(0.5+(1/(np.exp(H*nu/(KB*temp))-1))), x=nu)/KB
                                                      for temp in self.input.temperatures])

    def run_static(self):
        self.status.running = True
        self.run_ha()
        self.status.finished = True
        self.get_nu_data()
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
        self.harm_job = self.project.inspect('ha_job')