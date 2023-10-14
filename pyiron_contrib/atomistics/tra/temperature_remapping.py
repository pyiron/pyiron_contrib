# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import pint
unit = pint.UnitRegistry()

from pyiron_base import Project
from pyiron_base.jobs.job.generic import GenericJob
from pyiron_base.storage.datacontainer import DataContainer

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
        
class TemperatureRemapping(GenericJob):
    def __init__(self, project, job_name):
        super(TemperatureRemapping, self).__init__(project, job_name)
        self._python_only_job = True
        self.input = _TRInput(table_name="job_input")
        self.output = DataContainer(table_name="job_output")
        self.harm_job = None
        self.output.remapped_temperatures = None
        self.output.frequencies_at_gamma = None
        
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
        if masses is None:
            masses = self.structure.get_masses()
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
        if self.output.remapped_temperatures is None:
            hessian = self.get_hessian()
            nu_v = self.get_vibrational_frequencies(hessian, return_eigenvectors=False)
            self.output.frequencies_at_gamma = nu_v
            nu = nu_v[3:][nu_v[3:]>0.]
            self.output.remapped_temperatures = np.array([np.sum(H*nu*1e12*(0.5+(1/(np.exp(H*nu*1e12/(KB*temp))-1))))/KB/len(nu)
                                                         for temp in self.input.temperatures])
    
    def run_static(self):
        self.status.running = True
        self.run_ha()
        self.status.finished = True
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
        #self.harm_job = self.project.load('ha_job')
        