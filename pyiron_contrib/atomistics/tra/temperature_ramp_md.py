# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import cumtrapz
from scipy.constants import physical_constants
KB = physical_constants['Boltzmann constant in eV/K'][0]
bar_to_GPa = 1e-4

from jinja2 import Template
from io import StringIO
import os

lammps_input = """\
# This script runs a linear temperature ramp.

#---------------------------- Atomic setup ------------------------------------#

units            metal
dimension        3
boundary         p p p
atom_style       atomic

# Create atoms.
read_data        structure.inp

# Define interatomic potential.
include          potential.inp
#------------------------------------------------------------------------------#

#--------------------------- Simulation variables -----------------------------#

# Initalizes the random number generator.
variable         rnd equal round(random(0,99999,{{ seed }}))

# Simulation control parameters.
variable         t_eq equal round({{ n_equib }}) # Equilibration steps.
variable         t equal round({{ n_steps }}) # Switching steps.
variable         ts equal {{ time_step }}/1000 # Timestep
variable         t_print equal {{ n_print}} # dump frequency
variable         t_traj equal {{ n_traj }} # dump trajectory frequency
#------------------------------------------------------------------------------#

#----------------------------- Run simulation ---------------------------------#

# Integrator and thermostat.
timestep         ${ts}
{% if pressure is not none %}
fix              f1 all nph iso {{ pressure }} {{ pressure }} 1.0
{% else %}
fix              f1 all nve
{% endif %}
fix              f2 all langevin {{ T_start }} {{ T_start }} 0.1 ${rnd} zero yes

# Initial temperature to accelerate equilibration.
variable         T_0 equal 2.0*{{ T_start }}
velocity         all create ${T_0} ${rnd} dist gaussian

# Dummy thermo out, so pyiron does not complain.
thermo_style     custom temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo           0

# Run equilibraiton at T_start.
run              ${t_eq}

# Dump trajectory.
dump             1 all custom ${t_traj} dump.out id type xsu ysu zsu fx fy fz vx vy vz
dump_modify      1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"

# Dump properties just once, so that pyiron does not complain.
thermo_style     custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo           0

# Refix and run forward ramping.
unfix            f1
unfix            f2
{% if pressure is not none %}
fix              f1 all nph iso {{ pressure }} {{ pressure }} 1.0
{% else %}
fix              f1 all nve
{% endif %}
fix              f2 all langevin {{ T_start }} {{ T_stop }} 0.1 ${rnd} zero yes
fix              f3 all print ${t_print} "$(step) $(temp) $(pe) $(ke) $(enthalpy) $(vol) $(press)" title "# Step Temp[K] PE[eV] KE[eV] H[eV] Vol[Ang^3] P[eV/Ang^2]" screen no file forward.dat
run              ${t}
"""

class TemperatureRampMD():
    def __init__(self, ref_job, temperatures, pressure, n_samples=5, n_ramp_steps=1000, 
                 n_equib_steps=100, n_print=1, n_traj_print=1, time_step=1., recompress=False):
        self.ref_job = ref_job
        self.temperatures = temperatures
        self.pressure = pressure
        self.n_samples = n_samples
        self.n_ramp_steps = n_ramp_steps
        self.n_equib_steps = n_equib_steps
        self.n_print = n_print
        self.n_traj_print = n_traj_print
        self.time_step = time_step
        self.recompress = recompress
        
        self._project = self.ref_job.project
        self._job_names = ['sample_' + str(i) for i in range(self.n_samples)]
        self._n_atoms = self.ref_job.structure.get_number_of_atoms()
        self._raw_temperatures = np.linspace(self.temperatures[0], self.temperatures[-1], int(self.n_ramp_steps/self.n_print)+1)
        if self.pressure is not None:
            self.pressure *= 1/bar_to_GPa
    
    def _generate_input_script(self):
        template = Template(lammps_input)
        input_script = template.render(
            seed=np.random.randint(99999), 
            T_start=self.temperatures[0], 
            T_stop=self.temperatures[-1],
            n_equib=self.n_equib_steps,
            n_steps=self.n_ramp_steps,
            n_print=self.n_print,
            n_traj=self.n_traj_print,
            time_step=self.time_step,
            pressure=self.pressure
        )
        return input_script
        
    def _run_job(self, project, job_name):
        job = self.ref_job.copy_template(project=project, new_job_name=job_name)
        job.input.control.load_string(self._generate_input_script())
        job.run()
        
    def run_ramp_md(self, delete_existing_jobs=False):
        job_list = self._project.job_table().job.to_list()
        job_status = self._project.job_table().status.to_list()
        for i, job_name in enumerate(self._job_names): 
            if job_name not in job_list:
                self._run_job(project=self._project, job_name=job_name)
            elif job_status[i] != 'finished' or delete_existing_jobs:
                self._project.remove_job(job_name)
                self._run_job(project=self._project, job_name=job_name)
                
    def _collect_output(self, job_name):
        job = self._project.inspect(job_name)
        job.decompress()
        nest = {'steps': None, 'temp': None, 'en_pot': None, 'en_kin': None, 'H': None, 'vol': None, 'press': None}
        data_dict= {'forward': nest.copy()}
        forward = np.loadtxt(os.path.join(job.working_directory, "forward.dat"), unpack=True)
        data_dict['forward'].update(zip(data_dict['forward'], forward))
        data_dict['forward']['U'] = data_dict['forward']['en_pot']+data_dict['forward']['en_kin']
        if self.recompress:
            job.compress()
        return data_dict
    
    @staticmethod
    def _get_values(data_dict):
        return [key for key in data_dict['forward'].values()]
    
    def _get_raw_properties(self):
        output_dicts = [self._collect_output(job_name) for job_name in self._job_names]
        T_raw, U_raw, H_raw, V_raw, P_raw = zip(*[(val[1], val[7]/self._n_atoms, val[4]/self._n_atoms, val[5]/self._n_atoms,
                                                   val[6]*bar_to_GPa) for val in [self._get_values(data_dict) 
                                                                                  for data_dict in output_dicts]])
        return {'T_raw': T_raw,
                'U_raw': U_raw,
                'H_raw': H_raw,
                'V_raw': V_raw,
                'P_raw': P_raw
               }
    
    def _get_mean_properties(self):
        raw_dict = self._get_raw_properties()
        return {'T_mean': np.mean(raw_dict['T_raw'], axis=0),
                'U_mean': np.mean(raw_dict['U_raw'], axis=0),
                'H_mean': np.mean(raw_dict['H_raw'], axis=0),
                'V_mean': np.mean(raw_dict['V_raw'], axis=0),
                'P_mean': np.mean(raw_dict['P_raw'], axis=0),
                'T_se': np.std(raw_dict['T_raw'], axis=0)/np.sqrt(self.n_samples),
                'U_se': np.std(raw_dict['U_raw'], axis=0)/np.sqrt(self.n_samples),
                'H_se': np.std(raw_dict['H_raw'], axis=0)/np.sqrt(self.n_samples),
                'V_se': np.std(raw_dict['V_raw'], axis=0)/np.sqrt(self.n_samples),
                'P_se': np.std(raw_dict['P_raw'], axis=0)/np.sqrt(self.n_samples)
               }
    
    @staticmethod
    def _weight_function(x, a=0.2, b=1., c=5e-3):
        return a - (a - b) * np.exp(-c * x)

    def _get_mapped_quantity(self, quantity, remapped_temperatures, poly_order=5, spline=False):
        w = self._weight_function(self._raw_temperatures)
        if not spline:
            poly = np.polyfit(x=self._raw_temperatures, y=quantity, w=w, deg=poly_order)
            return np.poly1d(poly)(remapped_temperatures)
        else:
            spl = splrep(x=self._raw_temperatures, y=quantity, w=w, k=5)
            return splev(remapped_temperatures, spl)

    def _get_properties(self, U): 
        Cp = np.gradient(U, self.temperatures)
        S = cumtrapz(y=Cp/self.temperatures, x=self.temperatures, initial=0.)
        G = U-self.temperatures*S
        return Cp, S, G
    
    def get_unmapped_properties(self, poly_order=5, spline=True):
        mean_dict = self._get_mean_properties()
        U = self._get_mapped_quantity(mean_dict['U_mean'], self.temperatures, poly_order=poly_order, spline=spline)
        H = self._get_mapped_quantity(mean_dict['H_mean'], self.temperatures, poly_order=poly_order, spline=spline)
        V = self._get_mapped_quantity(mean_dict['V_mean'], self.temperatures, poly_order=poly_order, spline=spline)
        P = self._get_mapped_quantity(mean_dict['P_mean'], self.temperatures, poly_order=poly_order, spline=spline)
        Cp, S, G = self._get_properties(H)
        return {'T': self.temperatures.tolist(),
                'U': U.tolist(),
                'H': H.tolist(),
                'Cp': Cp.tolist(),
                'S': S.tolist(),
                'G': G.tolist(),
                'V': V.tolist(),
                'P': P.tolist()}
        
    def get_remapped_properties(self, remapped_temperatures, poly_order=5, spline=True):
        mean_dict = self._get_mean_properties()
        U = self._get_mapped_quantity(mean_dict['U_mean'], remapped_temperatures, poly_order=poly_order, spline=spline)
        H = self._get_mapped_quantity(mean_dict['H_mean'], remapped_temperatures, poly_order=poly_order, spline=spline)
        V = self._get_mapped_quantity(mean_dict['V_mean'], remapped_temperatures, poly_order=poly_order, spline=spline)
        P = self._get_mapped_quantity(mean_dict['P_mean'], remapped_temperatures, poly_order=poly_order, spline=spline)
        Cp, S, G = self._get_properties(H)
        return {'T': self.temperatures.tolist(),
                'remapped_U': U.tolist(),
                'remapped_H': H.tolist(),
                'remapped_Cp': Cp.tolist(),
                'remapped_S': S.tolist(),
                'remapped_G': G.tolist(),
                'remapped_V': V.tolist(),
                'remapped_P': P.tolist()}
