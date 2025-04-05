# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from scipy.interpolate import splrep, splev
from scipy.integrate import cumulative_trapezoid
from scipy.constants import physical_constants
KB = physical_constants['Boltzmann constant in eV/K'][0]
bar_to_GPa = 1e-4
eV_per_Ang_3_to_GPa = 160.2176565

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
variable         ts equal {{ time_step }}/1000 # Timestep.
variable         t_print equal {{ n_print}} # Dump frequency.
variable         t_traj equal {{ n_traj }} # Dump trajectory frequency
variable         t_avg equal round(0.5*{{ n_equib }}/{{ n_print}}) # N_repeat value for fix ave/atom.
#------------------------------------------------------------------------------#

#----------------------------- Run simulation ---------------------------------#

# Integrator and thermostat.
timestep         ${ts}
{% if pressure is not none %}
    {% if not langevin %}
        fix         f1 all npt temp {{ T_start }} {{ T_start }} 0.1 iso {{ pressure }} {{ pressure }} 1.0
    {% else %}
        fix         f1 all nph iso {{ pressure }} {{ pressure }} 1.0
        fix         f2 all langevin {{ T_start }} {{ T_start }} 0.1 ${rnd} zero yes
    {% endif %}
{% else %}
    {% if not langevin %}
        fix         f1 all nvt temp {{ T_start }} {{ T_start }} 0.1
    {% else %}
        fix         f1 all nve
        fix         f2 all langevin {{ T_start }} {{ T_start }} 0.1 ${rnd} zero yes
    {% endif %}
{% endif %}

# Initial temperature to accelerate equilibration.
variable         T_0 equal 2.0*{{ T_start }}
velocity         all create ${T_0} ${rnd} dist gaussian

# Compute mean positions and velocities.
fix              mean_pos all ave/atom ${t_print} ${t_avg} ${t_eq} x y z
fix              mean_vel all ave/atom ${t_print} ${t_avg} ${t_eq} vx vy vz

# Dump properties to check equilibration. Remove step, so that pyiron does not accidentally read this.
thermo_style     custom temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo           0

# Run equilibraiton at T_start.
run              ${t_eq}

# Run one extra step to ensure mean values are updated.
run              1

# Define atom-style variables to reference the computed mean values.
variable         xmean atom f_mean_pos[1]
variable         ymean atom f_mean_pos[2]
variable         zmean atom f_mean_pos[3]
variable         vxmean atom f_mean_vel[1]
variable         vymean atom f_mean_vel[2]
variable         vzmean atom f_mean_vel[3]

# Apply the mean positions and velocities to all atoms.
set              atom * x v_xmean y v_ymean z v_zmean
set              atom * vx v_vxmean vy v_vymean vz v_vzmean

# Unfix the collection of mean positions and velocities.
unfix            mean_pos
unfix            mean_vel

# Reset the counter.
reset_timestep   0

# Dump properties just once, so that pyiron has something to process.
thermo_style     custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol
thermo           0

# Dump trajectory.
dump             1 all custom ${t_traj} dump.out id type xsu ysu zsu fx fy fz vx vy vz
dump_modify      1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"

# Refix forward ramping.
{% if pressure is not none %}
    {% if not langevin %}
        unfix         f1
        fix           f1 all npt temp {{ T_start }} {{ T_stop }} 0.01 iso {{ pressure }} {{ pressure }} 1.0
    {% else %}
        unfix         f2
        fix           f2 all langevin {{ T_start }} {{ T_stop }} 0.01 ${rnd} zero yes
    {% endif %}
{% else %}
    {% if not langevin %}
        unfix         f1
        fix           f1 all nvt temp {{ T_start }} {{ T_stop }} 0.01
    {% else %}
        unfix         f2
        fix           f2 all langevin {{ T_start }} {{ T_stop }} 0.01 ${rnd} zero yes
    {% endif %}
{% endif %}

# Dump the necessary output.
fix              f3 all print ${t_print} "$(step) $(temp) $(pe) $(ke) $(enthalpy) $(vol) $(press)" title "# Step Temp[K] PE[eV] KE[eV] H[eV] Vol[Ang^3] P[eV/Ang^2]" screen no file forward.dat

# Run forward ramping between T_start and T_stop.
run              ${t}
"""

class TemperatureRampMD():
    def __init__(self, ref_job, temperatures, pressure, n_samples=5, n_ramp_steps=1000, 
                 n_equib_steps=100, n_print=1, n_traj_print=1, time_step=1., langevin=False, recompress=False):
        self.ref_job = ref_job
        self.temperatures = temperatures
        self.pressure = pressure
        self.n_samples = n_samples
        self.n_ramp_steps = n_ramp_steps
        self.n_equib_steps = n_equib_steps
        self.n_print = n_print
        self.n_traj_print = n_traj_print
        self.time_step = time_step
        self.langevin = langevin
        self.recompress = recompress
        
        self._project = self.ref_job.project
        self._job_names = ['sample_' + str(i) for i in range(self.n_samples)]
        self._n_atoms = self.ref_job.structure.get_number_of_atoms()
        if np.isclose(self.temperatures[0], 0, atol=1e-3):
            self.temperatures[0] += 0.01
        self._raw_temperatures = np.linspace(self.temperatures[0], self.temperatures[-1], int(self.n_ramp_steps/self.n_print)+1)
    
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
            pressure=self.pressure/bar_to_GPa if self.pressure is not None else self.pressure,
            langevin=self.langevin
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
            elif job_status[i] not in ['finished', 'running', 'collect'] or delete_existing_jobs:
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
        T_raw, E_raw, U_raw, H_raw, V_raw, P_raw = zip(*[(val[1], val[2]/self._n_atoms, val[7]/self._n_atoms, val[4]/self._n_atoms, 
                                                          val[5]/self._n_atoms, val[6]*bar_to_GPa) for val in [self._get_values(data_dict) 
                                                                                                               for data_dict in output_dicts]])
        if self.pressure is not None:
            H_raw = np.array([U_raw[i]+self.pressure/eV_per_Ang_3_to_GPa*V_raw[i] for i in range(self.n_samples)])
            
        return {'T_raw': T_raw,
                'E_raw': E_raw,
                'U_raw': U_raw,
                'H_raw': H_raw,
                'V_raw': V_raw,
                'P_raw': P_raw
               }
    
    def _get_mean_properties(self):
        raw_dict = self._get_raw_properties()
        return {'T_mean': np.mean(raw_dict['T_raw'], axis=0),
                'E_mean': np.mean(raw_dict['E_raw'], axis=0),
                'U_mean': np.mean(raw_dict['U_raw'], axis=0),
                'H_mean': np.mean(raw_dict['H_raw'], axis=0),
                'V_mean': np.mean(raw_dict['V_raw'], axis=0),
                'P_mean': np.mean(raw_dict['P_raw'], axis=0),
                'T_se': np.std(raw_dict['T_raw'], axis=0)/np.sqrt(self.n_samples),
                'E_se': np.std(raw_dict['E_raw'], axis=0)/np.sqrt(self.n_samples),
                'U_se': np.std(raw_dict['U_raw'], axis=0)/np.sqrt(self.n_samples),
                'H_se': np.std(raw_dict['H_raw'], axis=0)/np.sqrt(self.n_samples),
                'V_se': np.std(raw_dict['V_raw'], axis=0)/np.sqrt(self.n_samples),
                'P_se': np.std(raw_dict['P_raw'], axis=0)/np.sqrt(self.n_samples)
               }
    
    @staticmethod
    def _weight_function(x, a=0.2, b=1., c=5e-3):
        return a - (a - b) * np.exp(-c * x)

    def _get_mapped_quantity(self, quantity, remapped_temperatures, poly_order=5, spline=False, weight=True):
        if not weight:
            w = None
        else:
            w = self._weight_function(self._raw_temperatures)

        if not spline:
            poly = np.polyfit(x=self._raw_temperatures, y=quantity, w=w, deg=poly_order)
            return np.poly1d(poly)(remapped_temperatures)
        else:
            spl = splrep(x=self._raw_temperatures, y=quantity, w=w, k=poly_order)
            return splev(remapped_temperatures, spl)

    def _get_properties(self, U): 
        Cp = np.gradient(U, self.temperatures, edge_order=2)
        S = cumulative_trapezoid(y=Cp/self.temperatures, x=self.temperatures, initial=0.)
        F = U-self.temperatures*S
        return Cp, S, F
    
    def get_unmapped_properties(self, poly_order=5, spline=True, weight=True):
        mean_dict = self._get_mean_properties()
        U = self._get_mapped_quantity(mean_dict['U_mean'], self.temperatures, poly_order=poly_order, spline=spline, weight=weight)
        H = self._get_mapped_quantity(mean_dict['H_mean'], self.temperatures, poly_order=poly_order, spline=spline, weight=weight)
        V = self._get_mapped_quantity(mean_dict['V_mean'], self.temperatures, poly_order=poly_order, spline=spline, weight=weight)
        P = self._get_mapped_quantity(mean_dict['P_mean'], self.temperatures, poly_order=poly_order, spline=spline, weight=weight)
        Cv, Sv, F = self._get_properties(U)
        Cp, Sp, G = self._get_properties(H)
        return {'T': self.temperatures.tolist(),
                'U': U.tolist(),
                'H': H.tolist(),
                'Cv': Cv.tolist(),
                'Cp': Cp.tolist(),
                'Sv': Sv.tolist(),
                'Sp': Sp.tolist(),
                'F': F.tolist(),
                'G': G.tolist(),
                'V': V.tolist(),
                'P': P.tolist()}
        
    def get_remapped_properties(self, remapped_temperatures, poly_order=5, spline=True, weight=True):
        mean_dict = self._get_mean_properties()
        U = self._get_mapped_quantity(mean_dict['U_mean'], remapped_temperatures, poly_order=poly_order, spline=spline, weight=weight)
        H = self._get_mapped_quantity(mean_dict['H_mean'], remapped_temperatures, poly_order=poly_order, spline=spline, weight=weight)
        V = self._get_mapped_quantity(mean_dict['V_mean'], remapped_temperatures, poly_order=poly_order, spline=spline, weight=weight)
        P = self._get_mapped_quantity(mean_dict['P_mean'], remapped_temperatures, poly_order=poly_order, spline=spline, weight=weight)
        Cv, Sv, F = self._get_properties(U)
        Cp, Sp, G = self._get_properties(H)
        return {'T': self.temperatures.tolist(),
                'U': U.tolist(),
                'H': H.tolist(),
                'Cv': Cv.tolist(),
                'Cp': Cp.tolist(),
                'Sv': Sv.tolist(),
                'Sp': Sp.tolist(),
                'F': F.tolist(),
                'G': G.tolist(),
                'V': V.tolist(),
                'P': P.tolist()}
