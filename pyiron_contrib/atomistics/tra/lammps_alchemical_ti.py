# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from scipy.integrate import simpson
import scipy.constants as sc
KB = sc.value('Boltzmann constant in eV/K')
EV = sc.value('electron volt')
H = sc.value('Planck constant in eV/Hz')
AMU = sc.value('atomic mass constant')
bar_to_GPa = 1e-4

from jinja2 import Template
from io import StringIO
import os

lammps_input = """\
# This script runs alchemical TILD.

# Base parameters.
units            metal
dimension        3
boundary         p p p
atom_style       atomic
atom_modify      map yes

# Create atoms.
read_data        structure.inp

# Initalizes the random number generator.
variable         rnd equal round(random(0,99999,{{ seed }}))
variable         KB equal 8.617333262E-05

# Simulation control parameters.
variable         t_eq equal round({{ n_equib }}) # Equilibration steps.
variable         t equal round({{ n_steps }}) # Switching steps.
variable         ts equal {{ time_step }}/1000 # Timestep.
variable         t_print equal {{ n_print}} # Dump frequency of properties.
variable         atom_id equal round({{ atom_id }}+1) # Id of the atom to be removed

# Define interatomic potential.
pair_style       hybrid {{ pair_style }} zero 0.0 nocoeff
pair_coeff       {{ pair_coeff }} NULL
pair_coeff       1 2 none
pair_coeff       2 1 none
pair_coeff       2 2 zero

# Integrator and thermostat.
timestep         ${ts}
{% if pressure is not none %}
    {% if langevin == True %}
        fix f1 all nph iso {{ pressure }} {{ pressure }} 1.0
    {% else %}
        fix f1 all npt temp {{ temperature }} {{ temperature }} 0.1 iso {{ pressure }} {{ pressure }} 1.0
    {% endif %}
{% else %}
    {% if langevin == True %}
        fix f1 all nve
    {% else %}
        fix f1 all nvt temp {{ temperature }} {{ temperature }} 0.1
    {% endif %}
{% endif %}

{% if langevin == True %}
    fix f2 all langevin {{ temperature }} {{ temperature }} 0.1 ${rnd} zero yes
{% endif %}

# Initial temperature to accelerate equilibration.
variable         T_0 equal 2.0*{{ temperature }}
velocity         all create ${T_0} ${rnd} dist gaussian

# Replace harmonic osciallators with interacting atoms if pure.
set              type 2 type 1

# Compute and record the average msd to determine the spring constant k.
compute          msd all msd average yes com yes
variable         n_every equal round(${t_eq}/100)
variable         n_repeat equal 50
fix              ave_msd all ave/time ${n_every} ${n_repeat} ${t_eq} c_msd[4]

# Also record the average positions.
group            pos id ${atom_id}
compute          upos pos property/atom xu yu zu
fix              ave_pos pos ave/atom ${n_every} ${n_repeat} ${t_eq} c_upos[1] c_upos[2] c_upos[3]

# Run equilibraiton at T_start.
run              ${t_eq}

# Determine k.
variable         ave_msd equal f_ave_msd
variable         ave_x equal f_ave_pos[${atom_id}][1]
variable         ave_y equal f_ave_pos[${atom_id}][2]
variable         ave_z equal f_ave_pos[${atom_id}][3]
variable         k equal 3*${KB}*{{ temperature }}/${ave_msd}
print            "$k ${ave_x} ${ave_y} ${ave_z}" file spring.dat

# Define partitions.
variable         name world pure alloy

# Replace harmonic osciallators with interacting atoms if pure.
if "${name} == alloy" then "set atom 1 type 2"

# Define the harmonic oscillator by tethering the 0th atom to the origin.
if "${name} == alloy" then &
    "group ho id 1" & 
    "fix f_ho ho spring tether $k ${ave_x} ${ave_y} ${ave_z} 0.0" &
    "fix_modify f_ho energy yes"

# Define ramp variable to combine the two different partitions.
if "${name} == alloy" then &
    "variable ramp equal ramp(0.0,1.0)" &
else &
    "variable ramp equal ramp(1.0,0.0)"

# Call the fix.
fix              al all alchemy v_ramp

# Pressure of both the systems is the same and calculated with pressure/alchemy.
compute          pressure all pressure/alchemy al

# Thermo outputs.
thermo_style     custom step temp press ke f_al f_al[1] f_al[2] f_al[3]
thermo_modify    colname f_al lambda colname f_al[1] BulkEPot colname f_al[2] AlloyEPot colname f_al[3] EPot_mixed
thermo_modify    press pressure
thermo           ${t_print}

# Run MD.
run              ${t}
"""

class VacancyFormationTI():
    def __init__(self, ref_job, temperature, pressure, atom_id=0, n_samples=1, n_steps=1000, n_equib_steps=100, n_print=1, time_step=1.,
                 langevin=True, recompress=False):
        self.ref_job = ref_job
        self.temperature = temperature
        self.pressure = pressure
        self.atom_id = atom_id
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_equib_steps = n_equib_steps
        self.n_print = n_print
        self.time_step = time_step
        self.langevin = langevin
        self.recompress = recompress
        
        self._project = self.ref_job.project
        self._job_names = ['sample_' + str(i) for i in range(self.n_samples)]
        self._structure = self.ref_job.structure.copy()
        self._n_atoms = self._structure.get_number_of_atoms()
        if self.pressure is not None:
            self.pressure *= 1/bar_to_GPa
        self._split_potential_strings()
        self._create_ho_structure()
    
    def _generate_input_script(self):
        template = Template(lammps_input)
        input_script = template.render(
            seed=np.random.randint(99999),
            pair_style = self._pair_style,
            pair_coeff = self._pair_coeff,
            temperature=self.temperature,
            n_equib=self.n_equib_steps,
            n_steps=self.n_steps,
            n_print=self.n_print,
            time_step=self.time_step,
            pressure=self.pressure,
            atom_id=self.atom_id,
            langevin=self.langevin
        )
        return input_script

    def _split_potential_strings(self):
        potential = self.ref_job.potential
        self._pair_style = potential['Config'][0][0].split()[1]
        pair_coeff = potential['Config'][0][1].split()
        self._pair_coeff = ' '.join(pair_coeff[1:])

    def _create_ho_structure(self):
        self._structure = self.ref_job.structure.copy()
        self._structure[self.atom_id] = 'H'

    def _create_ho_potential(self):
        
        
    def _run_job(self, project, job_name):
        job = self.ref_job.copy_template(project=project, new_job_name=job_name)
        job.structure = self._structure.copy()
        job.potential = self._potential
        job.input.control.load_string(self._generate_input_script())
        try:
            job.run()
        except EmptyDataError:
            job.status.finished = True
            job.compress()
        
    def run_md(self, delete_existing_jobs=False):
        job_list = self._project.job_table().job.to_list()
        job_status = self._project.job_table().status.to_list()
        for job_name in self._job_names:
            if job_name not in job_list:
                self._run_job(project=self._project, job_name=job_name)
            elif job_status[job_list.index(job_name)] not in ['finished', 'running', 'collect'] or delete_existing_jobs:
                self._project.remove_job(job_name)
                self._run_job(project=self._project, job_name=job_name)

    def _collect_spring_data(self, working_directory):
        k, x, y, z = np.loadtxt(working_directory+'/spring.dat')
        return k, np.array([x, y, z])
                
    def _collect_output(self, job_name):
        job = self._project.inspect(job_name)
        job.decompress()
        log_name = job.working_directory+'/log.lammps.0'
        with open(log_name, "r") as f:
            read_thermo = False
            first_step = False
            thermo_lines = ""
            for l in f:
                l = l.lstrip()
                if read_thermo:
                    if l.startswith("Loop") or l.startswith("ERROR"):
                        read_thermo = False
                        continue
                    thermo_lines += l
                if l.startswith("Step"):
                    if not first_step:
                        first_step = True
                        continue
                    else:
                        read_thermo = True
                        thermo_lines += l
        df_pandas = pd.read_csv(StringIO(thermo_lines), sep="\s+", engine="python")
        df_arrays = df_pandas.to_numpy().T
        if self.recompress:
            job.compress()
        return df_arrays, *self._collect_spring_data(working_directory=job.working_directory)

    def get_output(self):
        output_arrays, k_s, tether_pos = zip(*[self._collect_output(job_name) for job_name in self._job_names])
        output_arrays = np.array(output_arrays)
        data_dict = {
            'Step': output_arrays[:, 0],
            'Temperature': output_arrays[:, 1],
            'Pressure': output_arrays[:, 2],
            'energy_kin': output_arrays[:, 3],
            'lambda': np.flip(output_arrays[:, 4]),
            'bulk_energy_pot': output_arrays[:, 5],
            'alloy_energy_pot': output_arrays[:, 6],
            'energy_pot_mix': output_arrays[:, 7],
            'spring_constant': np.array(k_s),
            'tether_positions': np.array(tether_pos)
        }
        return data_dict

    def get_formation_energy(self, per_atom_bulk_free_energy, averaged=True):
        data_dict = self.get_output()
        G_form = []
        for n in range(self.n_samples):
            lambdas = data_dict['lambda'][n]
            U_bulk = data_dict['bulk_energy_pot'][n]
            U_vac = data_dict['alloy_energy_pot'][n]
            k = data_dict['spring_constant'][n]
            
            mass = self._structure.get_masses()[0]
            omega = np.sqrt(k*EV/(mass*AMU))*1.0e+10
            nu = omega/(2*np.pi)
            F_harm = 3*KB*self.temperature*np.log(H*nu/(KB*self.temperature))
            G_form.append(per_atom_bulk_free_energy+simpson(y=U_vac-U_bulk, x=lambdas)-F_harm)
        G_form = np.array(G_form)
        if not averaged:
            return G_form
        else:
            return G_form.mean(), G_form.std()/np.sqrt(self.n_samples)
