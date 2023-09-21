# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from itertools import islice
import numbers
import numpy as np
import os
import pandas as pd
import posixpath
import scipy.constants
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_base import (
    state,
    GenericParameters,
    GenericJob,
    Executable,
    FlattenedStorage,
)
from pyiron_atomistics import ase_to_pyiron, Atoms
from pyiron_contrib.atomistics.ml.potentialfit import PotentialFit
from pyiron_contrib.atomistics.mlip.cfgs import (
    savecfgs,
    loadcfgs,
    Cfg,
    load_grades_ids_and_timesteps,
)
from pyiron_contrib.atomistics.mlip.potential import MtpPotential

__author__ = "Jan Janssen"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2017"


gpa_to_ev_ang = (
    1e22 / scipy.constants.physical_constants["joule-electron volt relationship"][0]
)


class Mlip(GenericJob, PotentialFit):
    def __init__(self, project, job_name):
        super(Mlip, self).__init__(project, job_name)
        self.__version__ = "0.1.0"
        self.__name__ = "Mlip"
        self._executable_activate()
        self._job_dict = {}
        self.input = MlipParameter()
        self._command_line = CommandLine()
        self._potential = MtpPotential()

    def _executable_activate(self, enforce=False):
        if self._executable is None or enforce:
            if len(self.__module__.split(".")) > 1:
                self._executable = Executable(
                    codename=self.__name__,
                    module=self.__module__.split(".")[-2],
                    path_binary_codes=state.settings.resource_paths,
                )
            else:
                self._executable = Executable(
                    codename=self.__name__,
                    path_binary_codes=state.settings.resource_paths,
                )

    @property
    def calculation_dataframe(self):
        (
            job_id_lst,
            time_step_start_lst,
            time_step_end_lst,
            time_step_delta_lst,
        ) = self._job_dict_lst(self._job_dict)
        return pd.DataFrame(
            {
                "Job id": job_id_lst,
                "Start config": time_step_start_lst,
                "End config": time_step_end_lst,
                "Step": time_step_delta_lst,
            }
        )

    @property
    def potential_files(self):
        pot = os.path.join(self.working_directory, "Trained.mtp_")
        states = os.path.join(self.working_directory, "state.mvs")
        if os.path.exists(pot) and os.path.exists(states):
            return [pot, states]

    def _get_elements(self):
        """
        Return elements in training in insertion order, i.e. elements seen earlier get lower indices.
        """
        elements = []
        for job_id in self._job_dict:
            j = self.project.inspect(job_id)
            if j["NAME"] == "TrainingContainer":
                candidates = j.to_object().get_elements()
            else:
                candidates = j["input/structure/species"]
            for e in candidates:
                if e not in elements:
                    elements.append(e)
        return elements

    def potential_dataframe(self, elements=None):
        """
        :class:`pandas.DataFrame`: potential dataframe for lammps jobs

        .. attention::

            The `elements` argument must be given for in any case for non-unary alloys, as the scrapping code from the
            training data is currently broken!

        Args:
            elements (list of str, optional): name of elements in this potential in the order of their indices in pyiron
                                              structures, if not given use elements seen in training data
        """
        if elements is None:
            elements = self.input["species"]
            if elements is None:
                elements = self._get_elements()  # AAAH

        if self.status.finished:
            return pd.DataFrame(
                {
                    "Name": ["".join(elements)],
                    "Filename": [self.potential_files],
                    "Model": ["Custom"],
                    "Species": [elements],
                    "Config": [["pair_style mlip mlip.ini\n", "pair_coeff * *\n"]],
                }
            )
        else:
            raise ValueError(
                f"Potential only available after job is finished, not {self.status}!"
            )

    @property
    def potential(self):
        if self.status.finished and self._potential is not None:
            return self._potential
        else:
            raise ValueError("potential only available on successfully finished jobs")

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        self.input.read_only = True
        self._command_line.read_only = True

    def add_job_to_fitting(
        self, job_id, time_step_start=0, time_step_end=-1, time_step_delta=10
    ):
        if time_step_end == -1:
            time_step_end = (
                np.shape(self.project.inspect(int(job_id))["output/generic/cells"])[0]
                - 1
            )
        self._job_dict[job_id] = {
            "time_step_start": time_step_start,
            "time_step_end": time_step_end,
            "time_step_delta": time_step_delta,
        }

    def write_input(self):
        restart_file_lst = [
            os.path.basename(f) for f in self._restart_file_list
        ] + list(self._restart_file_dict.values())
        if "testing.cfg" not in restart_file_lst:
            species = self._write_test_set(
                file_name="testing.cfg", cwd=self.working_directory
            )
            species_count = len(species)
            self.input["species"] = species
        elif self.input["filepath"] != "auto":
            species_count = 0
        else:
            raise ValueError("speciescount not set")
        if "training.cfg" not in restart_file_lst:
            self._write_training_set(
                file_name="training.cfg", cwd=self.working_directory
            )
        self._copy_potential(
            species_count=species_count,
            file_name="start.mtp",
            cwd=self.working_directory,
        )
        if self.version != "1.0.0":
            self._command_line.activate_postprocessing()
        self._command_line[0] = self._command_line[0].replace(
            "energy_auto", str(self.input["energy-weight"])
        )
        self._command_line[0] = self._command_line[0].replace(
            "force_auto", str(self.input["force-weight"])
        )
        self._command_line[0] = self._command_line[0].replace(
            "stress_auto", str(self.input["stress-weight"])
        )
        self._command_line[0] = self._command_line[0].replace(
            "iteration_auto", str(int(self.input["iteration"]))
        )
        self._command_line.write_file(file_name="mlip.sh", cwd=self.working_directory)

        species = np.array(species)
        input_store = StructureStorage()
        input_store.add_array("energy", dtype=np.float64, shape=(), per="chunk")
        input_store.add_array("forces", dtype=np.float64, shape=(3,), per="element")
        input_store.add_array("stress", dtype=np.float64, shape=(6,), per="chunk")
        for cfg in loadcfgs(os.path.join(self.working_directory, "training.cfg")):
            struct = Atoms(
                symbols=species[np.cast[np.int64](cfg.types)],
                positions=cfg.pos,
                cell=cfg.lat,
                pbc=[True] * 3,
            )  # HACK for pbc
            input_store.add_structure(
                struct,
                identifier=cfg.desc,
                energy=cfg.energy,
                forces=cfg.forces,
                stress=cfg.stresses,
            )

        with self.project_hdf5.open("input") as hdf5_input:
            input_store.to_hdf(hdf=hdf5_input, group_name="training_data")

    def collect_logfiles(self):
        pass

    def collect_output(self):
        file_name = os.path.join(self.working_directory, "diff.cfg")
        if os.path.exists(file_name):
            _, job_id_diff_lst, timestep_diff_lst = load_grades_ids_and_timesteps(
                file_name
            )
        else:
            job_id_diff_lst, timestep_diff_lst = [], []
        file_name = os.path.join(self.working_directory, "selected.cfg")
        if os.path.exists(file_name):
            (
                _,
                job_id_new_training_lst,
                timestep_new_training_lst,
            ) = load_grades_ids_and_timesteps(file_name)
        else:
            job_id_new_training_lst, timestep_new_training_lst = [], []
        file_name = os.path.join(self.working_directory, "grades.cfg")
        if os.path.exists(file_name):
            (
                grades_lst,
                job_id_grades_lst,
                timestep_grades_lst,
            ) = load_grades_ids_and_timesteps(file_name)
        else:
            grades_lst, job_id_grades_lst, timestep_grades_lst = [], [], []
        try:
            self._potential.load(os.path.join(self.working_directory, "Trained.mtp_"))
        except:
            self.logger.warn(
                "Failed to parse potential file! job.potential will not be available."
            )
            self._potential = None

        training_store = FlattenedStorage()
        training_store.add_array("energy", dtype=np.float64, shape=(), per="chunk")
        training_store.add_array("forces", dtype=np.float64, shape=(3,), per="element")
        training_store.add_array("stress", dtype=np.float64, shape=(6,), per="chunk")
        for cfg in loadcfgs(os.path.join(self.working_directory, "training_efs.cfg")):
            training_store.add_chunk(
                len(cfg.pos),
                identifier=cfg.desc,
                energy=cfg.energy,
                forces=cfg.forces,
                stress=cfg.stresses,
            )

        testing_store = FlattenedStorage()
        testing_store.add_array("energy", dtype=np.float64, shape=(), per="chunk")
        testing_store.add_array("forces", dtype=np.float64, shape=(3,), per="element")
        testing_store.add_array("stress", dtype=np.float64, shape=(6,), per="chunk")
        for cfg in loadcfgs(os.path.join(self.working_directory, "testing_efs.cfg")):
            testing_store.add_chunk(
                len(cfg.pos),
                identifier=cfg.desc,
                energy=cfg.energy,
                forces=cfg.forces,
                stress=cfg.stresses,
            )

        with self.project_hdf5.open("output") as hdf5_output:
            hdf5_output["grades"] = grades_lst
            hdf5_output["job_id"] = job_id_grades_lst
            hdf5_output["timestep"] = timestep_grades_lst
            hdf5_output["job_id_diff"] = job_id_diff_lst
            hdf5_output["timestep_diff"] = timestep_diff_lst
            hdf5_output["job_id_new"] = job_id_new_training_lst
            hdf5_output["timestep_new"] = timestep_new_training_lst
            if self._potential is not None:
                self._potential.to_hdf(hdf=hdf5_output)
            training_store.to_hdf(hdf=hdf5_output, group_name="training_efs")
            testing_store.to_hdf(hdf=hdf5_output, group_name="testing_efs")

    def get_structure(self, iteration_step=-1):
        job = self.project.load(self["output/job_id_diff"][iteration_step])
        return job.get_structure(self["output/timestep_diff"][iteration_step])

    def to_hdf(self, hdf=None, group_name=None):
        super(Mlip, self).to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(self._hdf5)
        with self._hdf5.open("input") as hdf_input:
            (
                job_id_lst,
                time_step_start_lst,
                time_step_end_lst,
                time_step_delta_lst,
            ) = self._job_dict_lst(self._job_dict)
            hdf_input["job_id_list"] = job_id_lst
            hdf_input["time_step_start_lst"] = time_step_start_lst
            hdf_input["time_step_end_lst"] = time_step_end_lst
            hdf_input["time_step_delta_lst"] = time_step_delta_lst

    def from_hdf(self, hdf=None, group_name=None):
        super(Mlip, self).from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(self._hdf5)
        with self._hdf5.open("input") as hdf_input:
            for job_id, start, end, delta in zip(
                hdf_input["job_id_list"],
                hdf_input["time_step_start_lst"],
                hdf_input["time_step_end_lst"],
                hdf_input["time_step_delta_lst"],
            ):
                self.add_job_to_fitting(
                    job_id=job_id,
                    time_step_start=start,
                    time_step_end=end,
                    time_step_delta=delta,
                )
        if self.status.finished:
            with self._hdf5.open("output") as hdf_output:
                if "potential" in hdf_output.list_groups():
                    self._potential.from_hdf(hdf=hdf_output)

    def get_suggested_number_of_configuration(
        self, species_count=None, multiplication_factor=2.0
    ):
        if self.input["filepath"] == "auto":
            potential = self.input["potential"]
            if isinstance(potential, numbers.Integral):
                potential = f"{potential:02}g.mtp"
            source = self._find_potential(potential)
        else:
            source = self.input["filepath"]
        with open(source) as f:
            lines = f.readlines()
        radial_basis_size, radial_funcs_count, alpha_scalar_moments = 0.0, 0.0, 0.0
        for line in lines:
            if "radial_basis_size" in line:
                radial_basis_size = int(line.split()[2])
            if "radial_funcs_count" in line:
                radial_funcs_count = int(line.split()[2])
            if "species_count" in line and species_count is None:
                species_count = int(line.split()[2])
            if "alpha_scalar_moments" in line:
                alpha_scalar_moments = int(line.split()[2])
        return int(
            multiplication_factor
            * (
                radial_basis_size * radial_funcs_count * species_count**2
                + alpha_scalar_moments
            )
        )

    @staticmethod
    def _update_species_count(lines, species_count):
        ind = 0
        for ind, line in enumerate(lines):
            if "species_count" in line:
                break
        species_line = lines[ind].split()
        species_line[-1] = str(species_count)
        lines[ind] = " ".join(species_line) + "\n"
        return lines

    @staticmethod
    def _update_min_max_distance(lines, min_distance, max_distance):
        min_dis_updated, max_dis_updated = False, False
        for ind, line in enumerate(lines):
            if "min_dist" in line and min_distance is not None:
                min_dist_line = lines[ind].split()
                min_dist_line[2] = str(min_distance)
                lines[ind] = " ".join(min_dist_line) + "\n"
                min_dis_updated = True
            elif "max_dist" in line and max_distance is not None:
                max_dist_line = lines[ind].split()
                max_dist_line[2] = str(max_distance)
                lines[ind] = " ".join(max_dist_line) + "\n"
                max_dis_updated = True
            elif min_dis_updated and max_dis_updated:
                break
        return lines

    def _copy_potential(self, species_count, file_name, cwd=None):
        if cwd is not None:
            file_name = posixpath.join(cwd, file_name)
        if self.input["filepath"] == "auto":
            potential = self.input["potential"]
            if isinstance(potential, numbers.Integral):
                potential = f"{potential:02}g.mtp"
            source = self._find_potential(potential)
        else:
            source = self.input["filepath"]
        with open(source) as f:
            lines = f.readlines()
        if self.input["filepath"] == "auto":
            lines_modified = self._update_species_count(lines, species_count)
            lines_modified = self._update_min_max_distance(
                lines=lines_modified,
                min_distance=self.input["min_dist"],
                max_distance=self.input["max_dist"],
            )
        else:
            lines_modified = lines
        with open(file_name, "w") as f:
            f.writelines(lines_modified)

    @staticmethod
    def stress_tensor_components(stresses):
        return (
            np.array(
                [
                    stresses[0][0],
                    stresses[1][1],
                    stresses[2][2],
                    stresses[1][2],
                    stresses[0][1],
                    stresses[0][2],
                ]
            )
            * gpa_to_ev_ang
        )

    def _write_configurations(
        self, file_name="training.cfg", cwd=None, respect_step=True
    ):
        if cwd is not None:
            file_name = posixpath.join(cwd, file_name)
        (
            indices_lst,
            position_lst,
            forces_lst,
            cell_lst,
            energy_lst,
            track_lst,
            stress_lst,
        ) = ([], [], [], [], [], [], [])
        all_species = set()
        for job_id, value in self._job_dict.items():
            ham = self.project.inspect(job_id)
            if respect_step:
                start = value["time_step_start"]
                end = value["time_step_end"] + 1
                delta = value["time_step_delta"]
            else:
                start, end, delta = 0, None, 1
            time_step = start
            # HACK: until the training container has a proper HDF5 interface
            if ham.__name__ == "TrainingContainer":
                job = ham.to_object()
                all_species.update(job.get_elements())
                symbol_map = {s: i for i, s in enumerate(sorted(set(all_species)))}
                if end is None:
                    end = job.number_of_structures
                for time_step in range(start, end, delta):
                    symbols = job._container.get_array("symbols", time_step)
                    indices_lst.append([symbol_map[s] for s in symbols])
                    position_lst.append(
                        job._container.get_array("positions", time_step)
                    )
                    forces_lst.append(job._container.get_array("forces", time_step))
                    cell_lst.append(job._container.get_array("cell", time_step))
                    energy_lst.append(job._container.get_array("energy", time_step))
                    if job._container.has_array("stress"):
                        volume = np.abs(np.linalg.det(cell_lst[-1]))
                        stress_lst.append(
                            job._container.get_array("stress", time_step) * volume
                        )
                        if np.isnan(stress_lst[-1]).any():
                            stress_lst[-1] = None
                    track_lst.append(str(ham.job_id) + "_" + str(time_step))
                continue
            original_dict = {
                el: ind for ind, el in enumerate(sorted(ham["input/structure/species"]))
            }
            species_dict = {
                ind: original_dict[el]
                for ind, el in enumerate(ham["input/structure/species"])
            }
            # HACK: this does not preserve element order, but the old code is broken for structures with different
            # species anyway
            all_species.update(ham["input/structure/species"])
            if ham.__name__ in [
                "Vasp",
                "ThermoIntDftEam",
                "ThermoIntDftMtp",
                "ThermoIntVasp",
            ]:
                if len(ham["output/outcar/stresses"]) != 0:
                    for position, forces, cell, energy, stresses, volume in zip(
                        ham["output/generic/positions"][start:end:delta],
                        ham["output/generic/forces"][start:end:delta],
                        ham["output/generic/cells"][start:end:delta],
                        ham["output/generic/dft/energy_free"][start:end:delta],
                        ham["output/outcar/stresses"][start:end:delta],
                        ham["output/generic/volume"][start:end:delta],
                    ):
                        indices_lst.append(
                            [species_dict[el] for el in ham["input/structure/indices"]]
                        )
                        position_lst.append(position)
                        forces_lst.append(forces)
                        cell_lst.append(cell)
                        energy_lst.append(energy)
                        stress_lst.append(stresses * volume / gpa_to_ev_ang)
                        track_lst.append(str(ham.job_id) + "_" + str(time_step))
                        time_step += delta
                else:
                    for position, forces, cell, energy in zip(
                        ham["output/generic/positions"][start:end:delta],
                        ham["output/generic/forces"][start:end:delta],
                        ham["output/generic/cells"][start:end:delta],
                        ham["output/generic/dft/energy_free"][start:end:delta],
                    ):
                        indices_lst.append(
                            [species_dict[el] for el in ham["input/structure/indices"]]
                        )
                        position_lst.append(position)
                        forces_lst.append(forces)
                        cell_lst.append(cell)
                        energy_lst.append(energy)
                        track_lst.append(str(ham.job_id) + "_" + str(time_step))
                        time_step += delta
            elif ham.__name__ in ["Lammps", "LammpsInt2", "LammpsMlip"]:
                for position, forces, cell, energy, stresses, volume in zip(
                    ham["output/generic/positions"][start:end:delta],
                    ham["output/generic/forces"][start:end:delta],
                    ham["output/generic/cells"][start:end:delta],
                    ham["output/generic/energy_pot"][start:end:delta],
                    ham["output/generic/pressures"][start:end:delta],
                    ham["output/generic/volume"][start:end:delta],
                ):
                    indices_lst.append(
                        [species_dict[el] for el in ham["input/structure/indices"]]
                    )
                    position_lst.append(position)
                    forces_lst.append(forces)
                    cell_lst.append(cell)
                    energy_lst.append(energy)
                    stress_lst.append(self.stress_tensor_components(stresses * volume))
                    track_lst.append(str(ham.job_id) + "_" + str(time_step))
                    time_step += delta
            else:
                for position, forces, cell, energy, stresses, volume in zip(
                    ham["output/generic/positions"][start:end:delta],
                    ham["output/generic/forces"][start:end:delta],
                    ham["output/generic/cells"][start:end:delta],
                    ham["output/generic/energy_pot"][start:end:delta],
                    ham["output/generic/pressures"][start:end:delta],
                    ham["output/generic/volume"][start:end:delta],
                ):
                    indices_lst.append(
                        [species_dict[el] for el in ham["input/structure/indices"]]
                    )
                    position_lst.append(position)
                    forces_lst.append(forces)
                    cell_lst.append(cell)
                    energy_lst.append(energy)
                    stress_lst.append(self.stress_tensor_components(stresses * volume))
                    track_lst.append(str(ham.job_id) + "_" + str(time_step))
                    time_step += delta
        write_cfg(
            file_name=file_name,
            indices_lst=indices_lst,
            position_lst=position_lst,
            cell_lst=cell_lst,
            forces_lst=forces_lst,
            energy_lst=energy_lst,
            track_lst=track_lst,
            stress_lst=stress_lst,
        )
        return list(all_species)

    def _write_test_set(self, file_name="testing.cfg", cwd=None):
        return self._write_configurations(
            file_name=file_name, cwd=cwd, respect_step=False
        )

    def _write_training_set(self, file_name="training.cfg", cwd=None):
        self._write_configurations(file_name=file_name, cwd=cwd, respect_step=True)

    @staticmethod
    def _job_dict_lst(job_dict):
        job_id_lst, time_step_start_lst, time_step_end_lst, time_step_delta_lst = (
            [],
            [],
            [],
            [],
        )
        for key, value in job_dict.items():
            job_id_lst.append(key)
            time_step_start_lst.append(value["time_step_start"])
            time_step_end_lst.append(value["time_step_end"])
            time_step_delta_lst.append(value["time_step_delta"])
        return job_id_lst, time_step_start_lst, time_step_end_lst, time_step_delta_lst

    @staticmethod
    def _find_potential(potential_name):
        for resource_path in state.settings.resource_paths:
            if os.path.exists(
                os.path.join(resource_path, "mlip", "potentials", "templates")
            ):
                resource_path = os.path.join(
                    resource_path, "mlip", "potentials", "templates"
                )
            if "potentials" in resource_path and os.path.exists(
                os.path.join(resource_path, potential_name)
            ):
                return os.path.join(resource_path, potential_name)
        raise ValueError("Potential not found!")

    # PotentialFit Implementation
    def _add_training_data(self, container):
        self.add_job_to_fitting(container.id, 0, container.number_of_structures - 1, 1)

    def _get_training_data(self):
        # TODO/BUG: only works after input is written for now, instead this should go over _job_
        return self["input/training_data"].to_object()

    def _get_predicted_data(self):
        return self["output/training_efs"].to_object()

    def get_lammps_potential(self):
        return self.potential_dataframe()


class MlipParameter(GenericParameters):
    def __init__(self, separator_char=" ", comment_char="#", table_name="mlip_inp"):
        super(MlipParameter, self).__init__(
            separator_char=separator_char,
            comment_char=comment_char,
            table_name=table_name,
        )

    def load_default(self, file_content=None):
        if file_content is None:
            file_content = """\
potential 16g.mtp
filepath auto
energy-weight 1
force-weight 0.01
stress-weight 0.001
iteration 1000
min_dist 2.0
max_dist 5.0
"""
        self.load_string(file_content)


class CommandLine(GenericParameters):
    def __init__(self, input_file_name=None, **qwargs):
        super(CommandLine, self).__init__(
            input_file_name=input_file_name,
            table_name="cmd",
            comment_char="#",
            val_only=True,
        )

    def load_default(self, file_content=None):
        if file_content is None:
            file_content = """\
$MLP_COMMAND_PARALLEL train --energy-weight=energy_auto --force-weight=force_auto --stress-weight=stress_auto --max-iter=iteration_auto start.mtp training.cfg > training.log
"""
        self.load_string(file_content)

    def activate_postprocessing(self, file_content=None):
        if file_content is None:
            file_content = """\
$MLP_COMMAND_PARALLEL train --energy-weight=energy_auto --force-weight=force_auto --stress-weight=stress_auto --max-iter=iteration_auto start.mtp training.cfg > training.log
$MLP_COMMAND_SERIAL calc-grade Trained.mtp_ training.cfg testing.cfg grades.cfg --mvs-filename=state.mvs > grading.log
$MLP_COMMAND_SERIAL calc-errors Trained.mtp_ training.cfg > training.errors
$MLP_COMMAND_SERIAL calc-errors Trained.mtp_ testing.cfg > testing.errors
$MLP_COMMAND_SERIAL calc-efs Trained.mtp_ training.cfg training_efs.cfg
$MLP_COMMAND_SERIAL calc-efs Trained.mtp_ testing.cfg testing_efs.cfg
$MLP_COMMAND_SERIAL select-add Trained.mtp_ training.cfg testing.cfg diff.cfg > select.log
cp training.cfg training_new.cfg 
cat diff.cfg >> training_new.cfg
"""
        self.load_string(file_content)


def write_cfg(
    file_name,
    indices_lst,
    position_lst,
    cell_lst,
    forces_lst=None,
    energy_lst=None,
    track_lst=None,
    stress_lst=None,
):
    if stress_lst is None or len(stress_lst) == 0:
        stress_lst = [None] * len(position_lst)
    if forces_lst is None or len(forces_lst) == 0:
        forces_lst = [None] * len(position_lst)
    if track_lst is None or len(track_lst) == 0:
        track_lst = [None] * len(position_lst)
    if energy_lst is None or len(energy_lst) == 0:
        energy_lst = [None] * len(position_lst)
    cfg_lst = []
    for indices, position, forces, cell, energy, stress, track_str in zip(
        indices_lst,
        position_lst,
        forces_lst,
        cell_lst,
        energy_lst,
        stress_lst,
        track_lst,
    ):
        cfg_object = Cfg()
        cfg_object.pos = position
        cfg_object.lat = cell
        cfg_object.types = indices
        cfg_object.energy = energy
        cfg_object.forces = forces
        cfg_object.stresses = stress
        if track_str is not None:
            cfg_object.desc = "pyiron\t" + track_str
        cfg_lst.append(cfg_object)
    savecfgs(filename=file_name, cfgs=cfg_lst, desc=None)
