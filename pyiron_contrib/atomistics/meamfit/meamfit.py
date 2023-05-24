# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

import numpy as np
import os
import pandas as pd
import posixpath
import shutil
import random
from pyiron_base import GenericParameters, GenericJob, DataContainer

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


class MeamFit(GenericJob):
    def __init__(self, project, job_name):
        """
        Class to setup and run and MeamFit simulations.

        Examples:
            Here is a simple example to setup and run a MeamFit job:

            >>> pr = Project('meamfit')
            >>> job = pr.create.job.MeamFit(job_name='test_job')
            >>> job.add_job_to_fitting(job_id=job_id) # job_id of the vasp MD job using for potential fitting
            >>> job.run()

        Args:
            project: Project object (defines path where job will be created and stored)
            job_name: name of the job (must be unique within this project path)

        """
        super(MeamFit, self).__init__(project, job_name)
        self.__version__ = None
        self.__name__ = "MeamFit"
        self.input = DataContainer(
            {
                "TYPE": "EAM",
                "SEED": "random",
                "CUTOFF_MAX": 5.0,
                "NTERMS": 3,
                "NTERMS_EMB": 3,
            },
            table_name="parameter",
        )
        self._executable_activate()
        self._potential_performance_dataframe = pd.DataFrame({})
        self._potential_timings_dataframe = pd.DataFrame({})
        self._calculation_dataframe = pd.DataFrame(
            {
                "Job id": [],
                "Start config": [],
                "End config": [],
                "Step": [],
                "Quantity to fit": [],
                "Weights": [],
            }
        )
        self._calculation_dataframe = self._calculation_dataframe.set_index("Job id")

    @property
    def calculation_dataframe(self):
        return self._calculation_dataframe

    @calculation_dataframe.setter
    def calculation_dataframe(self, df):
        self._calculation_dataframe = df

    @property
    def potential_performance_dataframe(self):
        return self._potential_performance_dataframe

    @property
    def potential_timings_dataframe(self):
        return self._potential_timings_dataframe

    @property
    def potentials(self):
        return list(self._potential_timings_dataframe.index)

    @property
    def potential_paths(self):
        return [posixpath.join(self.working_directory, pot) for pot in self.potentials]

    @property
    def random_seed(self):
        incar = self.input["SEED"]
        if incar == "random":
            self.input["SEED"] = random.randint(0, 100000)
            incar = self.input["SEED"]
        return incar

    @random_seed.setter
    def random_seed(self, seed):
        self.input["SEED"] = seed

    @property
    def publication(self):
        return {
            "MeamFit": [
                {
                    "title": "MEAMfit: A reference-free modified embedded atom method (RF-MEAM) energy and force-fitting code",
                    "journal": "Computer Physics Communications",
                    "volume": "196",
                    "year": "2015",
                    "doi": "10.1016/j.cpc.2015.05.016",
                    "url": "https://doi.org/10.1016/j.cpc.2015.05.016",
                    "author": [
                        "Andrew Ian Duff",
                        "M.W. Finnis",
                        "Philippe Maugis",
                        "Barend J. Thijsse",
                        "Marcel H.F. Sluiter",
                    ],
                },
            ]
        }

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        self.input.read_only = True

    def validate_ready_to_run(self):
        if self._calculation_dataframe.empty:
            raise ValueError("No training data added yet!")

    def write_input(self):
        """
        Call routines that generate the codespecifc input files

        Returns:

        """
        if self.input["SEED"] == "random":
            self.input["SEED"] = random.randint(0, 100000)
        self.input.write_file(file_name="settings", cwd=self.working_directory)
        self._write_calc_db(
            calculation_dataframe=self._calculation_dataframe,
            file_name="fitdbse",
            cwd=self.working_directory,
        )
        self._copy_vasprun_xml(cwd=self.working_directory)

    def add_job_to_fitting(
        self,
        job_id,
        time_step_start=0,
        time_step_end=-1,
        time_step_delta=10,
        quantity="E0",
        weight=[1.0, 0.0, 0.0],
    ):
        """
        Add output of VASP jobs to training data.

        Args:
            job_id (int): job_id of the vasp MD job you want to use for potential fitting.
            time_step_start (int): initial timestep - after equilibration
            time_step_end (int): last timestep to use
            time_step_delta (int): time step
            quantity (str): the property in the vasprun file to be fit to, and can take the values ['E0', 'Free‐Energy', 'Force']
                            ‘E0’: to the total energy (specifically the E0, sigma‐>0 value in the vasprun file);
                            ‘Free‐Energy’: will fit to the free energy (the F value in the vasprun file);
                            ‘Force’: will fit to atomic forces.
                             Note that only the first two letters are in fact read in by MEAMfit, so that it is sufficient to
                             write ‘Fr’ or ‘Fo’ for the second and third cases respectively.
            weight (list): default is [1.0, 0.0, 0.0].

        Raises:
            ValueError: if given job is a not a Vasp job
        """
        job = self.project.load(job_id)
        if job.__name__ != "Vasp":
            raise ValueError("Training data must be from VASP jobs!")
        if time_step_end == -1:
            time_step_end = (
                np.shape(self.project.inspect(int(job_id))["output/generic/cells"])[0]
                - 1
            )
        if int(job_id) in [int(job) for job in self._calculation_dataframe.index]:
            self._calculation_dataframe.loc[job_id] = pd.Series(
                {
                    "Start config": time_step_start,
                    "End config": time_step_end,
                    "Step": time_step_delta,
                    "Quantity to fit": quantity,
                    "Weights": weight,
                }
            )
        else:
            df_tmp = pd.DataFrame(
                {
                    "Job id": [int(job_id)],
                    "Start config": [time_step_start],
                    "End config": [time_step_end],
                    "Step": [time_step_delta],
                    "Quantity to fit": [quantity],
                    "Weights": [weight],
                }
            )
            self._calculation_dataframe = pd.concat(
                [self._calculation_dataframe, df_tmp.set_index("Job id")]
            )

    def _copy_vasprun_xml(self, cwd=None):
        for job_id in self._calculation_dataframe.index:
            job = self.project.load(job_id)
            job.decompress()
            working_directory = self.project.get_job_working_directory(int(job_id))
            shutil.copyfile(
                posixpath.join(working_directory, "vasprun.xml"),
                posixpath.join(cwd, "vasprun_" + str(int(job_id)) + ".xml"),
            )
            job.compress()

    @staticmethod
    def _write_calc_db(calculation_dataframe, file_name="fitdbse", cwd=None):
        if cwd is not None:
            file_name = posixpath.join(cwd, file_name)
        with open(file_name, "w") as f:
            f.write(
                str(len(calculation_dataframe))
                + " # Files | Configs to fit | Quantity to fit | Weights \n"
            )
            for entry in zip(
                [
                    "vasprun_" + str(int(job_id)) + ".xml"
                    for job_id in calculation_dataframe.index
                ],
                [start + 1 for start in calculation_dataframe["Start config"]],
                [end + 1 for end in calculation_dataframe["End config"]],
                list(calculation_dataframe["Step"]),
                list(calculation_dataframe["Quantity to fit"]),
                list(calculation_dataframe["Weights"]),
            ):
                file, start_config, end_config, step, quantity, weight = entry
                if isinstance(weight, list):
                    weight = (
                        str(weight[0]) + " " + str(weight[1]) + " " + str(weight[2])
                    )
                f.write(
                    str(file)
                    + " "
                    + str(int(start_config))
                    + "-"
                    + str(int(end_config))
                    + "s"
                    + str(int(step))
                    + " "
                    + str(quantity)
                    + " "
                    + str(weight)
                    + "\n"
                )

    # define routines that collect all output files
    def collect_output(self):
        potential_timings_df = self._collect_timings(
            file_name="bestoptfuncs", cwd=self.working_directory
        )
        self._potential_performance_dataframe = self._collect_potential_performance(
            cwd=self.working_directory
        )
        self._potential_timings_dataframe = self._calculate_std(
            potential_timings_df, self._potential_performance_dataframe
        )
        self.to_hdf()

    def collect_logfiles(self):
        pass

    def from_directory(self, directory):
        """
            Collect input and output of a finished MeamFit job from a directory and convert into pyiron hdf file.
        Args:
            directory: working directory of the job.

        """
        if not self.status.finished:
            self.status.collect = True
            self._import_directory = directory
            self.input.read_input(posixpath.join(directory, "settings"))
            self._calculation_dataframe = self._read_calc_db(
                file_name="fitdbse", cwd=directory
            )
            self.collect_output()
            self.status.finished = True
        else:
            raise RuntimeError(
                "Unable to import MEAMfit calculation into finished job. Needs to be `initialized`."
            )

    # define hdf5 input and output
    def to_hdf(self, hdf=None, group_name=None):
        super(MeamFit, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.to_hdf(hdf5_input)
            hdf5_input[
                "calculation"
            ] = self._calculation_dataframe.reset_index().to_dict(orient="list")
        with self.project_hdf5.open("output") as hdf5_output:
            hdf5_output["performance"] = self._potential_performance_dataframe.to_dict(
                orient="list"
            )
            hdf5_output[
                "timings"
            ] = self._potential_timings_dataframe.reset_index().to_dict(orient="list")

    def from_hdf(self, hdf=None, group_name=None):
        super(MeamFit, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            # backwards compatible loading of input
            if hdf5_input["parameter/NAME"] == "GenericParameters":
                gp = GenericParameters()
                gp.from_hdf(hdf5_input["parameter"])
                self.input["TYPE"] = gp["TYPE"]
                self.input["SEED"] = gp["SEED"]
                self.input["CUTOFF_MAX"] = gp["CUTOFF_MAX"]
                self.input["NTERMS"] = gp["NTERMS"]
                self.input["NTERMS_EMB"] = gp["NTERMS_EMB"]
            else:
                self.input.from_hdf(hdf5_input)
            self.input.from_hdf(hdf5_input)
            self._calculation_dataframe = pd.DataFrame(
                hdf5_input["calculation"]
            ).set_index("Job id")
        with self.project_hdf5.open("output") as hdf5_output:
            self._potential_performance_dataframe = pd.DataFrame(
                hdf5_output["performance"]
            )
            self._potential_timings_dataframe = pd.DataFrame(hdf5_output["timings"])
            if "File" in self._potential_timings_dataframe.columns:
                self._potential_timings_dataframe = (
                    self._potential_timings_dataframe.set_index("File")
                )

    @staticmethod
    def _read_calc_db(file_name="fitdbse", cwd=None):
        if cwd is not None:
            file_name = posixpath.join(cwd, file_name)
        with open(file_name, "r") as f:
            content = f.readlines()
        names_lst, range_lst, quantity_lst, weight_lst = zip(
            *[[part for part in line.split(" ") if part != ""] for line in content[1:]]
        )
        start_config_lst, end_config_lst = zip(
            *[[int(step) for step in step_range.split("-")] for step_range in range_lst]
        )
        df = pd.DataFrame(
            {
                "Files": names_lst,
                "Start config": start_config_lst,
                "End config": end_config_lst,
                "Quantity to fit": quantity_lst,
                "Weights": [float(weight.split("\n")[0]) for weight in weight_lst],
            }
        )
        return df

    @staticmethod
    def _collect_timings(file_name="bestoptfuncs", cwd=None):
        if cwd is not None:
            file_name = posixpath.join(cwd, file_name)
        with open(file_name, "r") as f:
            content = f.readlines()
        content_table_lst = [line.split() for line in content[1:-2]]
        content_table_re_lst = list(zip(*content_table_lst))
        file_name_lst = [file for file in os.listdir(cwd) if "alloy_" in file]
        file_name_lst.sort(key=lambda x: int(x.split("_")[-1]))
        df = pd.DataFrame(
            {
                "File": file_name_lst,
                "Precision": [
                    float(number.replace("D", "E"))
                    for number in content_table_re_lst[1]
                ],
                "Time [h]": [int(number) for number in content_table_re_lst[3]],
                "Time [min]": [int(number) for number in content_table_re_lst[5]],
            }
        )
        return df.set_index("File")

    @staticmethod
    def _collect_potential_performance(cwd=None):
        df = pd.DataFrame(
            {"Potential": [], "Structure": [], "fitdata": [], "truedata": []}
        )
        files_in_cwd_lst = sorted(os.listdir(cwd))
        for file in files_in_cwd_lst:
            if "datapnts_best" in file:
                with open(posixpath.join(cwd, file), "r") as f:
                    content = f.readlines()
                data_points_lst = list(zip(*[line.split() for line in content[6:]]))
                df_new = pd.DataFrame(
                    {
                        "Potential": [
                            pot_file
                            for pot_file in files_in_cwd_lst
                            if "alloy_" + str(file).split("_best")[-1]
                            == pot_file.split(".")[-1]
                        ]
                        * len(data_points_lst[0]),
                        "Structure": data_points_lst[0],
                        "fitdata": [float(number) for number in data_points_lst[1]],
                        "truedata": [float(number) for number in data_points_lst[2]],
                    }
                )
                df = pd.concat([df, df_new])
        return df

    @staticmethod
    def _calculate_std(potential_timings_df, potential_performance_df):
        if isinstance(potential_timings_df, pd.DataFrame):
            index_lst = potential_timings_df.index.values
        else:
            raise TypeError("Unsupported type for potential_lst.")
        std_lst = [
            np.std(
                potential_performance_df[potential_performance_df["Potential"] == ind][
                    "fitdata"
                ].values
                - potential_performance_df[
                    potential_performance_df["Potential"] == ind
                ]["truedata"].values
            )
            for ind in index_lst
        ]
        if "Std" in potential_timings_df.columns:
            potential_timings_df = potential_timings_df.drop(columns=["Std"], axis=1)
        std_df = pd.DataFrame({"Std": std_lst})
        std_df = std_df.set_index(index_lst)
        return pd.concat([potential_timings_df, std_df], axis=1)
