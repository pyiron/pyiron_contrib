# coding: utf-8
# Copyright (c) ICAMS, Ruhr University Bochum, 2022

## Executable required: $pyiron/resources/pacemaker/bin/run_pacemaker_tf_cpu.sh AND  run_pacemaker_tf.sh

import logging
from typing import List

import numpy as np
import os
import pandas as pd
import ruamel.yaml as yaml

from shutil import copyfile

from pyiron_base import (
    GenericJob,
    GenericParameters,
    state,
    Executable,
    FlattenedStorage,
)

from pyiron_contrib.atomistics.atomistics.job.trainingcontainer import (
    TrainingStorage,
    TrainingContainer,
)
from pyiron_contrib.atomistics.ml.potentialfit import PotentialFit

from pyiron_atomistics.atomistics.structure.atoms import (
    Atoms as pyironAtoms,
    ase_to_pyiron,
)
from ase.atoms import Atoms as aseAtoms

s = state.settings


class PacemakerJob(GenericJob, PotentialFit):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.__name__ = "Pacemaker2022"
        self.__version__ = "0.2"

        self._train_job_id_list = []

        self.input = GenericParameters(table_name="input")
        self._cutoff = 7.0
        self.input["cutoff"] = self._cutoff
        self.input["metadata"] = {"comment": "pyiron-generated fitting job"}

        # data_config
        self.input["data"] = {}
        # potential_config
        self.input["potential"] = {
            "elements": [],
            "bonds": {
                "ALL": {
                    "radbase": "SBessel",
                    "rcut": self._cutoff,
                    "dcut": 0.01,
                    "radparameters": [5.25],
                }
            },
            "embeddings": {
                "ALL": {
                    "fs_parameters": [1, 1, 1, 0.5],
                    "ndensity": 2,
                    "npot": "FinnisSinclairShiftedScaled",
                }
            },
            "functions": {
                "ALL": {
                    "nradmax_by_orders": [15, 3, 2, 1],
                    "lmax_by_orders": [0, 3, 2, 1],
                }
            },
        }

        # fit_config
        self.input["fit"] = {
            "loss": {
                "L1_coeffs": 1e-8,
                "L2_coeffs": 1e-8,
                "kappa": 0.3,
                "w0_rad": 0,
                "w1_rad": 0,
                "w2_rad": 0,
            },
            "maxiter": 1000,
            "optimizer": "BFGS",
            "fit_cycles": 1,
        }
        self.input["backend"] = {
            "batch_size": 100,
            "display_step": 50,
            "evaluator": "tensorpot",
        }  # backend_config

        self.structure_data = None
        self._executable = None
        self._executable_activate()

        state.publications.add(self.publication)

    @property
    def elements(self):
        return self.input["potential"].get("elements")

    @elements.setter
    def elements(self, val):
        self.input["potential"]["elements"] = val

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, val):
        self._cutoff = val
        self.input["cutoff"] = self._cutoff
        self.input["potential"]["bonds"]["ALL"]["rcut"] = self._cutoff

    @property
    def publication(self):
        return {
            "pacemaker": [
                {
                    "title": "Efficient parametrization of the atomic cluster expansion",
                    "journal": "Physical Review Materials",
                    "volume": "6",
                    "number": "1",
                    "year": "2022",
                    "doi": "10.1103/PhysRevMaterials.6.013804",
                    "url": "https://doi.org/10.1103/PhysRevMaterials.6.013804",
                    "author": [
                        "Anton Bochkarev",
                        "Yury Lysogorskiy",
                        "Sarath Menon",
                        "Minaam Qamar",
                        "Matous Mrovec",
                        "Ralf Drautz",
                    ],
                },
                {
                    "title": "Performant implementation of the atomic cluster expansion (PACE) and application to copper and silicon",
                    "journal": "npj Computational Materials",
                    "volume": "7",
                    "number": "1",
                    "year": "2021",
                    "doi": "10.1038/s41524-021-00559-9",
                    "url": "https://doi.org/10.1038/s41524-021-00559-9",
                    "author": [
                        "Yury Lysogorskiy",
                        "Cas van der Oord",
                        "Anton Bochkarev",
                        "Sarath Menon",
                        "Matteo Rinaldi",
                        "Thomas Hammerschmidt",
                        "Matous Mrovec",
                        "Aidan Thompson",
                        "Gábor Csányi",
                        "Christoph Ortner",
                        "Ralf Drautz",
                    ],
                },
                {
                    "title": "Atomic cluster expansion for accurate and transferable interatomic potentials",
                    "journal": "Physical Review B",
                    "volume": "99",
                    "year": "2019",
                    "doi": "10.1103/PhysRevB.99.014104",
                    "url": "https://doi.org/10.1103/PhysRevB.99.014104",
                    "author": ["Ralf Drautz"],
                },
            ]
        }

    def _save_structure_dataframe_pckl_gzip(self, df):
        if "NUMBER_OF_ATOMS" not in df.columns and "number_of_atoms" in df.columns:
            df.rename(columns={"number_of_atoms": "NUMBER_OF_ATOMS"}, inplace=True)
        df["NUMBER_OF_ATOMS"] = df["NUMBER_OF_ATOMS"].astype(int)

        # TODO: reference energy subtraction ?
        if "energy_corrected" not in df.columns and "energy" in df.columns:
            df.rename(columns={"energy": "energy_corrected"}, inplace=True)

        if "atoms" in df.columns:
            # check if this is pyironAtoms  -> aseAtoms
            at = df.iloc[0]["atoms"]
            if isinstance(at, pyironAtoms):
                df["ase_atoms"] = df["atoms"].map(lambda s: s.to_ase())
                df.drop(columns=["atoms"], inplace=True)
            else:
                assert isinstance(
                    at, aseAtoms
                ), "'atoms' column is not a valid ASE Atoms object"
                df.rename(columns={"atoms": "ase_atom"}, inplace=True)
        elif "ase_atoms" not in df.columns:
            raise ValueError(
                "DataFrame should contain 'atoms' (pyiron Atoms) or 'ase_atoms' (ASE atoms) columns"
            )

        if "stress" in df.columns:
            df.drop(columns=["stress"], inplace=True)

        if "pbc" not in df.columns:
            df["pbc"] = df["ase_atoms"].map(lambda atoms: np.all(atoms.pbc))

        data_file_name = os.path.join(self.working_directory, "df_fit.pckl.gzip")
        logging.info(
            "Saving training structures dataframe into {} with pickle protocol = 4, compression = gzip".format(
                data_file_name
            )
        )
        df.to_pickle(data_file_name, compression="gzip", protocol=4)
        return data_file_name

    def write_input(self):
        # prepare datafile
        if self._train_job_id_list and self.structure_data is None:
            train_df = self.create_training_dataframe(self._train_job_id_list)
            self.structure_data = train_df

        if isinstance(self.structure_data, pd.DataFrame):
            logging.info("structure_data is pandas.DataFrame")
            data_file_name = self._save_structure_dataframe_pckl_gzip(
                self.structure_data
            )
            self.input["data"] = {"filename": data_file_name}
            elements_set = set()
            for at in self.structure_data["ase_atoms"]:
                elements_set.update(at.get_chemical_symbols())
            elements = sorted(elements_set)
            print("Set automatically determined list of elements: {}".format(elements))
            self.elements = elements
        elif isinstance(self.structure_data, str):  # filename
            if os.path.isfile(self.structure_data):
                logging.info("structure_data is valid file path")
                self.input["data"] = {"filename": self.structure_data}
            else:
                raise ValueError(
                    "Provided structure_data filename ({}) doesn't exists".format(
                        self.structure_data
                    )
                )
        elif hasattr(
            self.structure_data, "get_pandas"
        ):  # duck-typing check for TrainingContainer
            logging.info("structure_data is TrainingContainer")
            df = self.structure_data.to_pandas()
            data_file_name = self._save_structure_dataframe_pckl_gzip(df)
            self.input["data"] = {"filename": data_file_name}
        elif self.structure_data is None:
            raise ValueError(
                "`structure_data` is none, but should be pd.DataFrame, TrainingContainer or valid pickle.gzip filename"
            )

        metadata_dict = self.input["metadata"]
        metadata_dict["pyiron_job_id"] = str(self.job_id)

        input_yaml_dict = {
            "cutoff": self.input["cutoff"],
            "metadata": metadata_dict,
            "potential": self.input["potential"],
            "data": self.input["data"],
            "fit": self.input["fit"],
            "backend": self.input["backend"],
        }

        if isinstance(self.input["potential"], str):
            pot_file_name = self.input["potential"]
            if os.path.isfile(pot_file_name):
                logging.info("Input potential is filename")
                pot_basename = os.path.basename(pot_file_name)
                copyfile(
                    pot_file_name, os.path.join(self.working_directory, pot_basename)
                )
                input_yaml_dict["potential"] = pot_basename
            else:
                raise ValueError(
                    "Provided potential filename ({}) doesn't exists".format(
                        self.input["potential"]
                    )
                )

        with open(os.path.join(self.working_directory, "input.yaml"), "w") as f:
            yaml.dump(input_yaml_dict, f)

    def _analyse_log(self, logfile="metrics.txt"):
        metrics_filename = os.path.join(self.working_directory, logfile)

        metrics_df = pd.read_csv(metrics_filename, sep="\s+")
        res_dict = metrics_df.to_dict(orient="list")
        return res_dict

    def collect_output(self):
        final_potential_filename_yaml = self.get_final_potential_filename()
        with open(final_potential_filename_yaml, "r") as f:
            yaml_lines = f.readlines()
        final_potential_yaml_string = "".join(yaml_lines)

        with open(self.get_final_potential_filename_ace(), "r") as f:
            ace_lines = f.readlines()
        final_potential_yace_string = "".join(ace_lines)

        with open(self.get_final_potential_filename_ace(), "r") as f:
            yace_data = yaml.safe_load(f)

        elements_name = yace_data["elements"]

        with self.project_hdf5.open("output/potential") as h5out:
            h5out["yaml"] = final_potential_yaml_string
            h5out["yace"] = final_potential_yace_string
            h5out["elements_name"] = elements_name

        log_res_dict = self._analyse_log()

        with self.project_hdf5.open("output/log") as h5out:
            for key, arr in log_res_dict.items():
                h5out[key] = arr

        # training data
        training_data_fname = os.path.join(
            self.working_directory, "fitting_data_info.pckl.gzip"
        )
        df = pd.read_pickle(training_data_fname, compression="gzip")
        df["atoms"] = df.ase_atoms.map(ase_to_pyiron)
        training_data_ts = TrainingStorage()
        for _, r in df.iterrows():
            training_data_ts.add_structure(
                r.atoms,
                energy=r.energy_corrected,
                forces=r.forces,
                identifier=r["name"],
            )

        # predicted data
        predicted_fname = os.path.join(self.working_directory, "train_pred.pckl.gzip")
        df = pd.read_pickle(predicted_fname, compression="gzip")
        predicted_data_fs = FlattenedStorage()
        predicted_data_fs.add_array("energy", dtype=np.float64, shape=(), per="chunk")
        predicted_data_fs.add_array(
            "energy_true", dtype=np.float64, shape=(), per="chunk"
        )

        predicted_data_fs.add_array(
            "number_of_atoms", dtype=np.int64, shape=(), per="chunk"
        )

        predicted_data_fs.add_array(
            "forces", dtype=np.float64, shape=(3,), per="element"
        )
        predicted_data_fs.add_array(
            "forces_true", dtype=np.float64, shape=(3,), per="element"
        )
        for i, r in df.iterrows():
            identifier = r["name"] if "name" in r else str(i)
            predicted_data_fs.add_chunk(
                r["NUMBER_OF_ATOMS"],
                identifier=identifier,
                energy=r.energy_pred,
                forces=r.forces_pred,
                energy_true=r.energy_corrected,
                forces_true=r.forces,
                number_of_atoms=r.NUMBER_OF_ATOMS,
                energy_per_atom=r.energy_pred / r.NUMBER_OF_ATOMS,
                energy_per_atom_true=r.energy_corrected / r.NUMBER_OF_ATOMS,
            )

        with self.project_hdf5.open("output") as hdf5_output:
            training_data_ts.to_hdf(hdf=hdf5_output, group_name="training_data")
            predicted_data_fs.to_hdf(hdf=hdf5_output, group_name="predicted_data")

    def get_lammps_potential(self):
        elements_name = self["output/potential/elements_name"]
        elem = " ".join(elements_name)
        pot_file_name = self.get_final_potential_filename_ace()
        pot_dict = {
            "Config": [
                [
                    "pair_style pace\n",
                    "pair_coeff  * * {} {}\n".format(pot_file_name, elem),
                ]
            ],
            "Filename": [""],
            "Model": ["ACE"],
            "Name": [self.job_name],
            "Species": [elements_name],
        }

        ace_potential = pd.DataFrame(pot_dict)

        return ace_potential

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as h5in:
            self.input.to_hdf(h5in)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as h5in:
            self.input.from_hdf(h5in)

    def get_final_potential_filename(self):
        return os.path.join(self.working_directory, "output_potential.yaml")

    def get_final_potential_filename_ace(self):
        return os.path.join(self.working_directory, "output_potential.yace")

    def get_current_potential_filename(self):
        return os.path.join(self.working_directory, "interim_potential_0.yaml")

    # To link to the executable from the notebook
    def _executable_activate(self, enforce=False, codename="pacemaker"):
        if self._executable is None or enforce:
            self._executable = Executable(
                codename=codename,
                module="pacemaker",
                path_binary_codes=state.settings.resource_paths,
            )

    def _add_training_data(self, container: TrainingContainer) -> None:
        self.add_job_to_fitting(container.id)

    def add_job_to_fitting(self, job_id, *args, **kwargs):
        self._train_job_id_list.append(job_id)

    def _get_training_data(self) -> TrainingStorage:
        return self["output/training_data"].to_object()

    def _get_predicted_data(self) -> FlattenedStorage:
        return self["output/predicted_data"].to_object()

    # copied/adapted from mlip.py
    def create_training_dataframe(
        self, _train_job_id_list: List = None
    ) -> pd.DataFrame:
        if _train_job_id_list is None:
            _train_job_id_list = self._train_job_id_list
        df_list = []
        for job_id in _train_job_id_list:
            ham = self.project.inspect(job_id)
            if ham.__name__ == "TrainingContainer":
                job = ham.to_object()
                data_df = job.to_pandas()
                df_list.append(data_df)
            else:
                raise NotImplementedError(
                    "Currently only TrainingContainer is supported"
                )

        total_training_df = pd.concat(df_list, axis=0)
        total_training_df.reset_index(drop=True, inplace=True)

        return total_training_df
