# coding: utf-8
# Copyright (c) ICAMS, Ruhr University Bochum, 2022

## Executable required: $pyiron/resources/pacemaker/bin/run_pacemaker_tf_cpu.sh AND  run_pacemaker_tf.sh


import logging
import numpy as np
import os
import pandas as pd
import re
import ruamel.yaml as yaml

from shutil import copyfile

from pyiron_base import GenericJob, GenericParameters, state, Executable, FlattenedStorage

from pyiron_contrib.atomistics.atomistics.job.trainingcontainer import TrainingStorage, TrainingContainer
from pyiron_contrib.atomistics.ml.potentialfit import PotentialFit

s = state.settings

# set loggers
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)


#TODO: maybe need better name
class Pacemaker2022(GenericJob, PotentialFit):

    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.__name__ = "Pacemaker2022"
        self.__version__ = "0.2"

        self._job_dict = {}

        self.input = GenericParameters(table_name="input")
        self.input['cutoff'] = 7.
        self.input['metadata'] = {'comment': 'pyiron-generated fitting job'}
        self.input['data'] = {}  # data_config
        self.input['potential'] = {}  # potential_config
        self.input['fit'] = {}  # fit_config
        self.input['backend'] = {'evaluator': 'tensorpot'}  # backend_config

        self.structure_data = None

        # self.executable = "pacemaker input.yaml -l log.txt"
        self._executable = None
        self._executable_activate()

        state.publications.add(self.publication)

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
                    "author": ["Anton Bochkarev", "Yury Lysogorskiy", "Sarath Menon", "Minaam Qamar", "Matous Mrovec",
                               "Ralf Drautz"],
                },

                {
                    "title": "Performant implementation of the atomic cluster expansion (PACE) and application to copper and silicon",
                    "journal": "npj Computational Materials",
                    "volume": "7",
                    "number": "1",
                    "year": "2021",
                    "doi": "10.1038/s41524-021-00559-9",
                    "url": "https://doi.org/10.1038/s41524-021-00559-9",
                    "author": ["Yury Lysogorskiy", "Cas van der Oord", "Anton Bochkarev", "Sarath Menon",
                               "Matteo Rinaldi",
                               "Thomas Hammerschmidt", "Matous Mrovec", "Aidan Thompson", "Gábor Csányi",
                               "Christoph Ortner",
                               "Ralf Drautz"],
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

    # TODO: rewrite?
    def _save_structure_dataframe_pckl_gzip(self, df):
        df.rename(columns={"number_of_atoms": "NUMBER_OF_ATOMS",
                           "energy": "energy_corrected",
                           "atoms": "ase_atoms"}, inplace=True)
        df["NUMBER_OF_ATOMS"] = df["NUMBER_OF_ATOMS"].astype(int)
        if "pbc" not in df.columns:
            df["pbc"] = df["ase_atoms"].map(lambda atoms: np.all(atoms.pbc))

        data_file_name = os.path.join(self.working_directory, "df_fit.pckl.gzip")
        logging.info("Saving training structures dataframe into {} with pickle protocol = 4, compression = gzip".format(
            data_file_name))
        df.to_pickle(data_file_name, compression="gzip", protocol=4)
        return data_file_name

    def write_input(self):
        # prepare datafile
        if self.structure_data is None:
            raise ValueError(
                "`structure_data` is none, but should be pd.DataFrame, TrainingContainer or valid pickle.gzip filename")
        if isinstance(self.structure_data, pd.DataFrame):
            logging.info("structure_data is pandas.DataFrame")
            data_file_name = self._save_structure_dataframe_pckl_gzip(self.structure_data)
            self.input["data"] = {"filename": data_file_name}
        elif isinstance(self.structure_data, str):  # filename
            if os.path.isfile(self.structure_data):
                logging.info("structure_data is valid file path")
                self.input["data"] = {"filename": self.structure_data}
            else:
                raise ValueError("Provided structure_data filename ({}) doesn't exists".format(self.structure_data))
        elif hasattr(self.structure_data, "get_pandas"):  # duck-typing check for TrainingContainer
            logging.info("structure_data is TrainingContainer")
            df = self.structure_data.to_pandas()
            data_file_name = self._save_structure_dataframe_pckl_gzip(df)
            self.input["data"] = {"filename": data_file_name}
        elif self._training_ids:
            logging.info("structure_data is from another pyiron jobs")

        metadata_dict = self.input["metadata"]
        metadata_dict["pyiron_job_id"] = str(self.job_id)

        input_yaml_dict = {
            "cutoff": self.input["cutoff"],
            "metadata": metadata_dict,
            'potential': self.input['potential'],
            'data': self.input["data"],
            'fit': self.input["fit"],
            'backend': self.input["backend"],
        }

        if isinstance(self.input["potential"], str):
            pot_file_name = self.input["potential"]
            if os.path.isfile(pot_file_name):
                logging.info("Input potential is filename")
                pot_basename = os.path.basename(pot_file_name)
                copyfile(pot_file_name, os.path.join(self.working_directory, pot_basename))
                input_yaml_dict['potential'] = pot_basename
            else:
                raise ValueError("Provided potential filename ({}) doesn't exists".format(self.input["potential"]))

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

        final_potential_filename_yace = self.get_final_potential_filename_ace()
        # os.system("pace_yaml2yace {}".format(final_potential_filename_yaml))

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

    def get_lammps_potential(self):
        elements_name = self["output/potential/elements_name"]
        elem = " ".join(elements_name)
        pot_file_name = self.get_final_potential_filename_ace()
        pot_dict = {
            'Config': [["pair_style pace\n", "pair_coeff  * * {} {}\n".format(pot_file_name, elem)]],
            'Filename': [""],
            'Model': ["ACE"],
            'Name': [self.job_name],
            'Species': [elements_name]
        }

        ace_potential = pd.DataFrame(pot_dict)

        return ace_potential

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(
            hdf=hdf,
            group_name=group_name
        )
        with self.project_hdf5.open("input") as h5in:
            self.input.to_hdf(h5in)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(
            hdf=hdf,
            group_name=group_name
        )
        with self.project_hdf5.open("input") as h5in:
            self.input.from_hdf(h5in)

    def get_final_potential_filename(self):
        return os.path.join(self.working_directory, "output_potential.yaml")

    def get_final_potential_filename_ace(self):
        return os.path.join(self.working_directory, "output_potential.yace")

    def get_current_potential_filename(self):
        return os.path.join(self.working_directory, "interim_potential_0.yaml")

    # To link to the executable from the notebook
    def _executable_activate(self, enforce=False):
        if self._executable is None or enforce:
            self._executable = Executable(
                codename="pacemaker", module="pacemaker", path_binary_codes=state.settings.resource_paths
            )

    def _add_training_data(self, container: TrainingContainer) -> None:
        self.add_job_to_fitting(container.id, 0, container.number_of_structures - 1, 1)

    def add_job_to_fitting(self, job_id, time_step_start=0, time_step_end=-1, time_step_delta=10):
        if time_step_end == -1:
            time_step_end = np.shape(self.project.inspect(int(job_id))['output/generic/cells'])[0] - 1
        self._job_dict[job_id] = {'time_step_start': time_step_start,
                                  'time_step_end': time_step_end,
                                  'time_step_delta': time_step_delta}

    def _get_training_data(self) -> TrainingStorage:
        raise NotImplementedError()

    def _get_predicted_data(self) -> FlattenedStorage:
        raise NotImplementedError()
