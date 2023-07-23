# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
from pyiron_atomistics.lammps.base import Input
from pyiron_atomistics.lammps.interactive import LammpsInteractive
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_contrib.atomistics.mlip.cfgs import loadcfgs
from pyiron_base import GenericParameters

import numpy as np

__author__ = "Jan Janssen"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Jan Janssen"
__email__ = "janssen@mpie.de"
__status__ = "development"
__date__ = "Sep 1, 2018"


class LammpsMlip(LammpsInteractive):
    def __init__(self, project, job_name):
        super(LammpsMlip, self).__init__(project, job_name)
        self.input = MlipInput()
        self.__name__ = "LammpsMlip"
        self.__version__ = None
        # Reset the version number to the executable is set automatically
        self._executable = None
        self._executable_activate()
        self._selected_structures = None

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        super(LammpsMlip, self).set_input_to_read_only()
        self.input.mlip.read_only = True

    def write_input(self):
        super(LammpsMlip, self).write_input()
        if self.input.mlip["mtp-filename"] == "auto":
            self.input.mlip["mtp-filename"] = os.path.basename(
                self.potential["Filename"][0][0]
            )
        self.input.mlip.write_file(file_name="mlip.ini", cwd=self.working_directory)

    def convergence_check(self):
        for line in self["error.out"]:
            if line.startswith("MLIP: Breaking threshold exceeded"):
                return False
        return True

    def enable_active_learning(self, threshold=2.0, threshold_break=5.0):
        """
        Enable active learning during MD run.

        Automatically collect structures on which the potential is extrapolating.

        Args:
            threshold (float): select structures with extrapolation grade larger than this
            threshold_break (float): stop the MD run after seeing a structure with extrapolation grade larger than this
        """
        self.executable.accepted_return_codes += [8]
        self.input.mlip.load_string(
            f"""\
mtp-filename auto
calculate-efs TRUE
select TRUE
select:threshold {threshold}
select:threshold-break {threshold_break}
select:save-selected selected.cfg
select:load-state state.mvs
select:log selection.log
write-cfgs:skip 0
"""
        )

    def _get_selection_file(self):
        return os.path.join(
            self.working_directory, self.input.mlip["select:save-selected"]
        )

    @property
    def _selection_enabled(self):
        return self.input.mlip["select"] == "TRUE"

    @property
    def selected_structures(self):
        """
        :class:`.StructureStorage`: structures that the potential extrapolated on during the run.

        Only available if :method:`.enable_active_learning` was called and once the job has been collected.
        """
        if not (
            self.status.collect or self.status.finished or self.status.not_converged
        ):
            raise ValueError(
                "Selected structures are only available once the job has finished!"
            )
        if not self._selection_enabled:
            raise ValueError(
                "Selected structures are only available after calling enable_active_learning()!"
            )
        if self._selected_structures is None:
            if "selected" in self["output"].list_groups():
                self._selected_structures = self["output/selected"].to_object()
            else:
                self._selected_structures = StructureStorage()
        return self._selected_structures

    def collect_output(self):
        super(LammpsMlip, self).collect_output()
        if "select:save-selected" in self.input.mlip._dataset["Parameter"]:
            file_name = self._get_selection_file()
            if os.path.exists(file_name):
                cell = []
                positions = []
                forces = []
                stress = []
                energy = []
                indicies = []
                for cfg in loadcfgs(file_name):
                    cell.append(cfg.lat)
                    positions.append(cfg.pos)
                    forces.append(cfg.forces)
                    stress.append(
                        [
                            [cgf.stresses[0], cgf.stresses[5], cgf.stresses[4]],
                            [cgf.stresses[5], cgf.stresses[1], cgf.stresses[3]],
                            [cgf.stresses[4], cgf.stresses[3], cgf.stresses[2]],
                        ]
                    )
                    energy.append(cfg.energy)
                    indicies.append(cfg.types)

                    species = np.array(self.potential.Species.iloc[0])[
                        cfg.types.astype(int)
                    ]
                    self.selected_structures.add_structure(
                        Atoms(
                            symbols=species,
                            positions=cfg.pos,
                            cell=cfg.lat,
                            pbc=[True, True, True],
                        ),
                        mv_grade=cfg.grade,
                    )
                with self.project_hdf5.open("output/mlip") as hdf5_output:
                    hdf5_output["forces"] = np.array(forces)
                    hdf5_output["energy_tot"] = np.array(energy)
                    hdf5_output["pressures"] = np.array(stress)
                    hdf5_output["cells"] = np.array(cell)
                    hdf5_output["positions"] = np.array(positions)
                    hdf5_output["indicies"] = np.array(indicies)
            self.selected_structures.to_hdf(
                self.project_hdf5.open("output"), "selected"
            )


class MlipInput(Input):
    def __init__(self):
        self.mlip = MlipParameter()
        super(MlipInput, self).__init__()

    def to_hdf(self, hdf5):
        """

        Args:
            hdf5:
        Returns:
        """
        with hdf5.open("input") as hdf5_input:
            self.mlip.to_hdf(hdf5_input)
        super(MlipInput, self).to_hdf(hdf5)

    def from_hdf(self, hdf5):
        """

        Args:
            hdf5:
        Returns:
        """
        with hdf5.open("input") as hdf5_input:
            self.mlip.from_hdf(hdf5_input)
        super(MlipInput, self).from_hdf(hdf5)


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
mtp-filename auto
select FALSE
"""
        self.load_string(file_content)
