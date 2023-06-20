from pyiron_contrib.tinybase.container import (
    AbstractOutput,
    StructureInput,
    StorageAttribute,
)
from pyiron_contrib.tinybase.task import (
    ListTaskGenerator,
    ListInput,
    ReturnStatus,
)

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.optimize as so

from pyiron_atomistics.atomistics.structure.has_structure import HasStructure


class MurnaghanInput(StructureInput, ListInput):
    strains = StorageAttribute()
    task = StorageAttribute()

    def check_ready(self):
        structure_ready = self.structure is not None
        strain_ready = len(self.strains) > 0
        task = self.task
        task.input.structure = self.structure
        return structure_ready and strain_ready and task.input.check_ready()

    def set_strain_range(self, volume_range, steps):
        self.strains = (1 + np.linspace(-volume_range, volume_range, steps)) ** (1 / 3)

    def _create_tasks(self):
        cell = self.structure.get_cell()
        tasks = []
        for s in self.strains:
            n = deepcopy(self.task)
            n.input.structure = self.structure.copy()
            n.input.structure.set_cell(cell * s, scale_atoms=True)
            tasks.append(n)
        return tasks


class MurnaghanOutput(AbstractOutput, HasStructure):
    base_structure = StorageAttribute()
    volumes = StorageAttribute().type(np.ndarray)
    energies = StorageAttribute().type(np.ndarray)

    def plot(self, per_atom=True):
        N = len(self.base_structure) if per_atom else 1
        plt.plot(self.volumes / N, self.energies / N)

    @property
    def equilibrium_volume(self):
        inter = si.interp1d(self.volumes, self.energies)
        return so.minimize_scalar(
            inter, bounds=(np.min(self.volumes), np.max(self.volumes))
        ).x

    def _number_of_structures(self):
        return 1

    def _get_structure(self, frame, wrap_atoms=True):
        s = self.base_structure
        s.set_cell(s.get_cell() * (self.equilibrium_volume / s.get_volume()) ** (1 / 3))
        return s


class MurnaghanTask(ListTaskGenerator):
    def _get_input(self):
        return MurnaghanInput()

    def _get_output(self):
        out = MurnaghanOutput()
        out.energies = np.zeros(len(self.input.strains))
        out.volumes = np.zeros(len(self.input.strains))
        out.base_structure = self.input.structure
        return out

    def _extract_output(self, output, step, task, ret, task_output):
        if ret.is_done():
            output.energies[step] = task_output.energy_pot
            output.volumes[step] = task.input.structure.get_volume()
