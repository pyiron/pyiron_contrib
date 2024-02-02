from pyiron_contrib.tinybase.container import (
    AbstractInput,
    AbstractOutput,
    StructureInput,
    field,
    USER_REQUIRED,
)

from pyiron_contrib.tinybase.task import (
    AbstractTask,
    ListTaskGenerator,
    ListInput,
    ReturnStatus,
)

from copy import deepcopy

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.optimize as so

from ase import Atoms
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure


class MurnaghanInput(StructureInput, ListInput):
    task: AbstractTask = USER_REQUIRED
    strains: npt.NDArray[float] = field(
        default_factory=lambda: np.linspace(-0.2, 0.2, 7)
    )

    def check_ready(self):
        if not super().check_ready():
            return False
        strain_ready = len(self.strains) > 0
        task = self.task
        task.input.structure = self.structure
        return strain_ready and task.input.check_ready()

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
    base_structure: Atoms
    volumes: npt.NDArray[float]
    energies: npt.NDArray[float]

    def plot(self, per_atom=True):
        N = len(self.base_structure) if per_atom else 1
        plt.plot(self.volumes / N, self.energies / N)

    @property
    def equilibrium_volume(self):
        inter = si.interp1d(self.volumes, self.energies, kind="cubic")
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

    def _extract_output(self, step, task, ret, output):
        if ret.is_done():
            return {
                "step": step,
                "energy_pot": output.energy_pot,
                "volume": task.input.structure.get_volume(),
            }

    def _join_output(self, outputs):
        energies = np.full(self.input.strains.shape, np.nan)
        volumes = np.full(self.input.strains.shape, np.nan)
        for output in outputs:
            energies[output["step"]] = output["energy_pot"]
            volumes[output["step"]] = output["volume"]
        return MurnaghanOutput(
            base_structure=self.input.structure.copy(),
            energies=energies,
            volumes=volumes,
        )
