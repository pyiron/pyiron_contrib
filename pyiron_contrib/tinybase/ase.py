from . import AbstractInput
from . import StructureInput
from . import AbstractOutput
from . import AbstractNode
from . import ReturnStatus
from . import make_storage_mapping

import numpy as np
import matplotlib.pyplot as plt

from pyiron_atomistics.atomistics.structure.has_structure import HasStructure

class AseInput(AbstractInput):
    def __init__(self):
        super().__init__()
        self.storage.calculator = None

    calculator = make_storage_mapping('calculator')

class AseStaticInput(AseInput, StructureInput):
    pass

class EnergyOutput(AbstractOutput):
    def __init__(self):
        super().__init__()
        self.storage.energy_pot = None

    energy_pot = make_storage_mapping('energy_pot')

class AseStaticNode(AbstractNode):

    def _get_input(self):
        return AseInput()

    def _get_output(self):
        return EnergyOutput()

    def _execute(self):
        structure = self.input.structure
        structure.calc = self.input.calculator
        self.output.energy_pot = structure.get_potential_energy()


class MDInput(StructureInput):
    def __init__(self):
        super().__init__()
        self.storage.steps = None
        self.storage.timestep = None
        self.storage.temperature = None
        self.storage.output_steps = None

    steps = make_storage_mapping('steps')
    timestep = make_storage_mapping('timestep')
    temperature = make_storage_mapping('temperature')
    output_steps = make_storage_mapping('output_steps')

class MDOutput(AbstractOutput, HasStructure):
    def __init__(self):
        super().__init__()

        self.storage.pot_energies = []
        self.storage.kin_energies = []
        self.storage.forces = []
        self.storage.structures = []

    pot_energies = make_storage_mapping("pot_energies")
    kin_energies = make_storage_mapping("kin_energies")
    forces = make_storage_mapping("forces")
    structures = make_storage_mapping("structures")

    def plot_energies(self):
        plt.plot(self.pot_energies - np.min(self.pot_energies), label='pot')
        plt.plot(self.kin_energies, label='kin')
        plt.legend()

    def _number_of_structures(self):
        return len(self.structures)

    def _get_structure(self, frame, wrap_atoms=True):
        return self.structures[frame]


class AseMDInput(AseInput, MDInput):
    pass

from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

class AseMDNode(AbstractNode):

    def _get_input(self):
        return AseMDInput()

    def _get_output(self):
        return MDOutput()

    def _execute(self):
        structure = self.input.structure.copy()
        structure.calc = self.input.calculator

        MaxwellBoltzmannDistribution(structure, temperature_K=self.input.temperature * 2)

        dyn = Langevin(
                structure,
                timestep=self.input.timestep * units.fs,
                temperature_K=self.input.temperature,
                friction=1e-3,
                append_trajectory=True
        )

        def parse():
            self.output.structures.append(structure.copy())
            self.output.pot_energies.append(structure.get_potential_energy())
            self.output.kin_energies.append(structure.get_kinetic_energy())
            self.output.forces.append(structure.get_forces())

        parse()
        dyn.attach(parse, interval=self.input.steps // self.input.output_steps)
        dyn.run(self.input.steps)
