from .container import (
            AbstractInput,
            StructureInput,
            MDInput,
            EnergyOutput,
            MDOutput
)
from . import AbstractNode

import numpy as np
import matplotlib.pyplot as plt

from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units


AseInput = AbstractInput.from_attributes('AseInput', 'calculator')


class AseStaticInput(AseInput, StructureInput):
    pass


class AseStaticNode(AbstractNode):

    def _get_input(self):
        return AseStaticInput()

    def _get_output(self):
        return EnergyOutput()

    def _execute(self):
        structure = self.input.structure
        structure.calc = self.input.calculator
        self.output.energy_pot = structure.get_potential_energy()


class AseMDInput(AseInput, MDInput):
    pass

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
