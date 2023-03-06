from .container import (
            AbstractInput,
            StorageAttribute,
            StructureInput,
            MDInput,
            EnergyOutput,
            MDOutput
)
from . import AbstractNode, ReturnStatus

import numpy as np
import matplotlib.pyplot as plt

from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.optimize.lbfgs import LBFGS
from ase.optimize.fire import FIRE
from ase.optimize.gpmin.gpmin import GPMin


AseInput = AbstractInput.from_attributes('AseInput', 'calculator')


class AseStaticInput(AseInput, StructureInput):
    pass


class AseStaticNode(AbstractNode):

    def _get_input(self):
        return AseStaticInput()

    def _get_output(self):
        return EnergyOutput()

    def _execute(self, output):
        structure = self.input.structure
        structure.calc = self.input.calculator
        output.energy_pot = structure.get_potential_energy()


class AseMDInput(AseInput, MDInput):
    pass

class AseMDNode(AbstractNode):

    def _get_input(self):
        return AseMDInput()

    def _get_output(self):
        return MDOutput()

    def _execute(self, output):
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
            output.structures.append(structure.copy())
            output.pot_energies.append(structure.get_potential_energy())
            output.kin_energies.append(structure.get_kinetic_energy())
            output.forces.append(structure.get_forces())

        parse()
        dyn.attach(parse, interval=self.input.steps // self.input.output_steps)
        dyn.run(self.input.steps)

MinimizeInput = AbstractInput.from_attributes(
        'MinimizeInput',
        ionic_force_tolerance=float,
        max_steps=int,
        output_steps=int,
)

class AseMinimizeInput(AseInput, StructureInput, MinimizeInput):

    """My first experimental docstring"""
    algo = StorageAttribute().type(str).default('LBFGS')
    """My second experimental docstring"""
    minimizer_kwargs = StorageAttribute().type(dict).default(dict)

    def lbfgs(self, damping=None, alpha=None):
        self.algo = 'LBFGS'
        if damping is not None:
            self.minimizer_kwargs['damping'] = damping
        if alpha is not None:
            self.minimizer_kwargs['alpha'] = alpha

    def fire(self):
        self.algo = 'FIRE'

    def gpmin(self):
        self.algo = 'GPMIN'

    def get_ase_optimizer(self, structure):
        return {
                'LBFGS': LBFGS,
                'FIRE': FIRE,
                'GPMIN': GPMin
        }[self.algo](structure, **self.minimizer_kwargs)


class AseMinimizeNode(AbstractNode):

    def _get_input(self):
        return AseMinimizeInput()

    def _get_output(self):
        return MDOutput()

    def _execute(self, output):
        structure = self.input.structure.copy()
        structure.calc = self.input.calculator

        opt = self.input.get_ase_optimizer(structure)

        def parse():
            output.structures.append(structure.copy())
            output.pot_energies.append(structure.get_potential_energy())
            output.kin_energies.append(structure.get_kinetic_energy())
            output.forces.append(structure.get_forces())

        opt.attach(parse, interval=self.input.output_steps)
        opt.run(fmax=self.input.ionic_force_tolerance, steps=self.input.max_steps)
        parse()

        max_force = abs(output.forces[-1]).max()
        force_tolerance = self.input.ionic_force_tolerance
        if max_force > force_tolerance:
            return ReturnStatus(
                    "not_converged",
                    f"force in last step ({max_force}) is larger than tolerance ({force_tolerance})!"
            )
