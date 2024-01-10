from typing import Literal

from pyiron_contrib.tinybase.container import (
    AbstractInput,
    StructureInput,
    MDInput,
    MinimizeInput,
    EnergyPotOutput,
    MDOutput,
    field,
    USER_REQUIRED,
)
from pyiron_contrib.tinybase.task import AbstractTask, ReturnStatus

import numpy as np
from ase.calculators.calculator import Calculator
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.optimize.lbfgs import LBFGS
from ase.optimize.fire import FIRE
from ase.optimize.gpmin.gpmin import GPMin


class AseInput(AbstractInput):
    calculator: Calculator = USER_REQUIRED

    def _store(self, storage):
        # if the calculator was attached to pyiron Atoms object, saving the calculator would fail, since it would be
        # pickled, but our Atoms cannot be pickled. Therefore remove the reference here.  If the task were to be
        # re-executed after being loaded, the atoms would be reattached anyway.
        self.calculator.atoms = None
        super()._store(storage=storage)


class AseStaticInput(AseInput, StructureInput):
    pass


class AseStaticTask(AbstractTask):
    def _get_input(self):
        return AseStaticInput()

    def _execute(self):
        structure = self.input.structure
        structure.calc = self.input.calculator
        return EnergyPotOutput(energy_pot=structure.get_potential_energy())


class AseMDInput(AseInput, StructureInput, MDInput):
    pass


class AseMDTask(AbstractTask):
    def _get_input(self):
        return AseMDInput()

    def _execute(self):
        structure = self.input.structure.copy()
        structure.calc = self.input.calculator

        MaxwellBoltzmannDistribution(
            structure, temperature_K=self.input.temperature * 2
        )

        dyn = Langevin(
            structure,
            timestep=self.input.timestep * units.fs,
            temperature_K=self.input.temperature,
            friction=1e-3,
            append_trajectory=True,
        )

        structures = []
        pot_energies = []
        kin_energies = []
        forces = []

        def parse():
            structures.append(structure.copy())
            pot_energies.append(structure.get_potential_energy())
            kin_energies.append(structure.get_kinetic_energy())
            forces.append(structure.get_forces())

        parse()
        dyn.attach(parse, interval=self.input.steps // self.input.output_steps)
        dyn.run(self.input.steps)

        return MDOutput(
            structures=structures,
            pot_energies=np.array(pot_energies),
            kin_energies=np.array(kin_energies),
            forces=np.array(forces),
        )


_ASE_OPTIMIZER_MAP = {"LBFGS": LBFGS, "FIRE": FIRE, "GPMIN": GPMin}


class AseMinimizeInput(AseInput, StructureInput, MinimizeInput):
    algo: Literal[list(_ASE_OPTIMIZER_MAP.keys())] = "LBFGS"
    minimizer_kwargs: dict = field(default_factory=dict)

    def lbfgs(self, damping=None, alpha=None):
        self.algo = "LBFGS"
        if damping is not None:
            self.minimizer_kwargs["damping"] = damping
        if alpha is not None:
            self.minimizer_kwargs["alpha"] = alpha

    def fire(self):
        self.algo = "FIRE"
        self.minimizer_kwargs = {}

    def gpmin(self):
        self.algo = "GPMIN"
        self.minimizer_kwargs = {}

    def get_ase_optimizer(self, structure):
        return _ASE_OPTIMIZER_MAP.get(self.algo)(structure, **self.minimizer_kwargs)


class AseMinimizeTask(AbstractTask):
    def _get_input(self):
        return AseMinimizeInput()

    def _execute(self):
        structure = self.input.structure.copy()
        structure.calc = self.input.calculator

        opt = self.input.get_ase_optimizer(structure)

        structures = []
        pot_energies = []
        kin_energies = []
        forces = []

        def parse():
            structures.append(structure.copy())
            pot_energies.append(structure.get_potential_energy())
            kin_energies.append(structure.get_kinetic_energy())
            forces.append(structure.get_forces())

        opt.attach(parse, interval=self.input.output_steps)
        opt.run(fmax=self.input.ionic_force_tolerance, steps=self.input.max_steps)
        parse()

        output = MDOutput(
            structures=structures,
            pot_energies=np.array(pot_energies),
            kin_energies=np.array(kin_energies),
            forces=np.array(forces),
        )

        max_force = abs(output.forces[-1]).max()
        force_tolerance = self.input.ionic_force_tolerance
        if max_force > force_tolerance:
            return (
                ReturnStatus(
                    "not_converged",
                    f"force in last step ({max_force}) is larger than tolerance ({force_tolerance})!",
                ),
                output,
            )
        else:
            return output
