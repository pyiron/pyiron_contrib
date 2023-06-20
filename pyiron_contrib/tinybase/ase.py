from pyiron_contrib.tinybase.container import (
    AbstractInput,
    StorageAttribute,
    StructureInput,
    MDInput,
    MinimizeInput,
    EnergyPotOutput,
    MDOutput,
)
from pyiron_contrib.tinybase.task import AbstractTask, ReturnStatus

from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.optimize.lbfgs import LBFGS
from ase.optimize.fire import FIRE
from ase.optimize.gpmin.gpmin import GPMin


class AseInput(AbstractInput):
    calculator = StorageAttribute()


class AseStaticInput(AseInput, StructureInput):
    pass


class AseStaticTask(AbstractTask):
    def _get_input(self):
        return AseStaticInput()

    def _get_output(self):
        return EnergyPotOutput()

    def _execute(self, output):
        structure = self.input.structure
        structure.calc = self.input.calculator
        output.energy_pot = structure.get_potential_energy()


class AseMDInput(AseInput, MDInput):
    pass


class AseMDTask(AbstractTask):
    def _get_input(self):
        return AseMDInput()

    def _get_output(self):
        return MDOutput()

    def _execute(self, output):
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

        def parse():
            output.structures.append(structure.copy())
            output.pot_energies.append(structure.get_potential_energy())
            output.kin_energies.append(structure.get_kinetic_energy())
            output.forces.append(structure.get_forces())

        parse()
        dyn.attach(parse, interval=self.input.steps // self.input.output_steps)
        dyn.run(self.input.steps)


class AseMinimizeInput(AseInput, StructureInput, MinimizeInput):
    algo = StorageAttribute().type(str).default("LBFGS")
    minimizer_kwargs = StorageAttribute().type(dict).constructor(dict)

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
        return {"LBFGS": LBFGS, "FIRE": FIRE, "GPMIN": GPMin}.get(self.algo)(
            structure, **self.minimizer_kwargs
        )


class AseMinimizeTask(AbstractTask):
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
                f"force in last step ({max_force}) is larger than tolerance ({force_tolerance})!",
            )
