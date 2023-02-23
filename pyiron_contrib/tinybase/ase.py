from . import AbstractInput
from . import StructureInput
from . import AbstractOutput
from . import AbstractNode
from . import ReturnStatus

import numpy as np
import matplotlib.pyplot as plt

def make_storage_mapping(name):
    def fget(self):
        return self.storage[name]

    def fset(self, value):
        self.storage[name] = value

    return property(fget=fget, fset=fset)


class AseInput(StructureInput):
    def __init__(self):
        super().__init__()
        self.storage.calculator = None

    calculator = make_storage_mapping('calculator')

class EnergyOutput(AbstractOutput):
    def __init__(self):
        super().__init__()
        self.storage.energy_pot = None

    energy_pot = make_storage_mapping('energy_pot')

class AseNode(AbstractNode):

    def _get_input(self):
        return AseInput()

    def _get_output(self):
        return EnergyOutput()

    def execute(self):
        structure = self.input.structure
        structure.calc = self.input.calculator
        self.output.energy_pot = structure.get_potential_energy()
        return ReturnStatus("done")

