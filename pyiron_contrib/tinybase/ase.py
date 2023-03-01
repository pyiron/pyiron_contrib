from . import AbstractInput
from . import StructureInput
from . import AbstractOutput
from . import AbstractNode
from . import ReturnStatus
from . import make_storage_mapping

import numpy as np
import matplotlib.pyplot as plt


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

    def _execute(self):
        structure = self.input.structure
        structure.calc = self.input.calculator
        self.output.energy_pot = structure.get_potential_energy()
