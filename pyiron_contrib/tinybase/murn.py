from . import AbstractInput
from . import StructureInput
from . import AbstractOutput
from . import AbstractNode, ListNode
from . import ReturnStatus
from . import make_storage_mapping

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

class MurnaghanInput(StructureInput):
    def __init__(self):
        super().__init__()
        self.storage.strains = None
        self.storage.node = None

    strains = make_storage_mapping('strains')
    node = make_storage_mapping('node')

    def set_strain_range(self, range, steps):
        self.strains = (1 + np.linspace(-range, range, steps))**(1/3)

class MurnaghanOutput(AbstractOutput):
    def __init__(self):
        super().__init__()
        self.storage.volumes = []
        self.storage.energies = []

    volumes = make_storage_mapping('volumes')
    energies = make_storage_mapping('energies')

    def plot(self):
        plt.plot(self.volumes, self.energies)

class MurnaghanNode(AbstractNode):

    def _get_input(self):
        return MurnaghanInput()

    def _get_output(self):
        return MurnaghanOutput()

    def _execute(self):
        cell = self.input.structure.get_cell()
        node = self.input.node
        energy = []
        volume = []
        returns = []
        for s in self.input.strains:
            structure = self.input.structure.copy()
            structure.set_cell(cell * s, scale_atoms=True)
            node.input.structure = structure
            ret = node.execute()
            returns.append(ret)
            if ret.is_done():
                energy.append(node.output.energy_pot)
                volume.append(structure.get_volume())
        self.output.energies = np.array(energy)
        self.output.volumes = np.array(volume)
        errors = [ret for ret in returns if not ret.is_done()]
        if len(errors) == 0:
            return ReturnStatus('done')
        else:
            return ReturnStatus('aborted', msg=errors)

class ListMurnaghanNode(ListNode):

    def _get_input(self):
        return MurnaghanInput()

    def _get_output(self):
        return MurnaghanOutput()

    def _create_nodes(self):
        cell = self.input.structure.get_cell()
        nodes = []
        for s in self.input.strains:
            n = deepcopy(self.input.node)
            n.input.structure = self.input.structure.copy()
            n.input.structure.set_cell(cell * s, scale_atoms=True)
            nodes.append(n)
        return nodes

    def _extract_output(self, i, node, ret):
        if ret.is_done():
            self.output.energies.append(node.output.energy_pot)
            self.output.volumes.append(node.input.structure.get_volume())
