"""Generic Input Base Clases"""

import abc

from pyiron_base.interfaces.object import HasStorage
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure

import numpy as np
import matplotlib.pyplot as plt

def make_storage_mapping(name, default=lambda: None):
    def fget(self):
        try:
            return self.storage[name]
        except KeyError:
            self.storage[name] = default()
            return self.storage[name]

    def fset(self, value):
        self.storage[name] = value

    return property(fget=fget, fset=fset)

class AbstractContainer(HasStorage, abc.ABC):
    # TODO: this should go into HasStorage, exists here only to give one location to define from_ methods
    @classmethod
    def from_attributes(cls, name, *attrs, module=None, **default_attrs):
        """
        Create a new sub class with given attributes.

        Args:
            name (str): name of the new class
            *attrs (str): names of the new attributes
            module (str, optional): the module path where this class is defined; you cannot pickle instances of classes
            defined by this function, if you do not provide this as __name__
            **default_attrs (str): names and defaults of new attributes
        """
        body = {a: make_storage_mapping(a) for a in attrs}
        body.update({a: make_storage_mapping(a, d) for a, d in default_attrs.items()})
        T = type(name, (cls,), body)
        T.__module__ = module
        return T


class AbstractInput(AbstractContainer, abc.ABC):
    pass

StructureInput = AbstractInput.from_attributes("StructureInput", "structure", module=__name__)

MDInput = AbstractInput.from_attributes(
        "MDInput",
        "steps",
        "timestep",
        "temperature",
        "output_steps",
        module=__name__
)


class AbstractOutput(AbstractContainer, abc.ABC):
    pass

EnergyOutput = AbstractOutput.from_attributes(
        "EnergyOutput",
        "energy_pot",
        module=__name__
)

MDOutputBase = AbstractOutput.from_attributes(
        "MDOutputBase",
        pot_energies=list,
        kin_energies=list,
        forces=list,
        structures=list,
)

class MDOutput(HasStructure, MDOutputBase):

    def plot_energies(self):
        plt.plot(self.pot_energies - np.min(self.pot_energies), label='pot')
        plt.plot(self.kin_energies, label='kin')
        plt.legend()

    def _number_of_structures(self):
        return len(self.structures)

    def _get_structure(self, frame, wrap_atoms=True):
        return self.structures[frame]
