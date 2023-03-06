"""Generic Input Base Clases"""

import abc
import sys

from pyiron_base.interfaces.object import HasStorage
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure

import numpy as np
import matplotlib.pyplot as plt

class StorageAttribute:
    """
    Create an attribute that is synced to a storage attribute.

    It must be created on a class that derives from :class:`.HasStorage`.  It essentially behaves like a property that
    writes and reads values to the underlying :attr:`.HasStorage.storage`. DataContainer of the class.  When accessing
    this property before setting it, `None` is returned.

    It's possible to modify the default value and the accepted type of values by using the builder-style :meth:`.type`
    and :meth:`.default` methods.

    >>> class MyType(HasStorage):
    ...     a = StorageAttribute()
    ...     b = StorageAttribute().type(int)
    ...     c = StorageAttribute().default(list)
    ...     d = StorageAttribute().default(lambda: 42).type(int)

    >>> t = MyType()
    >>> t.a # return None
    >>> t.b = 3
    >>> t.b
    3
    >>> t.b = 'asdf'
    TypeError(f"{value} is not of type {self.value_type}")
    >>> t.c
    []
    >>> t.d
    42
    """

    def __init__(self):
        self.name = None
        self.value_type = object
        self.default_constructor = None

    def __set_name__(self, obj, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        try:
            return obj.storage[self.name]
        except KeyError:
            if self.default_constructor is not None:
                ret = obj.storage[self.name] = self.default_constructor()
                return ret
            else:
                return None

    def __set__(self, obj, value):
        if isinstance(value, self.value_type):
            obj.storage[self.name] = value
        else:
            raise TypeError(f"{value} is not of type {self.value_type}")

    def type(self, value_type):
        """
        Set a type to type check values set on the attribute.

        Args:
            value_type (type, tuple of type): a class or list of classes that are acceptable for the attribute

        Returns:
            self: the object it is called on
        """
        self.value_type = value_type
        return self

    def default(self, default_constructor):
        """
        Set a function to create a default value, if the attribute is accessed without being set.

        Args:
            default_constructor (function, type): Either a class or a function that takes no arguments

        Returns:
            self: the object it is called on
        """
        self.default_constructor = default_constructor
        return self

    def doc(self, text):
        self.__doc__ = text

class AbstractContainer(HasStorage, abc.ABC):
    # TODO: this should go into HasStorage, exists here only to give one location to define from_ methods
    @classmethod
    def from_attributes(cls, name, *attrs, module=None, **default_attrs):
        """
        Create a new sub class with given attributes.

        Args:
            name (str): name of the new class
            *attrs (str): names of the new attributes
            module (str, optional): the module path where this class is defined; on CPython this is automagically filled
                    in, in other python implementations you need to manually provide this value as __name__ when you
                    call this method for the resulting class to be picklable.
            **default_attrs (str): names and defaults of new attributes
        """
        body = {a: StorageAttribute() for a in attrs}
        body.update({a: StorageAttribute().default(d) for a, d in default_attrs.items()})
        T = type(name, (cls,), body)
        if module is None:
            # this is also how cpython does it for namedtuple
            module = sys._getframe(1).f_globals['__name__']
        T.__module__ = module
        return T

    def transfer(self, other):
        """
        Copy the contents of another 
        """
        if isinstance(self, other.__class__):
            self.storage.update(other.storage)
        else:
            raise TypeError("Must pass a superclass to transfer from!")


class AbstractInput(AbstractContainer, abc.ABC):
    def check_ready(self):
        return True

StructureInput = AbstractInput.from_attributes("StructureInput", "structure")

MDInput = AbstractInput.from_attributes(
        "MDInput",
        "steps",
        "timestep",
        "temperature",
        "output_steps",
)


class AbstractOutput(AbstractContainer, abc.ABC):
    pass

EnergyOutput = AbstractOutput.from_attributes(
        "EnergyOutput",
        "energy_pot",
)

MDOutputBase = AbstractOutput.from_attributes(
        "MDOutputBase",
        pot_energies=list,
        kin_energies=list,
        forces=list,
        structures=list,
)

class MDOutput(HasStructure, MDOutputBase, EnergyOutput):

    def plot_energies(self):
        plt.plot(self.pot_energies - np.min(self.pot_energies), label='pot')
        plt.plot(self.kin_energies, label='kin')
        plt.legend()

    def _number_of_structures(self):
        return len(self.structures)

    def _get_structure(self, frame, wrap_atoms=True):
        return self.structures[frame]

    # TODO: how to make sure this is generally conforming?
    @property
    def energy_pot(self):
        return self.pot_energies[-1]
