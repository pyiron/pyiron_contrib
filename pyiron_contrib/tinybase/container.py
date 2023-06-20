"""Generic Input Base Clases"""

import abc
from copy import deepcopy

from pyiron_contrib.tinybase.storage import HasHDFAdapaterMixin

from pyiron_base.interfaces.object import HasStorage
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure

from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt


class StorageAttribute:
    """
    Create an attribute that is synced to a storage attribute.

    It must be created on a class that derives from :class:`.HasStorage`.  It essentially behaves like a property that
    writes and reads values to the underlying :attr:`.HasStorage.storage`. DataContainer of the class.  When accessing
    this property before setting it, `None` is returned.

    It's possible to modify the default value and the accepted type of values by using the builder-style :meth:`.type`,
    :meth:`.default` and :meth:`.constructor` methods.

    >>> class MyType(HasStorage):
    ...     a = StorageAttribute()
    ...     b = StorageAttribute().type(int)
    ...     c = StorageAttribute().constructor(list)
    ...     d = StorageAttribute().default(42).type(int)

    >>> t = MyType()
    >>> t.a # return None
    >>> t.b = 3
    >>> t.b
    3
    >>> t.b = 'asdf'
    TypeError("'asdf' is not of type <class 'int'>")
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

    def default(self, value):
        """
        Set a default value, if the attribute is accessed without being set.

        The given value is deep copied before being set.  If your type does not
        support this or it is inefficient, use :meth:`.constructor`.

        Args:
            value: any value to be used as default

        Returns:
            self: the object it is called on
        """
        self.default_constructor = lambda: deepcopy(value)
        return self

    def constructor(self, default_constructor):
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
        return self


class AbstractContainer(HasStorage, HasHDFAdapaterMixin, abc.ABC):
    def take(self, other: "AbstractContainer"):
        # TODO: think hard about variance of types
        if not isinstance(self, type(other)):
            raise TypeError("Must pass a superclass to transfer from!")

        mro_iter = {k: v for c in type(other).__mro__ for k, v in c.__dict__.items()}
        for name, attr in mro_iter.items():
            if isinstance(attr, StorageAttribute):
                a = getattr(other, name)
                if a is not None:
                    setattr(self, name, a)

    def put(self, other: "AbstractContainer"):
        other.take(self)


class AbstractInput(AbstractContainer, abc.ABC):
    def check_ready(self):
        return True


class StructureInput(AbstractInput):
    structure = StorageAttribute().type(Atoms)


class MDInput(AbstractInput):
    steps = StorageAttribute().type(int)
    timestep = StorageAttribute().type(float)
    temperature = StorageAttribute().type(float)
    output_steps = StorageAttribute().type(int)


class MinimizeInput(AbstractInput):
    ionic_force_tolerance = StorageAttribute().type(float)
    max_steps = StorageAttribute().type(int)
    output_steps = StorageAttribute().type(int)


class AbstractOutput(AbstractContainer, abc.ABC):
    pass


class EnergyPotOutput(AbstractOutput):
    energy_pot = StorageAttribute().type(float)


class EnergyKinOutput(AbstractOutput):
    energy_kin = StorageAttribute().type(float)


class ForceOutput(AbstractOutput):
    forces = StorageAttribute().type(np.ndarray)


class MDOutput(HasStructure, EnergyPotOutput):
    pot_energies = StorageAttribute().type(list).constructor(list)
    kin_energies = StorageAttribute().type(list).constructor(list)
    forces = StorageAttribute().type(list).constructor(list)
    structures = StorageAttribute().type(list).constructor(list)

    def plot_energies(self):
        plt.plot(self.pot_energies - np.min(self.pot_energies), label="pot")
        plt.plot(self.kin_energies, label="kin")
        plt.legend()

    def _number_of_structures(self):
        return len(self.structures)

    def _get_structure(self, frame, wrap_atoms=True):
        return self.structures[frame]

    # TODO: how to make sure this is generally conforming?
    @property
    def energy_pot(self):
        return self.pot_energies[-1]
