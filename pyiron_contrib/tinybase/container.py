"""Generic Input Base Clases"""

import abc
import dataclasses
from dataclasses import dataclass, field, fields
from copy import deepcopy
from typing import TypeVar, List

from pyiron_contrib.tinybase.storage import Storable, GenericStorage

from pyiron_base.interfaces.object import HasStorage
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure

from ase import Atoms
import numpy as np
import numpy.typing as npt
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


# derives from ABCMeta instead of type, so that other classes can use it as a metaclass and still derive from Storable
# (which derives from ABC and therefor already has a metaclass)
class _MakeDataclass(abc.ABCMeta):
    def __new__(meta, name, bases, ns, **kwargs):
        cls = super().__new__(meta, name, bases, ns)
        return dataclass(cls)


class StorableDataclass(Storable, metaclass=_MakeDataclass):
    """
    Base class for data classes that automatically implement Storable.

    Sub classes are automatically turned into dataclasses without the need for a separate decorator.
    """

    def _store(self, storage):
        for field in dataclasses.fields(self):
            storage[field.name] = getattr(self, field.name)

    @classmethod
    def _restore(cls, storage, version: str):
        state = {}
        for name in storage.list_nodes():
            # FIXME/TODO: avoid this somehow, likely will be related to
            # splitting the storage layer into a raw and an object level
            # version
            if name in ["MODULE", "NAME", "VERSION"]:
                continue
            state[name] = storage[name]

        for name in storage.list_groups():
            value = storage[name]
            if isinstance(value, GenericStorage):
                state[name] = value.to_object()
            else:
                state[name] = value

        return cls(**state)

    # exists so that SeriesInput can overload it
    def fields(self):
        return fields(self)


class Sentinel:
    _instances = {}

    def __new__(cls, name):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(self, name):
        self._name = name

    def __copy__(self):
        return self

    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return f"{type(self).__name__}({self._name})"

    def __str__(self):
        return self._name


USER_REQUIRED = Sentinel("USERINPUT")


class AbstractInput(StorableDataclass):
    def check_ready(self):
        return all(
            getattr(self, field.name) is not USER_REQUIRED for field in fields(self)
        )


class StructureInput(AbstractInput):
    structure: Atoms = USER_REQUIRED


class MDInput(AbstractInput):
    steps: int = USER_REQUIRED
    timestep: float = USER_REQUIRED
    temperature: float = USER_REQUIRED
    output_steps: int = USER_REQUIRED


class MinimizeInput(AbstractInput):
    ionic_force_tolerance: float = 1e-5
    max_steps: int = 500
    output_steps: int = 500


class AbstractOutput(StorableDataclass):
    pass


class EnergyPotOutput(AbstractOutput):
    energy_pot: float


class EnergyKinOutput(AbstractOutput):
    energy_kin: float


class ForceOutput(AbstractOutput):
    forces: npt.NDArray[float]


class StaticOutput(EnergyPotOutput, EnergyKinOutput):
    pass


T = TypeVar("T")


class StaticMode(abc.ABC):
    @abc.abstractmethod
    def select(self, array: npt.NDArray[T]) -> T:
        pass


class MDOutput(HasStructure, AbstractOutput):
    pot_energies: npt.NDArray[float]
    kin_energies: npt.NDArray[float]
    forces: npt.NDArray[float]
    structures: List[Atoms]

    def plot_energies(self):
        plt.plot(self.pot_energies - np.min(self.pot_energies), label="pot")
        plt.plot(self.kin_energies, label="kin")
        plt.legend()

    def _number_of_structures(self):
        return len(self.structures)

    def _get_structure(self, frame, wrap_atoms=True):
        return self.structures[frame]

    # both StaticMode sub classes live here only so that users can easily
    # access them later
    class Mean(StaticMode):
        """
        Average over the given range of steps.
        """

        __slots__ = ("_start", "_stop")

        def __init__(self, start: float, stop: float = 1.0):
            assert 0 <= start <= 1 and 0 <= stop <= 1, "Range check!"
            self._start, self._stop = start, stop

        def select(self, array):
            na = int((len(array) - 1) * self._start)
            no = int((len(array) - 1) * self._stop)
            return array[na:no].mean(axis=0)

    class Last(StaticMode):
        """
        Return the last step.
        """

        __slots__ = ()

        def select(self, array):
            return array[-1]

    def static_output(self, how: StaticMode = Last()) -> StaticOutput:
        """
        Act as a static output.
        """
        state = {
            "energy_pot": how.select(self.pot_energies),
            "energy_kin": how.select(self.kin_energies),
        }
        return StaticOutput(**state)
