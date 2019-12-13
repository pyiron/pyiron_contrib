# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_contrib.protocol.utils.misc import LoggerMixin, ensure_iterable
from pyiron.atomistics.structure.atoms import Atoms
import numpy as np
import logging

"""
Classes to override compare
"""

__author__ = "Dominik Gehringer, Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "December 10, 2019"

try:
    from xxhash import xxh64_hexdigest
    hashfunction = xxh64_hexdigest
except ImportError:
    from hashlib import sha1
    logging.getLogger('pyiron_contrib.protocol.generic').debug('Falling back to SHA1 hashing')
    hashfunction = sha1

ensure_iterable_tuple = lambda o: tuple(ensure_iterable)


class Registry(type):
    def __init__(cls, name, bases, nmspc):
        super(Registry, cls).__init__(name, bases, nmspc)
        if not hasattr(cls, 'registry'):
            cls.registry = set()
        cls.registry.add(cls)
        cls.registry -= set(bases) # Remove base classes

class Comparer(LoggerMixin, metaclass=Registry):
    """
    Class is aware of its subclasses. Subclasses must have a "type" attrivute of type "type"
    Compare two object, where the behaviour can be overriden, automatically determines the type
    """

    def __init__(self, obj):
        super(Comparer, self).__init__()
        self._object = obj
        self._cls = type(obj)
        # this is a private object, one cannot access is
        self.__registry_cache = {}

        if not isinstance(self._object, self._cls):
            raise TypeError

    @property
    def object(self):
        return self._object

    def _equals(self, b):
        if isinstance(b, Comparer):
            if b._cls != self._cls:
                return False
            else:
                b = b._object
        elif type(b) != self._cls:
            # TODO: Maybe throw a arning here
            return False
        else:
            comparer = self._get_comparer()
            return self.default(b) if comparer is None else comparer(self.object).equals(b)

    def default(self, b):
        return self._object == b

    def _get_comparer(self):
        # one can create on the fly subclasses, therefore we have to check it
        if len(self.__registry_cache) != len(self.registry):
            for cls in self.registry:
                if not hasattr(cls, 'type'):
                    raise TypeError('The subclass "%s" must have a "type" attribute' % cls.__name__)
                self.__registry_cache[cls.type] = cls

        # registry is updated
        # check if we can resolve it directly
        if self._cls in self.__registry_cache:
            return self.__registry_cache[self._cls]
        else:
            # it could be the subclass of one of the entries
            for k in self.__registry_cache.keys():
                if issubclass(self._cls, k):
                    return self.__registry_cache[k]

            return None

    def equals(self, b):
        return self.default(b)

    def __eq__(self, other):
        return self._equals(other)


# shortcut
c_ = Comparer

class NumpyArrayComparer(Comparer):

    """
    Used to compare numpy arrays
    """

    type = np.ndarray


    def get_machine_epsilon(self, A):
        """
        Returns the machine inaccuracy for the datatype of a

        Args:
            A: (np.ndarray) the array

        Returns: (float or None) the machine epsilon or None if it is an exact datatype

        """
        try:
            epsilon = np.finfo(A.dtype).eps
        except ValueError:
            epsilon = None
        return epsilon

    def equals(self, b):
        factor = 10
        epsilon = self.get_machine_epsilon(b)
        # check if the datatype is inexact at all
        inexact = epsilon is not None
        self.logger.debug("Array comparison {}, {} eval to {} & {}".format(
            self.object,
            b,
            self.object.shape == b.shape,
            np.allclose(self.object, b, atol=factor*epsilon)
        ))
        if inexact:
            return self.object.shape == b.shape and np.allclose(self.object, b, atol=factor*epsilon)
        else:
            # it is an exact data type such as int
            return np.array_equal(self.object, b)


class AtomsComparer(Comparer):

    """
    Used to compare pyiron Atoms objects
    """

    type = Atoms

    def equals(self, b):
        assert isinstance(b, Atoms)
        assert isinstance(self.object, Atoms)

        index_spec_mapping = lambda atoms: {site.index: site.symbol for site in atoms}
        # compare structures
        # https://github.com/pyiron/pyiron/blob/c447ffb4f1e003d0ebaced50a12def46beefab4f/pyiron/atomistics/job/interactive.py
        conditions = [
            len(self.object) == len(b),
            c_(self.object.cell) == b.cell,
            c_(self.object.get_scaled_positions()) == b.get_scaled_positions(),
            c_(self.object.get_initial_magnetic_moments()) == b.get_initial_magnetic_moments(),
            index_spec_mapping(self.object) == index_spec_mapping(b)
        ]
        return all(conditions)




