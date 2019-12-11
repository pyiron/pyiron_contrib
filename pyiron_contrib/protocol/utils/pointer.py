# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from enum import Enum
from pyiron_contrib.protocol.utils.misc import  LoggerMixin, requires_arguments
from types import MethodType, FunctionType

"""
Python implementation of pointers. Pointer can be resolved using ~ operator
"""

__author__ = "Dominik Gehringer, Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "December 10, 2019"

class CrumbType(Enum):
    """
    Enum class. Provides the types which Crumbs in an IODictionary are allowed to have
    """

    Root = 0
    Attribute = 1
    Item = 2


class Crumb(LoggerMixin):
    """
    Represents a piece in the path of the IODictionary. The Crumbs are used to resolve a recipe path correctly
    """

    def __init__(self, crumb_type, name):
        """
        Initializer from crumb
        Args:
            crumb_type (CrumbType): the crumb type of the object
            name (str, object): An object if crumb type is CrumbType.Root otherwise a string
        """
        self._crumb_type = crumb_type
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def crumb_type(self):
        return self._crumb_type

    @property
    def object(self):
        if self.crumb_type != CrumbType.Root:
            raise ValueError('Only root crumbs store an object')
        return self._name

    @staticmethod
    def attribute(name):
        """
        Convenience method to produce an attribute crumb

        Args:
            name (str): the name of the attribute

        Returns:
            Crumb: the attribute crumbs

        """
        return Crumb(CrumbType.Attribute, name)

    @staticmethod
    def item(name):
        """
        Convenience method to produce an item crumb

        Args:
            name (str): the name of the item

        Returns:
            Crumb: the item crumb
        """
        return Crumb(CrumbType.Item, name)

    @staticmethod
    def root(obj):
        """
        Convenience method to produce a root crumb

        Args:
            obj: The root object of the path in the IODictionary

        Returns:
            Crumb: A root crumb, meant to be the first item in a IODictionary path
        """
        return Crumb(CrumbType.Root, obj)

    def __repr__(self):
        return '<{}({}, {})'.format(self.__class__.__name__,
                                    self.crumb_type.name,
                                    self.name if isinstance(self.name, str) else self.name.__class__.__name__)

    def __hash__(self):
        crumb_hash = hash(self._crumb_type)
        if self._crumb_type == CrumbType.Root:
            crumb_hash += hash(hex(id(self.object)))
        else:
            try:
                crumb_hash += hash(self._name)
            except Exception as e:
                self.logger.exception('Failed to hash "{}" object'.format(type(self._name).__name__), exc_info=e)
        return crumb_hash

    def __eq__(self, other):
        """
        Compares the crumb with an an object "other". Will return False if "other" is not an instance of Crumb
        Args:
            other: (object) the object to compare

        Returns: (bool) wether "other" is equal to self

        """
        if not isinstance(other, Crumb):
            return False
        if self.crumb_type == other.crumb_type:
            if self.crumb_type == CrumbType.Root:
                return hex(id(self.object)) == hex(id(other.object))
            else:
                return self.name == other.name
        else:
            return False


class Path(list, LoggerMixin):
    """
    A object representing a path to an objects attribute. It is a list of "Crumbs"
    The first object of always a Crumb of type CrumbType.Root followed by an arbitrary sequence of CrumbType.Item
    or CrumbType.Attribute
    """

    def append(self, item):
        if not isinstance(item, Crumb):
            raise TypeError('A path can only consist of crumbs')
        else:
            super(Path, self).append(item)

    def extend(self, collection):
        if not all([isinstance(item, Crumb) for item in collection]):
            raise TypeError('A path can only consist of crumbs')
        else:
            super(Path, self).extend(collection)

    def index(self, item, **kwargs):
        if not isinstance(item, Crumb):
            raise TypeError('A path can only consist of crumbs')
        else:
            return super(Path, self).index(item, **kwargs)

    def count(self, item):
        if not isinstance(item, Crumb):
            raise TypeError('A path can only consist of crumbs')
        else:
            return super(Path, self).count(item)

    @classmethod
    def join(cls, *p):
        return Path(p)


class Pointer(LoggerMixin):
    """
    A class pointing to an object. Can be resolved with ~p. It can be dangling at definition
    """

    def __init__(self, root):
        if root is not None:
            if not isinstance(root, Path):
                if not isinstance(root, Crumb):
                    path = [Crumb.root(root)]
                else:
                    path = [root]
            else:
                path = root.copy()
        else:
            raise ValueError('Root object can never be "None"')
        self.__path = path

    def __getattr__(self, item):
        return Pointer(Path.join(*self.__path, Crumb.attribute(item)))

    def __getitem__(self, item):
        return Pointer(Path.join(*self.__path, Crumb.item(item)))

    @property
    def path(self):
        return self.__path

    def _resolve_path(self):
        """
        This method resolves the object hiding behind a path (A list of Crumbs)

        Args:
            path (list<Crumb>): A list of path crumbs used to resolve the data object
            remaining (int): How many crumbs should be resolved

        Returns:
            The underlying data object, hiding behind path parameter

        """
        # Make a copy to ensure, because by passing by reference
        path = self.path.copy()

        # Have a look at the path and check that it starts with a root crumb
        root = path.pop(0)
        if root.crumb_type != CrumbType.Root:
            raise ValueError('Got invalid path. A valid path starts with a root object')
        # First element is always an object
        result = root.object
        while len(path) > 0:
            # Take one step in the path, pop the next crumb from the list
            crumb = path.pop(0)
            crumb_type = crumb.crumb_type
            crumb_name = crumb.name
            # If the result is a pointer itself we have to resolve it first
            if isinstance(result, Pointer):
                self.logger.info('Resolved pointer in a pointer path')
                result = ~result
            if isinstance(crumb_name, Pointer):
                self.logger.info('Resolved pointer in a pointer path')
                crumb_name = ~crumb_name
            # Resolve it with the correct method - dig deeper
            if crumb_type == CrumbType.Attribute:
                try:
                    result = getattr(result, crumb_name)
                except AttributeError as e:
                    self.logger.exception('Cannot fetch value "{}"'.format(crumb_name), exc_info=e)
                    raise e
            elif crumb_type == CrumbType.Item:
                try:
                    result = result.__getitem__(crumb_name)
                except (TypeError, KeyError) as e:
                    raise e

            # Get out of resolve mode
        return result

    def _resolve_function(self, function):
        """
        Convenience function to make IODictionary.resolve more readable. If the value is a function it calls the
        resolved functions if the do not require arguments

        Args:
            key (str): the key the value belongs to, just for logging purposes
            value (object/function): the object to resolve

        Returns:
            (object): The return value of the functions, if no functions were passed "value" is returned
        """
        result = function
        if isinstance(function, (MethodType, FunctionType)):
            # Make sure that the function has not parameters or all parameters start with
            if not requires_arguments(function):
                try:
                    # Get the return value
                    result = function()
                except Exception as e:
                    self.logger.exception('Failed to execute callable to resolve values for '
                                          'path {}'.format(self), exc_info=e)
                else:
                    self.logger.debug('Successfully resolved callable for path {}'.format(self))
            else:
                self.logger.warning('Found function, but it takes arguments! I \'ll not resolve it.')
        return result

    def __invert__(self):
        return self.resolve()

    def resolve(self):
        return self._resolve_function(self._resolve_path())
