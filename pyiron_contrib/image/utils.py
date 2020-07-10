from __future__ import print_function
# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import inspect
from importlib import import_module
from pyiron_contrib.protocol.generic import LoggerMixin
from weakref import WeakKeyDictionary
from collections import UserList
from operator import itemgetter

"""
Code used by the image library which isn't specific to the task of images, but which doesn't have a home anywhere else
yet.
"""

__author__ = "Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Feb 3, 2020"


def _elementwise_other(function):
    def wrapper(self, other):
        if hasattr(other, '__len__') and len(other) == len(self):
            return DistributingList(getattr(obj, function.__name__)(oth) for obj, oth in zip(self, other))
        else:
            return DistributingList(getattr(obj, function.__name__)(other) for obj in self)
    return wrapper


def _decorate_by_name(decorator, names):
    def decorate(cls):
        for name in names:
            if hasattr(cls, name):
                setattr(cls, name, decorator(getattr(cls, name)))
        return cls
    return decorate


@_decorate_by_name(decorator=_elementwise_other,
                   names=[
                       '__eq__', '__ne__', '__lt__', '__gt__', '__le__', '__ge__',
                       '__add__', '__sub__', '__mul__', '__floordiv__', '__div__',
                       '__mod__', '__divmod__', '__pow__',
                       '__lshift__', '__rshift__', '__and__', '__or__', '__xor__',
                       '__radd__', '__rsub__', '__rmul__', '__rfloordiv__', '__rdiv__',
                       '__rmod__', '__rdivmod__', '__rpow__',
                       '__rlshift__', '__rrshift__', '__rand__', '__ror__', '__rxor__',
                       '__iadd__', '__isub__', '__imul__', '__ifloordiv__', '__idiv__',
                       '__imod__', '__idivmod__', '__ipow__',
                       '__ilshift__', '__irshift__', '__iand__', '__ior__', '__ixor__',
                   ])
class DistributingList(UserList):
    """
    A list-like class which resolves attribute and function calls by returning a list-like class of the corresponding
    call on each child object.

    TODO:
        - __dir__ for autocomplete?
        - Elementwise decoration for in-place magic methods
        - Setattr stuff
        - Why do strings not have an __iadd__ method?! They clearly work with +=, so what is it calling?
    """

    def __getattr__(self, item):
        return DistributingList([getattr(obj, item) for obj in self])

    def __call__(self, *args, **kwargs):
        ret = [obj.__call__(*args, **kwargs) for obj in self]
        if all(x is None for x in ret):
            return None
        else:
            return DistributingList(ret)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return DistributingList(super(DistributingList, self).__getitem__(item))
        elif isinstance(item, (list, tuple, np.ndarray)):
            if len(self) == len(item) and all(isinstance(i, bool) for i in item):
                return DistributingList(obj for obj, i in zip(self, item) if i)
            else:
                return DistributingList(itemgetter(*item)(self))
        else:
            return super(DistributingList, self).__getitem__(item)


class LockedIfAttributeTrue(LoggerMixin):
    """
    A descriptor which prevents modification when the provided attribute of the owning instance is True. 
    
    Educational credit goes to:
    https://nbviewer.jupyter.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb
    
    Attributes:
        default: The default value for the descriptor.
        attribute_name (str): The name of the attribute to look for in the owner instance when determining lock state.
        data (weakref.WeakKeyDictionary): A container to track instances.
        name (str): The name of the attribute the descriptor is being assigned to.
    """
    def __init__(self, default, attribute_name):
        self.default = default
        self.attribute_name = attribute_name
        self.data = WeakKeyDictionary()
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return self.data.get(instance, self.default)

    def __set__(self, instance, value):
        if getattr(instance, self.attribute_name):
            self.logger.warning("The attribute '{}' cannot be modified while '{}' is True".format(self.name,
                                                                                                  self.attribute_name))
        else:
            self.data[instance] = value


class ModuleScraper:
    """
    A class which scrapes through a module and applies classes and primitives found as attributes of itself, functions
    found as methods of itself, and sub-modules found recursively as new `ModuleScraper` attributes of itself.

    A decorator can optionally be applied to all functions found. This is the real strength, since

    Note:
        Doesn't do anything until its `activate` method is called.

    TODO:
        - Allow decorators also for classes
        - Perhaps some sort of more complex mapping so that the decorators aren't just uniformly applied to every
          function/class that's found, but can be selectively applied.

    Attributes:
        safe (bool): Whether to skip values beginning with an underscore. (Default is True, do skip.)
        recursive (bool): Whether to recursively activate submodules. (Default is False: submodules are noted by
            creating a new `ModuleScraper` instance and setting it as an attribute, but contents cannot be accessed
            until the submodule itself is explicitly activated.)
        scrape_functions (bool): Whether to look for functions in the module. (Default is True.)
        scrape_classes (bool): Whether to look for class definitions in the module. (Default is True.)
        scrape_primitives (bool): Whether to look for primitives in the module. (Default is True.)
        primitives_list (tuple/list): A list of which types count as primitive. (Default is None, which uses `(int,
            float, bool, numpy.ndarray)`.)
    """

    safe = LockedIfAttributeTrue(True, '_activated')
    recursive = LockedIfAttributeTrue(True, '_activated')
    scrape_functions = LockedIfAttributeTrue(True, '_activated')
    scrape_classes = LockedIfAttributeTrue(True, '_activated')
    scrape_primitives = LockedIfAttributeTrue(True, '_activated')
    primitives_list = LockedIfAttributeTrue((int, float, bool, np.ndarray), '_activated')

    def __init__(
            self,
            module,
            decorator=None,
            decorator_args=None,
            safe=True,
            recursive=True,
            scrape_functions=True,
            scrape_classes=True,
            scrape_primitives=True,
            primitives_list=None
    ):
        """
        Args:
            module (module/str): The module from which to scrape, or the name of the module from which to escape, e.g.
                `skimage.filters` or `'skimage.filters'`.
            decorator (fnc): A decorator function to apply to scraped functions. (Default is None, no decorator.)
            decorator_args (tuple/list): Arguments to pass to the decorator. (Default is None, no args.)
            safe (bool): Whether to skip values beginning with an underscore. (Default is True, do skip.)
            recursive (bool): Whether to recursively activate submodules. (Default is False: submodules are noted by
                creating a new `ModuleScraper` instance and setting it as an attribute, but contents cannot be accessed
                until the submodule itself is explicitly activated.)
            scrape_functions (bool): Whether to look for functions in the module. (Default is True.)
            scrape_classes (bool): Whether to look for class definitions in the module. (Default is True.)
            scrape_primitives (bool): Whether to look for primitives in the module. (Default is True.)
            primitives_list (tuple/list): A list of which types count as primitive. (Default is None, which uses `(int,
                float, bool, numpy.ndarray)`.)
        """
        self._module = module
        self._decorator = decorator
        self._decorator_args = decorator_args or ()
        self._activated = False

        self.safe = safe
        self.recursive = recursive
        self.scrape_functions = scrape_functions
        self.scrape_classes = scrape_classes
        self.scrape_primitives = scrape_primitives
        self.primitives_list = primitives_list or (int, float, bool, np.ndarray)

    def activate(self):
        """
        Parse the module assigned at instantiation, using decorator information assigned at instantiation.
        """
        if inspect.ismodule(self._module):
            module = self._module
        else:
            module = import_module(self._module)

        for name, obj in inspect.getmembers(module):
            if self.safe and name[0] == '_':
                continue
            elif self.recursive and inspect.ismodule(obj) and obj.__package__ == module.__package__:
                # Behave recursively for submodules
                submodule = ModuleScraper(
                    obj,
                    decorator=self._decorator,
                    decorator_args=self._decorator_args,
                    safe=self.safe,
                    recursive=self.recursive,
                    scrape_functions=self.scrape_functions,
                    scrape_classes=self.scrape_classes,
                    scrape_primitives=self.scrape_primitives,
                    primitives_list=self.primitives_list
                )
                setattr(self, name, submodule)
                if self.recursive:
                    submodule.activate()
            elif self.scrape_functions and inspect.isfunction(obj):
                # Set all module functions as methods
                if self._decorator is not None:
                    fnc = self._decorator(*self._decorator_args)(obj)
                else:
                    fnc = obj
                setattr(self, name, fnc)
            elif self.scrape_classes and inspect.isclass(obj):
                # Grab classes
                setattr(self, name, obj)
            elif self.scrape_primitives and isinstance(obj, self.primitives_list):
                # Grab primitives
                setattr(self, name, obj)

        self._activated = True

    def __getattr__(self, item):
        if not self._activated:
            self.activate()
            return getattr(self, item)
        else:
            if inspect.ismodule(self._module):
                name = self._module.__name__.split('.')[-1]
            else:
                name = self._module.split('.')[-1]
            raise AttributeError(
                "'{0}' has no attribute '{1}' first.".format(name, item))

    def to_hdf(self):
        pass

    def from_hdf(self):
        pass
