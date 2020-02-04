from __future__ import print_function
# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import inspect
from importlib import import_module

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


class ModuleScraper:
    """
    A class which scrapes through a module and applies primitives found as attributes of itself, functions found as
    methods of itself, and sub-modules found recursively as new `ModuleScraper` attributes of itself.

    A decorator can optionally be applied to all functions found.

    Note:
        Doesn't do anything until its `activate` method is called.

    Note:
        Skips all module values beginning with an underscore unless activated with.

    Warning:
        Doesn't do anything with classes that are found.
    """

    def __init__(self, module, decorator=None, decorator_args=None):
        """
        Args:
            module (module/str): The module from which to scrape, or the name of the module from which to escape, e.g.
                `skimage.filters` or `'skimage.filters'`.
            decorator (fnc): A decorator function to apply to scraped functions. (Default is None, no decorator.)
            decorator_args (tuple/list): Arguments to pass to the decorator. (Default is None, no args.)
        """
        self._module = module
        self._decorator = decorator
        self._decorator_args = decorator_args or ()
        self._activated = False

    def activate(self, safe=True, recursive=False):
        """
        Parse the module assigned at instantiation, using decorator information assigned at instantiation.

        Args:
            safe (bool): Whether to skip values beginning with an underscore. (Default is True, do skip.)
            recursive (bool): Whether to recursively activate submodules. (Default is False: submodules are noted by
                creating a new `ModuleScraper` instance and setting it as an attribute, but contents cannot be accessed
                until the submodule itself is explicitly activated.)
        """
        if inspect.ismodule(self._module):
            module = self._module
        else:
            module = import_module(self._module)

        for name, obj in inspect.getmembers(module):
            if safe and name[0] == '_':
                continue

            primitives = (int, float, bool, np.ndarray)
            if inspect.isfunction(obj):
                # Set all module functions as methods
                if self._decorator is not None:
                    fnc = self._decorator(*self._decorator_args)(obj)
                else:
                    fnc = obj
                setattr(self, name, fnc)
            elif recursive and inspect.ismodule(obj) and obj.__package__ == module.__package__:
                # Behave recursively for submodules
                submodule = ModuleScraper(obj, decorator=self._decorator, decorator_args=self._decorator_args)
                setattr(self, name, submodule)
                if recursive:
                    submodule.activate(safe=safe, recursive=True)
            elif inspect.isclass(obj) or isinstance(obj, primitives):
                # Set all classes and all primitives (of the pre-registered types) as attributes
                setattr(self, name, obj)

    def __getattr__(self, item):
        try:
            super(self.__class__, self).__getattribute__(item)
        except AttributeError:
            if inspect.ismodule(self._module):
                name = self._module.__name__.split('.')[-1]
            else:
                name = self._module.split('.')[-1]
            raise AttributeError(
                "'{0}' has no attribute '{1}'. Try running '....{0}.activate()' first.".format(name, item))

    def to_hdf(self):
        pass

    def from_hdf(self):
        pass
