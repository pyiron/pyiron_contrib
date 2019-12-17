from __future__ import print_function
# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from logging import getLogger
from inspect import getargspec
from pydoc import locate
from itertools import islice
import re
"""
Classes for handling protocols, particularly setting up input and output pipes.
"""

__author__ = "Dominik Gehringer, Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "18 July, 2019"



def ordered_dict_get_index(ordered_dict, index):
    """
    Gets the object at "index" of an collections.OrderedDict without copying the keys list
    Args:
        ordered_dict: (collections.OrderedDict) the dict to get the value from
        index: (int) the index

    Returns: (object) the object at "index"

    """
    return ordered_dict[next(islice(ordered_dict, index, None))]

def ordered_dict_get_last(ordered_dict):
    """
    Gets the last most recently added object of an collections.OrderedDict instance

    Args:
        ordered_dict: (collections.OrderedDict) the dict to get the value from

    Returns: (object) the object at the back

    """

    return ordered_dict[next(reversed(ordered_dict))]


class LoggerMixin(object):
    """
    A class which is meant to be inherited from. Provides a logger attribute. The loggers name is the fully
    qualified type name of the instance
    """

    def fullname(self):
        """
        Returns the fully qualified type name of the instance

        Returns:
            str: fully qualified type name of the instance
        """
        return '{}.{}'.format(self.__class__.__module__, self.__class__.__name__)

    @property
    def logger(self):
        return getLogger(self.fullname())


def requires_arguments(func):
    """
    Determines if a function of method needs arguments, ingores self

    Args:
        func: (callable) the callable

    Returns: (bool) wether arguments (except "self" for methods) are needed

    """
    args, varargs, varkw, defaults = getargspec(func)
    if defaults:
        args = args[:-len(defaults)]
    # It could be a bound method too
    if 'self' in args:
        args.remove('self')
    return len(args) > 0

flatten = lambda l: [item for sublist in l for item in sublist]

def fullname(obj):
    """
    Returns the fully qualified class name of an object

    Args:
        obj: (object) the object

    Returns: (str) the class name

    """
    obj_type = type(obj)
    return '{}.{}'.format(obj_type.__module__, obj_type.__name__)


def get_cls(string):
    return locate([v for v in re.findall(r'(?!\.)[\w\.]+(?!\.)', string)if v != 'class'][0])


def is_iterable(o):
    """
    Convenience method to test for an iterator

    Args:
        o: the object to test

    Returns:
        bool: wether the input argument is iterable or not
    """
    try:
        iter(o)
    except TypeError:
        return False
    else:
        return not isinstance(o, str)

# convenience function to ensure the passed argument is iterable
ensure_iterable = lambda v: v if is_iterable(v) else [v]


class Registry(type):
    def __init__(cls, name, bases, nmspc):
        super(Registry, cls).__init__(name, bases, nmspc)
        if not hasattr(cls, 'registry'):
            cls.registry = set()
        cls.registry.add(cls)
        cls.registry -= set(bases) # Remove base classes