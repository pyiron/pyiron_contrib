# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from enum import Enum
from logging import getLogger
from types import FunctionType, MethodType
from inspect import getargspec
import dill
import threading
import logging
import inspect
from pyiron.atomistics.structure.atoms import Atoms
import numpy as np
from pydoc import locate
import re
from pyiron.atomistics.job.interactive import GenericInteractive

"""
Classes for handling protocols, particularly setting up input and output pipes.
"""

__author__ = "Dominik Noeger, Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "18 July, 2019"


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
    args, varargs, varkw, defaults = getargspec(func)
    if defaults:
        args = args[:-len(defaults)]
    # It could be a bound method too
    if 'self' in args:
        args.remove('self')
    return len(args) > 0

flatten = lambda l: [item for sublist in l for item in sublist]

def fullname(obj):
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


class IODictionary(dict, LoggerMixin):
    """
    A dictionary class representing the parameters of a Command class. The dictionary holds a path which is recipe
    that can be resolved at runtime to obtain underlying values. A dictionary instance can hold multiple instances of
    IODictionary as value items which can be resolved into the real values when desired.
    """

    # those members are not accessible
    _protected_members = [
        'resolve',
        'to_hdf',
        'from_hdf'
    ]

    def __init__(self, **kwargs):
        super(IODictionary, self).__init__(**kwargs)

    def __getattr__(self, item):
        if item in IODictionary._protected_members:
            return object.__getattribute__(self, item)
        return self.__getitem__(item)

    def __getitem__(self, item):
        if item in self.keys():
            value = super(IODictionary, self).__getitem__(item)
            if isinstance(value, Pointer):
                return ~value
            elif isinstance(value, list):  # TODO: Allow other containers than list
                cls = type(value)
                return cls([element if not isinstance(element, Pointer) else ~element for element in value])
            else:
                return value
        return super(IODictionary, self).__getitem__(item)

    def __setattr__(self, key, value):
        super(IODictionary, self).__setitem__(key, value)

    def resolve(self):
        """
        Even though printing the dictionary, or asking for a particular item resolves paths fine, somehow using the **
        unpacking syntax fails to resolve pointers. This is to cover that issue since I couldn't find anything on
        google how to modify the ** behaviour.
        """
        resolved = {}
        for key in self.keys():
            resolved[key] = self.__getitem__(key)
        return resolved

    def _try_save_key(self, k, v, hdf):
        if hasattr(v, 'to_hdf'):
            v.to_hdf(hdf, group_name=k)
            result = True
        else:
            try:
                hdf[k] = v
                result = True
            except TypeError as e:
                result = False
        return result


    def _generic_to_hdf(self, value, hdf, group_name=None):
                if isinstance(value, dict):
                    with hdf.open(group_name) as server:
                        server['TYPE'] = str(type(value))
                        server['FULLNAME'] = fullname(value)
                        for k, v in value.items():
                            # try to save it
                            if not isinstance(k, str):
                                self.logger.warning('Key "%s" is not string, it will be converted to %s' %( k, str(k)))
                                k = str(k)
                            if self._try_save_key(k, v, server):
                                pass # everything was successful
                            else:
                                self._generic_to_hdf(v, server, group_name=k)
                elif isinstance(value, (list, tuple)):
                    # check if all do have the same type -> then we can make a numpy array out of it
                    if len(value) == 0:
                        pass # there is nothing to do, no data to store
                    else:
                        first_type = type(value[0])
                        same = all([type(v) == first_type for v in value])
                        # if all items are of the same type and it is simple
                        if same and issubclass(first_type, (float, str, complex, int, np.ndarray)):
                             # that is trivial we do have an array
                            if issubclass(first_type, np.ndarray):
                                # we do not want dtype object, thus we do make this distinction
                                hdf[group_name] = np.array(value)
                            else:
                                hdf[group_name] = np.array(value, dtype=first_type)
                        else:
                            with hdf.open(group_name) as server:
                                server['TYPE'] = str(type(value))
                                server['FULLNAME'] = fullname(value)
                                for i, v in enumerate(value):
                                    if self._try_save_key(str(i), v, server):
                                        pass  # everything was successful
                                    else:
                                        self._generic_to_hdf(v, server, group_name=str(i))




    def to_hdf(self, hdf, group_name=None):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server['TYPE'] = str(type(self))
            for key in list(self.keys()):
                # default value is not to save any property
                try:
                    value = getattr(self, key)
                    try:
                        value.to_hdf(hdf5_server, group_name=key)
                    except AttributeError:
                        hdf5_server[key] = value
                except KeyError:
                    # to_hdf will get called *before* protocols have run, so the pointers in these dictionaries
                    # won't be able to resolve. For now just let it not resolve and don't save it.
                    continue
                except TypeError as e:
                    # Saving a list of complex objects, e.g. Atoms, was failing. Here we save them individually
                    # it seems pyiron could not handle those
                    self.logger.warning('TypeError(%s): %s : %s' %(e, key, value))
                    # TODO: Treat arbitrarily deep nesting of such objects
                   try:
                       self._generic_to_hdf(value, hdf5_server, group_name=key)
                   except Exception:
                       raise

    def from_hdf(self, hdf, group_name):
        with hdf.open(group_name) as hdf5_server:
            for key in hdf5_server.list_nodes():
                if key == 'TYPE':
                    continue
                # Nodes are leaves, so just save them directly
                try:
                    setattr(self, key, hdf5_server[key])
                except Exception as e:
                    self.logger.exception('Failed to load "{}"'.format(key), exc_info=e)
            for key in hdf5_server.list_groups():
                # Groups are more complex data types with their own depth
                # For now we only treat other IODicts and Atoms (i.e. structures) explicitly.
                if hdf5_server[key]['TYPE'] == str(IODictionary):
                    iodict = IODictionary()
                    iodict.from_hdf(hdf5_server, key)
                    setattr(self, key, iodict)
                elif hdf5_server[key]['TYPE'] == str(Atoms):
                    struct = Atoms()
                    struct.from_hdf(hdf5_server, key)
                    setattr(self, key, struct)
                # From dominik's branch
                # elif get_cls(hdf5_server[key]['TYPE']) in known_types:
                #
                #     value = generic_from_hdf(hdf5_server[key])
                #     setattr(self, key, value)
                else:
                    raise TypeError(
                        "The only complex group-level hdf objects allowed are IODictionaries or Atoms, but {} was "
                        "{}".format(key, hdf5_server[key]['TYPE'])
                    )

class InputDictionary(IODictionary):
    """
    An ``IODictionary`` which is instantiated with a child dictionary to store default values. If a requested item
    can't be found in this dictionary, a default value is sought.
    """

    def __init__(self):
        super(InputDictionary, self).__init__()
        self.default = IODictionary()

    def __getitem__(self, item):
        try:
            return super(InputDictionary, self).__getitem__(item)
        except (KeyError, IndexError):
            #self.logger.warning('Falling back to default values for key "{}"'.format(item))
            return self.default.__getitem__(item)

    def __getattr__(self, item):
        if item == 'default':
            return object.__getattribute__(self, item)
        else:
            return super(InputDictionary, self).__getattr__(item)

    def __setattr__(self, key, value):
        if key == 'default':
            object.__setattr__(self, key, value)
        else:
            super(InputDictionary, self).__setattr__(key, value)

    def keys(self):
        both_keys = set(super(InputDictionary, self).keys()).union(self.default.keys())
        for k in both_keys:
            yield k

    def values(self):
        for k in self.keys():
            yield self[k]

    def items(self):
        # Make sure the dictionary resolve pointers
        for k in self.keys():
            # It resolves the, since we call __getitem__
            yield k, self[k]

    def __iter__(self):
        # Make sure all keys get into the ** unpacking also those from the default dictionary
        return self.keys().__iter__()



"""
Provides classes for a custom more feature rich signal-slot system than PyQt4's. Allows slot deletion, and explicit
handler execution.
"""


class Synchronization(object):
    """
    Helper class which creates a mutex.
    """

    def __init__(self):
        super(Synchronization, self).__init__()
        self.mutex = threading.RLock()


class Event(Synchronization):
    """
    A which implements a signal slot system. It executes event handlers synchronized. An event handler may be
    :obj:`EventHandler` object, :obj:`function` or a :obj:`lambda`. Something :obj:`callable` at least.

    If you pass a :obj:`callable` as event handler it must have a ``__name__`` attribute. The handlers name will be
    ``eventHandler.name`` if it is a :obj:`EventHandler` object or a ``functions.__name__`` attribute.

    Lambdas do not get an identifier. So you cannot remove them except by :func:`clear_handlers`,
    but this resets all handlers.

    Attributes:
        suppressed (bool): A flag that indicates if the event object should block the fire calls. If set to
            `True` signals will be blocked.

    """

    def __init__(self):
        super(Event, self).__init__()
        self.__handlers = {}
        self.__unnamed_handlers = []
        self.supressed = False
        self.__synchronize(
            names=['add_event_handler', 'remove_event_handler', '__add__', '__sub__', 'handler_count', '__getitem__'])

    @property
    def handler_count(self):
        """
        Counts how many handlers are registered

        Returns:
            int: The total number of handlers
        """
        return len(self.__handlers) + len(self.__unnamed_handlers)

    @property
    def handler_keys(self):
        """
        `Property:` Creates a list with the names of all handlers

        Returns:
            :obj:`list` of :obj:`str`: a list with all handler names
        """
        return self.__handlers.keys()

    @property
    def named_handlers(self):
        """
        `Property:` All handlers which are registered with a name

        Returns:
             :obj:`list`: a list of all named handlers (handler with an identifier)
        """
        return self.__handlers.items()

    @property
    def unnamed_handlers(self):
        """
       `Property:` List of all lambdas

        Returns:
            :obj:`list` of :obj:`function`: a list of all registered lambdas
        """
        return self.__unnamed_handlers

    def add_event_handler(self, handler):
        """
        Adds a event handler to event. Each handler is identified by a string. The handlers identifier will be
        ``eventHandler.name`` if it is a :obj:`EventHandler` object or a ``functions.__name__`` attribute.
        Lambdas do not get an identifier. To fire this handler explicitly use::

            event.fire_handler(identifier)

        You can shortcut this function by using::

            event += lambda args: do_something(arg)
            event += EventHandler('my_handler_name', my_awesome_handler_func)
            event += my_object.my_awesome_handler_func

        Args:
            handler (:obj:`EventHandler`, :obj:`function` or :obj:`lambda`): the callable handler object

        Raises:
            TypeError: If handler is no instance of :obj:`EventHandler` a :obj:`function` or a :obj:`lambda`
            KeyError: If handlers identifier is already registered
        """
        if callable(handler):
            if hasattr(handler, '__name__'):
                name = handler.__name__
                if name == '<lambda>':
                    self.__unnamed_handlers.append(handler)
                    return
            else:
                self.__unnamed_handlers.append(handler)
                return
        elif isinstance(handler, EventHandler):
            name = handler.name
        else:
            raise TypeError(
                'The argument of add_event_handler must be of type event.EventHandler, function or a lambda expression')
        if name not in self.__handlers.keys():
            self.__handlers[name] = handler
        else:
            raise KeyError('The event already contains a handler named "{0}"'.format(name))

    def clear_handlers(self):
        """
        Removes all handlers, even registered :obj:`lambda` s
        """
        self.__handlers = {}
        self.__unnamed_handlers = []

    def fire(self, *args, **kwargs):
        """
        Causes all registered handlers to execute

        Note:
            The arguments specified in `*args` and `**kwargs` must be consistent with the handlers signatures

        Args:
            *args: Arguments passed to the handler functions
            **kwargs: Keyword arguments passed to the handler functions
        """
        self.mutex.acquire()
        try:
            local_copy = list(self.__handlers.values()) + self.__unnamed_handlers
        finally:
            self.mutex.release()
        if not self.supressed:
            for handler in local_copy:
                try:
                    if callable(handler):
                        if hasattr(handler, '__name__'):
                            name = handler.__name__
                        else:
                            name = str(handler)
                        handler(*args, **kwargs)
                    else:
                        name = handler.name
                        handler.func(*args, **kwargs)
                except:
                    logging.getLogger(__name__).exception(
                        'An error ocurred while executing function {func}.'.format(func=name), exc_info=True)

    def fire_handler(self, handler_name, *args, **kwargs):
        """
        Fires only a specific registered hander. Only handlers with a name can be fired explicitly.

        Theses handlers may be :obj:`EventHandler` objects or named :obj:`functions`

        Args:
            handler_name (:obj:`str`): The identifier of the handler
                If the underlying event handler is of type :obj:`EventHandler` the name is ``event_handler_obj.name``
                In case of a :obj:`function` the identifier is ``my_handler_func.__name__``
            *args: Arguments passed to the handler function
            **kwargs: Keyword arguments passed to the handler function
        """
        self.mutex.acquire()
        try:
            handler = self[handler_name]
        finally:
            self.mutex.release()
        if not self.supressed:
            try:
                if callable(handler):
                    if hasattr(handler, '__name__'):
                        name = handler.__name__
                    else:
                        name = str(handler)
                    handler(*args, **kwargs)
                else:
                    name = handler.name
                    handler.func(*args, **kwargs)
            except:
                logging.getLogger(__name__).exception(
                    'An error ocurred while executing function {func}.'.format(func=name), exc_info=True)

    def has_handler(self, handler):
        """
        Method to check if a certain event handler is already registered at the event

        Args:
            handler (:obj:`function`, :obj:`str` or :obj:`EventHandler`):  The handler or its name to check

        Returns:
            bool: `True` if handler is available else `False`
        """
        if callable(handler):
            if hasattr(handler, '__name__'):
                name = handler.__name__
            else:
                name = str(handler)
        elif isinstance(handler, EventHandler):
            name = handler.name
        elif isinstance(handler, str):
            name = handler
        return name in self.__handlers.keys()

    def remove_event_handler(self, handler):
        """
        Removes the given event handler from the event

        You can also shortcut this function by using::

            event -= 'my_handlers_identifier_string'
            event -= my_event_handler_object
            event -= my_object.my_awesome_handler_func

        Note:
            You cannot remove :obj:`lambda` expressions explicitly.

        Args:
            handler (:obj:`EventHandler`, :obj:`function` or :obj:`lambda`): the callable handler object

        Raises:
            TypeError: If handler is no instance of :obj:`EventHandler` a :obj:`function` or a :obj:`lambda`
            KeyError: If handlers name is not registered
        """
        if isinstance(handler, EventHandler):
            name = handler.name
        elif isinstance(handler, str):
            name = handler
        elif isinstance(handler, FunctionType) or inspect.ismethod(handler):
            name = handler.__name__
        else:
            raise TypeError('The argument of remove_event_handler must be of type event. EventHandler, function or a '
                            'lambda expression')
        if name in self.__handlers.keys():
            del self.__handlers[name]
        else:
            raise KeyError('The event does not contain a handler named "{0}"'.format(name))

    def set_event_handler(self, handler):
        """
        Reassigns a event handler

        Args:
            handler (:obj:`EventHandler`, :obj:`function` or :obj:`lambda`): the callable handler object

        Raises:
            TypeError: If handler is no instance of :obj:`EventHandler` a :obj:`function` or a :obj:`lambda`
            KeyError: If handlers name is not registered
        """
        if callable(handler):
            if hasattr(handler, '__name__'):
                name = handler.__name__
                if name == '<lambda>':
                    raise KeyError('A lambda expression cannot be reassigned')
            else:
                raise KeyError('Only a named handler can be reassigned')
        elif isinstance(handler, EventHandler):
            name = handler.name
        else:
            raise TypeError(
                'The argument of set_event_handler must be of type event.EventHandler, function or a lambda expression')
        self.__handlers[name] = handler

    def __synchronize(self, names=None):
        """Synchronize methods in the given class.
        Only synchronize the methods whose names are
        given, or all methods if names=None."""
        for (name, val) in self.__dict__.items():
            if callable(val) and name != '__init__' and \
                    (names == None or name in names):
                # print("synchronizing", name)
                setattr(self, name, synchronized(val))

    def __add__(self, other):
        self.add_event_handler(other)
        return self

    def __getitem__(self, item):
        return self.__handlers[item]

    def __sub__(self, other):
        self.remove_event_handler(other)
        return self


class EventHandler:
    """
    Utility class to identify a callback classs with a name.

    Attributes:
        name (:obj:`str`): The name of the event handler
        func (:obj:`function`): The function which will be executed
    """

    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __eq__(self, other):
        return self.name == other.name and type(self) == type(other)

    def __hash__(self):
        return self.name.__hash__()

    def __repr__(self):
        return 'event.EventHandler(name={0}, func={1})'.format(self.name, self.func)


def synchronized(name):
    """
    Function that creates a lock object and stores in the callers __dict__. Wrappes method for synchronized execution
    :param name: name of the callable to wrap

    Returns:
        :obj:`function`: the wrapped function
    """

    def synchronized_method(method):
        outer_lock = threading.Lock()
        lock_name = "__" + method.__name__ + "_lock_" + name + "__"

        def sync_method(self, *args, **kws):
            with outer_lock:
                if not hasattr(self, lock_name): setattr(self, lock_name, threading.Lock())
                lock = getattr(self, lock_name)
                with lock:
                    return method(self, *args, **kws)

        return sync_method

    return synchronized_method
