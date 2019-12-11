# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.


import threading
import logging
import inspect
from types import FunctionType, MethodType

"""
Provides classes for a custom more feature rich signal-slot system than PyQt4's. Allows slot deletion, and explicit
handler execution.
"""

__author__ = "Dominik Gehringer, Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "December 10, 2019"

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
        elif isinstance(handler, (FunctionType, MethodType)) or inspect.ismethod(handler):
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
