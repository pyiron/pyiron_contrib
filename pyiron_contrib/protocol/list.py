# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from pyiron_contrib.protocol.generic import Vertex, PrimitiveVertex, Protocol
from pyiron_contrib.protocol.utils import InputDictionary, Pointer
import numpy as np
from abc import abstractmethod
from multiprocessing import Process, Queue
from pyiron.vasp.interactive import VaspInteractive
from pyiron.sphinx.interactive import SphinxInteractive

"""
A command class for running multiple of the same node
"""

__author__ = "Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Jul 23, 2019"


class ListVertex(PrimitiveVertex):
    """
    A base class for making wrappers to run multiple instances of the same protocol. Protocols which inherit from this
    class require the sub-protocol being copied to be passed in at initialization as an argument.

    Attributes to be assigned to the

    Attributes:
        child_type (Command): A class inheriting from ``Command`` with which to create child instances. (Passed as an
            argument at instantiation.)
        children (list): The instances of the child command to execute. These are created automatically (at run-time if
            they don't exist already).
        broadcast (InputDictionary): Input data to be split element-wise across all the child commands. (Entries here
            should always have the same number of entries as there are children, i.e. each child can have its own
            value.)
        direct (InputDictionary): Input data which is to be copied to each child (i.e. all children have the same
            value.)

    Input attributes:
        n_children (int): How many children to create.
    """

    def __init__(self, child_type, name=None):
        super(ListVertex, self).__init__(name=name)
        if not issubclass(child_type, Vertex):
            raise TypeError('CommandList children must inherit from Protocol.')
        self.child_type = child_type
        self._initialized = False
        self.direct = InputDictionary()
        self.broadcast = InputDictionary()
        self.children = None

    @abstractmethod
    def command(self, n_children):
        pass

    def finish(self):
        super(ListVertex, self).finish()
        for child in self.children:
            child.finish()

    def _initialize(self, n_children):
        children = [self.child_type(name=self.name + "_child_{}".format(n)) for n in range(n_children)]

        # Link input to input.direct
        for key in list(self.direct.keys()):
            for child in children:
                setattr(child.input, key, getattr(Pointer(self.direct), key))

        # Link input.default to input.direct.default
        for key in list(self.direct.default.keys()):
            for child in children:
                setattr(child.input.default, key, getattr(Pointer(self.direct.default), key))

        # Link input to input.broadcast
        for key in list(self.broadcast.keys()):
            for n, child in enumerate(children):
                setattr(child.input, key, getattr(Pointer(self.broadcast), key)[n])

        # Link input.default to input.broadcast.default
        for key in list(self.broadcast.default.keys()):
            for n, child in enumerate(children):
                setattr(child.input.default, key, getattr(Pointer(self.broadcast.default), key)[n])

        # Sync child archive values
        for child in children:
            child.archive.period = Pointer(self.archive).period
            child.archive.clock = Pointer(self.archive).clock

        self.children = children
        self._initialized = True

    @property
    def n_history(self):
        return self._n_history

    @n_history.setter
    def n_history(self, n_hist):
        self._n_history = n_hist
        if self.children is not None:
            for child in self.children:
                child.n_history = n_hist

    def _extract_output_data_from_children(self):
        output_keys = list(self.children[0].output.keys())  # Assumes that all the children are the same...
        if len(output_keys) > 0:
            output_data = {}
            for key in output_keys:
                values = []
                for child in self.children:
                    values.append(child.output[key][-1])
                output_data[key] = values
        else:
            output_data = None
        return output_data

    def to_hdf(self, hdf=None, group_name=None):
        super(ListVertex, self).to_hdf(hdf=hdf, group_name=group_name)
        hdf[group_name]['initialized'] = self._initialized
        if self.children is not None:
            with hdf.open(group_name + "/children") as hdf5_server:
                for n, child in enumerate(self.children):
                    child.to_hdf(hdf=hdf5_server, group_name="child" + str(n))

    def from_hdf(self, hdf=None, group_name=None):
        super(ListVertex, self).from_hdf(hdf=hdf, group_name=group_name)
        self._initialized = hdf[group_name]['initialized']
        if self._initialized:
            with hdf.open(group_name + "/children") as hdf5_server:
                children = []
                for n in np.arange(self.input.n_children, dtype=int):
                    child = self.child_type()
                    child.from_hdf(hdf=hdf5_server, group_name="child" + str(n))
                    children.append(child)
            self.children = children


class ParallelList(ListVertex):
    """
    A list of commands which are executed in in parallel. The current implementation uses multiprocessing, which creates
    a new python instance and uses pickle to communicate data to it. (That means objects modified in the new processes
    don't also modify the instance we have in the parent python instance!) This all creates some overhead, and so
    parallel is not likely to be worth calling on most

    Attributes:
        queue (multiprocessing.Queue): The queue used for collecting output from the subprocesses. (Instantiation and
            closing are handled by default in `__init__` and `finish`.)
    """
    def __init__(self, child_type, name=None):
        super(ParallelList, self).__init__(child_type, name=name)
        self.queue = Queue()

    def command(self, n_children):
        """This controls how the commands are run and is about logistics."""
        if self.children is None:
            self._initialize(n_children)

        for child in self.children:
            child.parallel_setup()

        self.queue = Queue()
        # TODO: Instead, check if the queue is open, if not open it (I know how to close, but need to look up opening)

        # Get all the commands running at once
        jobs = []
        for n, child in enumerate(self.children):
            job = Process(target=child.execute_parallel, args=(self.queue, n, child.input.resolve()))
            jobs.append(job)
            job.start()

        # And then grab their output...in order?
        for _ in jobs:
            n, child_output = self.queue.get()
            self.children[n].update_and_archive(child_output)

        # Wait for everything to finish
        for job in jobs:
            job.join()
        # (I'm not even sure that I need this)

        # Parse output as usual
        output_data = self._extract_output_data_from_children()
        return output_data

    def finish(self):
        super(ParallelList, self).finish()
        self.queue.close()


class SerialList(ListVertex):
    """
    A list of commands which are run in serial.
    """

    def command(self, n_children):
        """This controls how the commands are run and is about logistics."""
        if self.children is None:
            self._initialize(n_children)

        for child in self.children:
            child.execute()

        output_data = self._extract_output_data_from_children()
        return output_data


class AutoList(ParallelList, SerialList):
    """
    Choose between `SerialList` and `ParallelList` depending on the nature of the children.
    """

    def _is_expensive(self):
        try:
            if isinstance(
                self.children[0],
                (VaspInteractive, SphinxInteractive, Protocol)
            ):  # Whitelist types that should be treated in parallel
                return True
            else:  # Everything else is cheap enough to treat in serial
                return False
        except (AttributeError, KeyError, TypeError):
            return False

    def command(self, n_children):
        if self._is_expensive():
            return ParallelList.command(self, n_children)
        else:
            return SerialList.command(self, n_children)
