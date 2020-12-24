# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from pyiron_contrib.protocol.generic import Vertex, PrimitiveVertex, CompoundVertex
from pyiron_contrib.protocol.utils import InputDictionary, Pointer
import numpy as np
import time
from abc import abstractmethod
from multiprocessing import Process, Manager
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

    def __init__(self, child_type):
        if not issubclass(child_type, Vertex):
            raise TypeError('ListVertex children must inherit from Protocol.')
        self.children = None  # Ahead of super so the n_history call doesn't trigger the setter and find no children
        super(ListVertex, self).__init__()
        self.child_type = child_type
        self._initialized = False
        self.direct = InputDictionary()
        self.broadcast = InputDictionary()
        self._n_history = None
        self.n_history = 1

    @abstractmethod
    def command(self, n_children):
        pass

    def finish(self):
        for child in self.children:
            child.finish()
        super(ListVertex, self).finish()

    def _initialize(self, n_children):
        children = [self.child_type(name="child_{}".format(n)) for n in range(n_children)]

        # Locate children in graph
        for n, child in enumerate(children):
            child.graph_parent = self
            child.vertex_name = "child_{}".format(n)

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
                    child = self.child_type(name="child_{}".format(n))
                    child.from_hdf(hdf=hdf5_server, group_name="child" + str(n))
                    child.graph_parent = self
                    children.append(child)
            self.children = children


class ParallelList(ListVertex):
    """
    A list of commands which are executed in in parallel. The current implementation uses multiprocessing.Pool.

    Attributes:
        pool (multiprocessing.Pool): Define the number of workers that will be utilized to run the child jobs.
            Instantiation and closing are handled by default in `__init__` and `finish`.

        Note: It is best to set the number of workers to the number of cores so as to prevent larger computation
            times due to subprocess communication between the large number of workers in a single core.
    """

    def __init__(self, child_type, sleep_time=0):
        super(ParallelList, self).__init__(child_type)
        self.sleep_time = sleep_time

    def command(self, n_children):
        """This controls how the commands are run and is about logistics."""
        if self.children is None:
            self._initialize(n_children)

        for child in self.children:
            child.parallel_setup()

        start_time = time.time()
        sleep_time = ~self.sleep_time

        all_child_output = Manager().dict()

        jobs = []
        for i, child in enumerate(self.children):
            job = Process(target=child.execute_parallel, args=(i, all_child_output))
            job.start()
            time.sleep(sleep_time)
            jobs.append(job)

        for job in jobs:
            job.join()
            time.sleep(sleep_time)

        print(all_child_output.keys())

        ordered_child_output = dict.fromkeys(range(len(all_child_output)))
        for i in range(len(all_child_output)):
            ordered_child_output[i] = all_child_output[i]

        output_keys = list(ordered_child_output[0].keys())  # Assumes that all the children are the same...
        if len(output_keys) > 0:
            output_data = {}
            for key in output_keys:
                values = []
                for child in ordered_child_output.values():
                    values.append(child[key])
                output_data[key] = values
        else:
            output_data = None

        stop_time = time.time()
        print('Time elapsed :', stop_time - start_time)

        return output_data

    def finish(self):
        super(ParallelList, self).finish()


class SerialList(ListVertex):
    """
    A list of commands which are run in serial.
    """
    def __init__(self, child_type):
        super(SerialList, self).__init__(child_type=child_type)

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
    def __init__(self, child_type):
        super(AutoList, self).__init__(child_type=child_type)

    def _is_expensive(self):
        try:
            if isinstance(
                self.children[0],
                (VaspInteractive, SphinxInteractive, CompoundVertex)
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
