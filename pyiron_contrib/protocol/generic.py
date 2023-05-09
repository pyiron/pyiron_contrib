# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
import sys
from pyiron_base import GenericJob
from pyiron_contrib.protocol.utils import (
    IODictionary,
    InputDictionary,
    LoggerMixin,
    Event,
    EventHandler,
    Pointer,
    CrumbType,
    ordered_dict_get_last,
    Comparer,
    TimelineDict,
)

# from pyiron_contrib.protocol.utils.types import PyironJobTypeRegistry
from pyiron_contrib.protocol.utils.pptree import print_tree as pptree
from abc import ABC, abstractmethod


"""
The objective is to iterate over a directed acyclic graph of simulation instructions.
"""

__author__ = "Liam Huber, Dominik Gehringer, Jan Janssen"
__copyright__ = (
    "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Aug 16, 2019"


# defines the name, for which a subclass of Protocol will be searched in order to get a default whitelist configuration
DEFAULT_WHITE_LIST_ATTRIBUTE_NAME = "DefaultWhitelist"


class Vertex(LoggerMixin, ABC):
    """
    A parent class for objects which are valid vertices of a directed acyclic graph.

    Attributes:
        input (InputDictionary): A pointer-capable dictionary for inputs, including a sub-dictionary for defaults.
            (Default is a clean dictionary.)
        output (IODictionary): A pointer-capable dictionary for outputs. (Default is a clean dictionary.)
        archive (IODictionary): A pointer-capable dictionary for sampling the history of inputs and outputs. (Default
            is a clean dictionary.)
        vertex_state (str): Which edge to follow out of this vertex. (Default is "next".)
        possible_vertex_states (list[str]): Allowable exiting edge names. (Default is ["next"], one edge only!)
        vertex_name (str): Vertex name. (Default is None, gets set automatically when if the vertex is being added to
            a graph, or if the vertex is a Protocol being instantiated.)
        n_history (int): The length of each list stored in the output dictionary. (Default is 1, keep only the most
            recent output.)
        on (bool): Whether to execute the vertex when it is the active vertex of the graph, or simply skip over it.
            (default is True -- actually execute!)
        graph_parent (Vertex): The object who owns the graph that this vertex resides in. (Default is None.)

    Input attributes:
        default (IODictionary): A dictionary for fall-back values in case a key is requested that isn't in the main
            input dictionary.

    Archive attributes:
        whitelist (IODictionary): A nested dictionary of periods for archiving input and output values. Stores on
            executions where `clock % period = 0`.
        clock (int): The timer for whether whether or not input/output should be archived to hdf5.
    """

    def __init__(self, **kwargs):
        try:  # Super magic when the inheritance path is just vertices
            super(Vertex, self).__init__()
        except (
            TypeError
        ):  # Super magic when the inheritance path includes GenericJob (i.e. for a Protocol)
            super(Vertex, self).__init__(**kwargs)
        self.input = InputDictionary()
        self.output = IODictionary()
        self.archive = IODictionary()
        self.archive.clock = 0
        self.archive.output = IODictionary()
        self.archive.input = IODictionary()
        self.archive.whitelist = IODictionary()
        self.archive.whitelist.input = IODictionary()
        self.archive.whitelist.output = IODictionary()
        self._vertex_state = "next"
        self.possible_vertex_states = ["next"]
        self.vertex_name = None
        self.n_history = 1
        self.on = True
        self.graph_parent = None

    def get_graph_location(self):
        return self._get_graph_location()[:-1]  # Cut the trailing underscore

    def _get_graph_location(self, loc=""):
        new_loc = self.vertex_name + "_" + loc
        if self.graph_parent is None:
            return new_loc
        else:
            return self.graph_parent._get_graph_location(loc=new_loc)

    @property
    def vertex_state(self):
        return self._vertex_state

    @vertex_state.setter
    def vertex_state(self, new_state):
        if new_state not in self.possible_vertex_states:
            raise ValueError("New state not in list of possible states")
        self._vertex_state = new_state

    @abstractmethod
    def execute(self):
        """What to do when this vertex is the active vertex during graph traversal."""
        pass

    @property
    def whitelist(self):
        return {
            "input": self.archive.whitelist.input,
            "output": self.archive.whitelist.output,
        }

    @whitelist.setter
    def whitelist(self, value):
        self.set_whitelist(value)

    def set_whitelist(self, dictionary):
        """
        Sets whitelist of the current vertex. Argument defines the form:
        ```
            {'input': 5,
             'output': 1}
            # sets all keys of input to dump at every fith execution cycle
            # sets all keys of output to dump at every execution cycle
            {'input': {'structure': None,
                        'forces': 5}
            }
            # disables the archiveing of input.structure but keeps forces
        ```
        Args:
            dictionary (dict): The whitelist specification.
        """
        for k, v in dictionary.items():
            if k not in ("input", "output"):
                raise ValueError

            if isinstance(v, int):
                self._set_archive_period(k, v)
            elif isinstance(v, dict):
                self._set_archive_whitelist(k, **v)
            else:
                raise TypeError

    def _set_archive_whitelist(self, archive, **kwargs):
        """
        Whitelist properties of either "input" or "output" archive and set their dump period.

        Args:
            archive (str): either 'input' or 'output'.
            **kwargs: property names, values should be positive integers, specifies the dump freq, None = inf = < 0.

        """
        for k, v in kwargs.items():
            whitelist = getattr(self.archive.whitelist, archive)
            whitelist[k] = v

    def _set_archive_period(self, archive, n, keys=None):
        """
        Sets the archive period for each property of to "n" if keys is not specified.
        If keys is a list of property names, "n" will be set a s archiving period only for those

        Args:
            archive (str): Either 'input' or 'output'.
            n (int): Dump at every `n` steps
            keys (list of str): The affected keys

        """
        if keys is None:
            keys = list(getattr(self, archive).keys())
        self._set_archive_whitelist(archive, **{k: n for k in keys})

    def set_input_archive_period(self, n, keys=None):
        self._set_archive_period("input", n, keys=keys)

    def set_output_archive_period(self, n, keys=None):
        self._set_archive_period("output", n, keys=keys)

    def set_input_whitelist(self, **kwargs):
        self._set_archive_whitelist("input", **kwargs)

    def set_output_whitelist(self, **kwargs):
        self._set_archive_whitelist("output", **kwargs)

    def set_archive_period(self, n):
        self.set_input_archive_period(n)
        self.set_output_archive_period(n)

    def _update_archive(self):
        # Update input
        history_key = "t_%s" % self.archive.clock

        for key, value in self.input.items():
            if key in self.archive.whitelist.input:
                # the keys there, but it could be explicitly set to < 0 or None
                period = self.archive.whitelist.input[key]
                if period is not None and period >= 0:
                    # TODO: Notifaction when whitelist contains items which are not items of input
                    # check if the period matches that of the key
                    if self.archive.clock % period == 0:
                        if key not in self.archive.input:
                            self.archive.input[key] = TimelineDict()
                            self.archive.input[key][history_key] = value
                        else:
                            # we want to archive it only if there is a change, thus get the last element
                            last_val = ordered_dict_get_last(self.archive.input[key])
                            if not Comparer(last_val) == value:
                                self.archive.input[key][history_key] = value
                                self.logger.info(
                                    'Property "{}" did change in input ({} -> {})'.format(
                                        key, last_val, value
                                    )
                                )
                            else:
                                self.logger.info(
                                    'Property "{}" did not change in input'.format(key)
                                )

        # Update output
        for key, value in self.output.items():
            if key in self.archive.whitelist.output:
                period = self.archive.whitelist.output[key]
                if period is not None and period >= 0:
                    # TODO: Notifaction when whitelist contains items which are not items of input
                    # check if the period matches that of the key
                    if self.archive.clock % period == 0:
                        val = value[-1]
                        if key not in self.archive.output:
                            self.archive.output[key] = TimelineDict()
                            self.archive.output[key][history_key] = val
                        else:
                            # we want to archive it only if there is a change, thus get the last element
                            last_val = ordered_dict_get_last(self.archive.output[key])
                            if not Comparer(last_val) == val:
                                self.archive.output[key][history_key] = val
                            else:
                                self.logger.info(
                                    'Property "{}" did not change in input'.format(key)
                                )

    def _update_output(self, output_data):
        if output_data is None:
            return

        for key, value in output_data.items():
            if key not in self.output:
                self.output[key] = [value]
            else:
                history = list(self.output[key])
                # Roll the list if it is necessary
                history.append(value)
                if len(history) > self.n_history:
                    # Remove the head of the queue
                    history.pop(0)
                self.output[key] = history

    def update_and_archive(self, output_data):
        self._update_output(output_data)
        self._update_archive()

    def finish(self):
        pass

    def parallel_setup(self):
        """How to prepare to execute in parallel when there's a list of these vertices together."""
        pass

    def to_hdf(self, hdf, group_name=None):
        """
        Store the Vertex in an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object.
            group_name (str): HDF5 subgroup name. (Default is None.)
        """
        if group_name is not None:
            hdf5_server = hdf.open(group_name)
        else:
            hdf5_server = hdf

        hdf5_server["TYPE"] = str(type(self))
        hdf5_server["possiblevertexstates"] = self.possible_vertex_states
        hdf5_server["vertexstate"] = self.vertex_state
        hdf5_server["vertexname"] = self.vertex_name
        hdf5_server["nhistory"] = self.n_history
        self.input.to_hdf(hdf=hdf5_server, group_name="input")
        self.output.to_hdf(hdf=hdf5_server, group_name="output")
        self.archive.to_hdf(hdf=hdf5_server, group_name="archive")

    def from_hdf(self, hdf, group_name=None):
        """
        Load the Vertex from an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object.
            group_name (str): HDF5 subgroup name. (Default is None.)
        """
        if group_name is not None:
            hdf5_server = hdf.open(group_name)
        else:
            hdf5_server = hdf

        self.possible_vertex_states = hdf5_server["possiblevertexstates"]
        self._vertex_state = hdf5_server["vertexstate"]
        self.vertex_name = hdf5_server["vertexname"]
        self.n_history = hdf5_server["nhistory"]
        self.input.from_hdf(hdf=hdf5_server, group_name="input")
        self.output.from_hdf(hdf=hdf5_server, group_name="output")
        self.archive.from_hdf(hdf=hdf5_server, group_name="archive")

        # sort the dictionaries after loading, do it for both input and output dictionaries
        for archive_name in ("input", "output"):
            archive = getattr(self.archive, archive_name)
            for key in archive.keys():
                history = archive[key]
                # create an ordered dictionary from it, convert it to integer back again
                archive[key] = TimelineDict(
                    sorted(
                        history.items(), key=lambda item: int(item[0].replace("t_", ""))
                    )
                )


class PrimitiveVertex(Vertex):
    """
    Vertices which do not contain their a sub-graph but directly produce output from input.
    """

    def execute(self):
        """Just parse the input and do your physics, then store the output."""
        output_data = self.command(**self.input.resolve())
        self.update_and_archive(output_data)

    @abstractmethod
    def command(self, *args, **kwargs):
        """The command method controls the physics"""
        pass

    def execute_parallel(self, n, return_dict):
        """How to execute in parallel when there's a list of these vertices together."""
        output_data = self.command(**self.input.resolve())
        return_dict[n] = output_data


class CompoundVertex(Vertex):
    """
    Vertices which contain a graph and produce output only after traversing their graph to its exit point.

    Input:
        graph (Graph): The graph of vertices to traverse.
        protocol_finished (Event):
        protocol_started (Event):
        vertex_processing (Event):
        vertex_processed (Event):
        finished (bool):
    """

    def __init__(self, **kwargs):
        super(CompoundVertex, self).__init__(**kwargs)

        self.graph = Graph()
        self.graph.owner = self

        # Initialize event system
        self.protocol_finished = Event()
        self.protocol_started = Event()
        self.vertex_processing = Event()
        self.vertex_processed = Event()

        # Set up the graph
        self.define_vertices()
        self.define_execution_flow()
        self.define_information_flow()

        # On initialization, set the active verex to starting vertex
        self.graph.active_vertex = self.graph.starting_vertex

        self.finished = False

        self.restore_default_whitelist()

    @property
    def default_whitelist(self):
        # set the default whitelist -> check if the class has an attribute, otherwise skip the configuration
        cls = type(self)
        if hasattr(cls, DEFAULT_WHITE_LIST_ATTRIBUTE_NAME):
            return getattr(cls, DEFAULT_WHITE_LIST_ATTRIBUTE_NAME)
        else:
            return None

    @abstractmethod
    def define_vertices(self):
        """Add child vertices to the graph."""
        pass

    @abstractmethod
    def define_execution_flow(self):
        """Wire the logic for traversing the graph edges."""
        pass

    @abstractmethod
    def define_information_flow(self):
        """Connect input and output information inside the graph. Also set the archive clock for all vertices."""
        pass

    @abstractmethod
    def get_output(self):
        """
        Define the output dictionary to be returned when the graph traversal completes. This synchronizes the
        behaviour of primitive vertices and compound vertices when they themselves are the child vertex in another
        graph.
        """
        pass

    def execute(self):
        """Traverse graph until the active vertex is None."""
        # Subscribe graph vertices to the protocol_finished Event
        for vertex_name, vertex in self.graph.vertices.items():
            handler_name = "{}_close_handler".format(vertex_name)
            if not self.protocol_finished.has_handler(handler_name):
                self.protocol_finished += EventHandler(handler_name, vertex.finish)

        # Run the graph
        if self.graph.active_vertex is None:
            self.graph.active_vertex = self.graph.starting_vertex
        self.protocol_started.fire()
        while self.graph.active_vertex is not None:
            vertex_on = self.graph.active_vertex.on
            if isinstance(vertex_on, Pointer):
                vertex_on = ~vertex_on
            if not vertex_on:
                self.logger.info(
                    'Skipping vertex "{}":{}'.format(
                        self.graph.active_vertex.vertex_name,
                        type(self.graph.active_vertex).__name__,
                    )
                )
                self.graph.step()
            self.vertex_processing.fire(self.graph.active_vertex)
            self.graph.active_vertex.execute()
            self.vertex_processed.fire(self.graph.active_vertex)
            self.graph.step()
        self.graph.active_vertex = self.graph.restarting_vertex
        self.update_and_archive(self.get_output())

    def execute_parallel(self, n, all_child_output):
        """How to execute in parallel when there's a list of these vertices together."""
        self.execute()
        all_child_output[n] = self.get_output()

    def set_graph_archive_clock(self, clock, recursive=False):
        for _, vertex in self.graph.vertices.items():
            vertex.archive.clock = clock
            if recursive and isinstance(vertex, CompoundVertex):
                vertex.set_graph_archive_clock(clock, recursive=True)

    def to_hdf(self, hdf, group_name=None):
        """
        Store the Protocol in an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object.
            group_name (str): HDF5 subgroup name - optional
        """
        super(CompoundVertex, self).to_hdf(hdf=hdf, group_name=group_name)
        self.graph.to_hdf(hdf=hdf, group_name="graph")

    def from_hdf(self, hdf=None, group_name=None):
        """
        Load the Protocol from an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(CompoundVertex, self).from_hdf(
            hdf=hdf, group_name=group_name or self.vertex_name
        )
        with hdf.open(self.vertex_name) as hdf5_server:
            self.graph.from_hdf(hdf=hdf5_server, group_name="graph")
        self.define_information_flow()  # Rewire pointers

    def visualize(self, execution=True, dataflow=True):
        return self.graph.visualize(
            self.fullname(), execution=execution, dataflow=dataflow
        )

    @property
    def whitelist(self):
        return {
            vertex_name: vertex.whitelist
            for vertex_name, vertex in self.graph.vertices.items()
        }

    @whitelist.setter
    def whitelist(self, value):
        self.set_whitelist(value)

    def _set_archive_period(self, archive, n, keys=None):
        """
        In constrast to Vertex._set_archive_period, this calls "n". if keys=None it will be applied to all vertices
        Args:
            archive: (str) input or output
            n: (int) dump every "n" steps
            keys: (list of str) the vertex names which the it should be applied to (default: None)
        """
        if keys is None:
            keys = self.graph.vertices.keys()

        for key in keys:
            vertex = self.graph.vertices[key]
            vertex._set_archive_period(archive, n)

    def set_input_archive_period(self, n, keys=None):
        self._set_archive_period("input", n, keys=keys)

    def set_output_archive_period(self, n, keys=None):
        self._set_archive_period("output", n, keys=keys)

    def _set_archive_whitelist(self, archive, **kwargs):
        """
        Sets a whitelist configuration for this protocol
        Args:
            archive: (str) "input" or "output" the archive the whitelist should be applied to
            **kwargs: vertex_name = dict the whitelist configuration
        """
        for key, value in kwargs.items():
            if key not in self.graph.vertices:
                self.logger.warning(
                    'Cannot set the whitelist of vertex "%s" since it is no a part of protocol "%s'
                    % (key, self.vertex_name)
                )
                continue
            vertex = self.graph.vertices[key]
            vertex._set_archive_whitelist(archive, **value)

    def set_input_whitelist(self, **kwargs):
        self._set_archive_whitelist("input", **kwargs)

    def set_output_whitelist(self, **kwargs):
        self._set_archive_whitelist("output", **kwargs)

    def set_whitelist(self, dictionary):
        """
        Sets the whitelist for this protocol. The first level of keys must contain valid vertex ids. (A warning will
        be printed otherwise.
            1. Level one: graph vertex ids -> specifies to which vertices the nested whitelist will be applied to
            2. Level two: "input" or "output" -> defines which dictionary is affected by the archiving periods
            3. Level three: the properties of the corresponding dictionary which will be affected
        Example:
        ```
            { 'calc_static' : {
                'input': 5 #sets all keys of input dict to 5,
                'output': {
                    'energy_pot': 1,
                    'structure': None
                }
        ```

        Args:
            dictionary: (dict) the whitelist configuration
        """

        for key, vertex_dict in dictionary.items():
            if key not in self.graph.vertices:
                self.logger.warning(
                    'Cannot set the whitelist of vertex "%s" since it is no a part of protocol "%s'
                    % (key, self.vertex_name)
                )
                continue
            vertex = self.graph.vertices[key]
            # call it for each vertex
            vertex.set_whitelist(vertex_dict)

    def set_archive_period(self, n):
        """
        Sets the archive period for all key for both input and output dictionaries
        Args:
            n: (int) the archiving period
        """
        self.set_input_archive_period(n)
        self.set_output_archive_period(n)

    def restore_default_whitelist(self):
        """
        If the protcol type has an attribute DefaultWhitelist, it will be restored
        """
        if self.default_whitelist is not None:
            # first we have to clear again all white lists
            for vertex_name, vertex in self.graph.vertices.items():
                vertex.archive.whitelist.input.clear()
                vertex.archive.whitelist.output.clear()
            self.whitelist = self.default_whitelist
            self.logger.debug(
                'Whitelist configured for protocol "%s"' % self.vertex_name
            )

    def format_whitelist(self, format="tree", file=sys.stdout):
        if format == "code":
            start = [self.vertex_name, "graph"]
            path_format = "{path} = {value}\n"
            for vertex_name, conf in self.whitelist.items():
                vertex_path = start + [vertex_name, "archive", "whitelist"]
                for archive, vertex_conf in conf.items():
                    archive_path = vertex_path + [archive]
                    for key, value in vertex_conf.items():
                        key_path = archive_path + [key]
                        file.write(
                            path_format.format(path=".".join(key_path), value=value)
                        )
        elif format == "simple":
            from pyiron_contrib.protocol.utils.pptree import count_paths

            def print_tree(node, level=0):
                if not isinstance(node, dict):
                    return "%s\n" % str(node)
                elif count_paths(node) == 0:
                    return ""
                else:
                    output = ""
                    for key, val in node.items():
                        # only print the path if there is a leaf node
                        if count_paths(val) > 0:
                            output += "%s%s:" % ("\t" * level, key)
                            # now lets do the second part of the comma
                            # add an extra \n if val is a dectionary
                            output += "\n" if isinstance(val, dict) else " "
                            output += print_tree(val, level=level + 1)
                    return output

            # print the result
            file.write(print_tree(self.whitelist))
        elif format == "tree":
            # print the nice tree
            pptree(
                self.whitelist,
                file=file,
                name="%s.%s" % (self.vertex_name, "whitelist"),
            )


class Protocol(CompoundVertex, GenericJob):
    """
    A parent class for compound vertices which are being instantiated as regular pyiron jobs, i.e. the highest level
    graph in their context.
    Example: if `X` inherits from `CompoundVertex` and performs the desired logic, then
    ```
    class ProtocolX(Protocol, X):
        pass
    ```
    can be added to the `pyiron_contrib`-level `__init__` file and jobs performing X-logic can be instantiated with
    in a project `pr` with the name `job_name` using `pr.create_job(pr.job_type.ProtocolX, job_name)`.
    """

    def __init__(self, project=None, job_name=None):
        super(Protocol, self).__init__(project=project, job_name=job_name)
        self.vertex_name = job_name

    def execute(self):
        super(Protocol, self).execute()

    def run_static(self):
        """If this CompoundVertex is the highest level, it can be run as a regular pyiron job."""
        self.status.running = True
        self.execute()
        self.status.collect = True  # Assume modal for now
        self.protocol_finished.fire()
        self.run()  # This is an artifact of inheriting from GenericJob, to get all that run functionality

    def run(
        self,
        delete_existing_job=False,
        repair=False,
        debug=False,
        run_mode=None,
        continue_run=False,
    ):
        """A wrapper for the run which allows us to simply keep going with a new variable `continue_run`"""
        if continue_run:
            self.status.created = True
        super(CompoundVertex, self).run(
            delete_existing_job=delete_existing_job,
            repair=repair,
            debug=debug,
            run_mode=run_mode,
        )

    def collect_output(self):
        # Dear Reader: This feels like a hack, but it works. Sincerely, -Liam
        self.to_hdf()

    def write_input(self):
        # Dear Reader: I looked at base/master/list and /parallel where this appears, but it's still not clear to me
        # what I should be using this for. But, I get a NotImplementedError if I leave it out, so here it is. -Liam
        pass

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the Protocol in an HDF5 file.
        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        if hdf is None:
            hdf = self.project_hdf5
        super(CompoundVertex, self).to_hdf(hdf=hdf, group_name=group_name)

    def from_hdf(self, hdf=None, group_name=None):
        """
        Load the Protocol from an HDF5 file.
        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        if hdf is None:
            hdf = self.project_hdf5
        super(CompoundVertex, self).from_hdf(hdf=hdf, group_name=group_name)


class Graph(dict, LoggerMixin):
    """
    A directed graph of vertices and edges, and a method for iterating through the graph.

    Vertices and edges are the graph are explicitly stored as child classes inheriting from `dict` so that all
    'graphiness' is fully decoupled from the objects sitting at the vertices.

    Attributes:
        vertices (Vertices): Vertices of the graph.
        edges (Edges): Directed edges between the vertices.
        starting_vertex (Vertex): The element of `vertices` for the graph to begin on.
        active_vertex (Vertex): The element of `vertices` for the vertex the graph iteration is on.
        restarting_vertex (Vertex): The element of `vertices` for the graph to restart on if the graph has been loaded.
    """

    def __init__(self, **kwargs):
        super(Graph, self).__init__(**kwargs)
        self.vertices = Vertices()
        self.edges = Edges()
        self.starting_vertex = None
        self.active_vertex = None
        self.restarting_vertex = None
        self.owner = None

    def __setattr__(self, key, val):
        if key == "vertices":
            if not isinstance(val, Vertices):
                raise ValueError("'vertices' is a protected attribute for graphs.")
            self[key] = val
        elif key == "edges":
            if not isinstance(val, Edges):
                raise ValueError("'edges' is a protected attribute for graphs.")
            self[key] = val
        elif key in ["active_vertex", "starting_vertex", "restarting_vertex"]:
            if val is None or isinstance(val, Vertex):
                self[key] = val
            else:
                raise ValueError(
                    "The active, starting, and restarting vertices must inherit `Vertex` or be `None`."
                )
        elif key == "owner":
            if not (isinstance(val, CompoundVertex) or val is None):
                raise ValueError(
                    "Only protocols can hold graphs, but the assigned owner has type",
                    type(val),
                )
            else:
                self[key] = val
        elif isinstance(val, Vertex):
            val.vertex_name = key
            val.graph_parent = self.owner
            self.vertices[key] = val
            self.edges.initialize(val)
        else:
            raise TypeError("Graph vertices must inherit from `Vertex`")

    def __getattr__(self, name):
        try:
            return self["vertices"][name]
        except KeyError:
            try:
                return self[name]
            except KeyError:
                return object.__getattribute__(self, name)

    def visualize(self, protocol_name, execution=True, dataflow=True):
        """
        Plot a visual representation of the graph.

        Args:
            protocol_name:
            execution (bool): Show the lines dictating the flow of graph traversal.
            dataflow (bool): Show the lines dictating where vertex input comes from.

        Returns:
            (graphviz.Digraph) The image representation of the protocols workflow
        """
        try:
            from graphviz import Digraph
        except ImportError as import_error:
            self.logger.exception(
                'Failed to import "graphviz" package', exc_info=import_error
            )
            return

        # Create graph object
        workflow = Digraph(comment=protocol_name)

        # Define styles for the individual classes
        class_style_mapping = {
            CompoundVertex: {"shape": "box"},
            # CommandBool: {'shape': 'diamond'},
            PrimitiveVertex: {"shape": "circle"},
        }

        def resolve_type(type_):
            if type_ in class_style_mapping.keys():
                return type_
            else:
                parents = [
                    key for key in class_style_mapping.keys() if issubclass(type_, key)
                ]
                if len(parents) == 0:
                    raise TypeError(
                        'I do not know how to visualize "{}"'.format(type_.__name__)
                    )
                elif len(parents) > 1:
                    self.logger.warn(
                        'More than one parent class found for type "{}"'.format(
                            type_.__name__
                        )
                    )
                return parents[0]

        for vertex_name, vertex in self.vertices.items():
            vertex_type = type(vertex)
            vertex_type_style = class_style_mapping[resolve_type(vertex_type)]

            node_label = """<<B>{vertex_type}</B><BR/>{vertex_name}>"""
            node_label = node_label.format(
                vertex_type=vertex_type.__name__, vertex_name=vertex_name
            )
            if self.active_vertex == vertex:
                # Highlight active vertex
                highlight = {
                    "style": "filled",
                    "color": "green",
                }
            else:
                highlight = {}

            # Highlight the active vertex in green color
            highlight.update(vertex_type_style)
            workflow.node(vertex_name, label=node_label, **highlight)
        # Add end node
        workflow.node(
            "end", "END", **{"shape": "doublecircle", "style": "filled", "color": "red"}
        )
        protocol_input_node = None

        if execution:
            for vertex_start, edges in self.edges.items():
                for vertex_state, vertex_end in edges.items():
                    if vertex_end is None:
                        vertex_end = "end"
                    workflow.edge(vertex_start, vertex_end, label=vertex_state)
        if dataflow:
            dataflow_edge_style = {
                "style": "dotted",
                "color": "blue",
                "labelfontcolor": "blue",
                "labelangle": "90",
            }
            for vertex_name, vertex in self.vertices.items():
                items = super(InputDictionary, vertex.input).items()
                for key, value in items:
                    if isinstance(value, Pointer):
                        vertex_end = self._edge_from_pointer(key, value)
                        if vertex_end is not None:
                            if isinstance(vertex_end, Vertex):
                                workflow.edge(
                                    vertex_end.vertex_name,
                                    vertex_name,
                                    label=key,
                                    **dataflow_edge_style
                                )
                            elif isinstance(vertex_end, (IODictionary, Vertex)):
                                self.logger.warning(
                                    "vertex_end is IODIctionary() I have to decide what to do"
                                )
                                if protocol_input_node is None:
                                    # Initialize a node for protocol level input
                                    protocol_input_node = workflow.node(
                                        "protocol_input",
                                        "{}.input".format(protocol_name),
                                        style="filled",
                                        shape="folder",
                                    )
                                workflow.edge(
                                    "protocol_input",
                                    vertex_name,
                                    label=key,
                                    **dataflow_edge_style
                                )
                            else:
                                pass

        return workflow

    def _edge_from_pointer(self, key, p):
        assert isinstance(p, Pointer)
        path = p.path.copy()
        root = path.pop(0)

        result = root.object
        while len(path) > 0:
            crumb = path.pop(0)
            crumb_type = crumb.crumb_type
            crumb_name = crumb.name

            if isinstance(result, (Vertex, IODictionary)):
                return result

            # If the result is a pointer itself we have to resolve it first
            if isinstance(result, Pointer):
                self.logger.info("Resolved pointer in a pointer path")
                result = ~result
            if isinstance(crumb_name, Pointer):
                self.logger.info("Resolved pointer in a pointer path")
                crumb_name = ~crumb_name
            # Resolve it with the correct method - dig deeper
            if crumb_type == CrumbType.Attribute:
                try:
                    result = getattr(result, crumb_name)
                except AttributeError as e:
                    raise e
            elif crumb_type == CrumbType.Item:
                try:
                    result = result.__getitem__(crumb_name)
                except (TypeError, KeyError) as e:
                    raise e

        # If we reached this point we have no Command at all, give a warning
        self.logger.warning(
            'I could not find a graph in the pointer {} for key "{}"'.format(
                p.path, key
            )
        )
        return None

    def step(self):
        """
        Follows the edge out of the active vertex to get the name of the next vertex and set it as the active vertex.
        If the active vertex has multiple possible states, the outbound edge for the current state will be chosen.

        Returns:
            (str) The name of the next vertex.
        """
        vertex = self.active_vertex
        if vertex is not None:
            state = vertex.vertex_state
            next_vertex_name = self.edges[vertex.vertex_name][state]

            if next_vertex_name is None:
                self.active_vertex = None
            else:
                self.active_vertex = self.vertices[next_vertex_name]

    def make_edge(self, start, end, state="next"):
        """
        Makes a directed edge connecting two vertices.

        Args:
            start (Vertex): The vertex for the edge to start at.
            end (Vertex): The vertex for the edge to end at.
            state (str): The state for the vertex to be in when it points to this particular end. (Default, "next", is
                the parent-level state for vertices without multiple outbound edges.)
        """
        if start.vertex_name not in self.vertices.keys():
            raise ValueError(
                "The vertex {} was not found among graph vertices, {}".format(
                    start.vertex_name, self.vertices.keys()
                )
            )

        if state not in start.possible_vertex_states:
            raise ValueError(
                "{} not found in possible vertex states for {}, {}".format(
                    state, start.vertex_name, start.possible_vertex_states
                )
            )

        if end is not None:
            if end.vertex_name not in self.vertices.keys():
                raise ValueError(
                    "{} is not among vertices, {}".format(
                        end.vertex_name, self.vertices.keys()
                    )
                )
            self.edges[start.vertex_name][state] = end.vertex_name
        else:
            self.edges[start.vertex_name][state] = None

    def make_pipeline(self, *args):
        """
        Adds an edge between every argument, in the order they're given. The edge is added for the vertex state "next",
        so this is only appropriate for vertices which don't have a non-trivial `vertex_state`.

        Args:
            *args (Vertex/str): Vertices to connect in a row, or the state connecting two vertices.
        """
        for n, vertex in enumerate(args[:-1]):
            if isinstance(vertex, str):
                continue
            next_vertex = args[n + 1]
            if isinstance(next_vertex, str):
                state = next_vertex
                next_vertex = args[n + 2]
            else:
                state = "next"
            self.make_edge(vertex, next_vertex, state=state)

    def to_hdf(self, hdf, group_name="graph"):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server["TYPE"] = str(type(self))
            hdf5_server["startingvertexname"] = self.starting_vertex.vertex_name
            if self.active_vertex is not None:
                hdf5_server["activevertexname"] = self.active_vertex.vertex_name
            else:
                hdf5_server["activevertexname"] = None
            if self.restarting_vertex is None:
                hdf5_server["restartingvertexname"] = self.starting_vertex.vertex_name
            else:
                hdf5_server["restartingvertexname"] = self.restarting_vertex.vertex_name
            self.vertices.to_hdf(hdf=hdf5_server, group_name="vertices")
            self.edges.to_hdf(hdf=hdf5_server, group_name="edges")

    def from_hdf(self, hdf, group_name="graph"):
        with hdf.open(group_name) as hdf5_server:
            active_vertex_name = hdf5_server["activevertexname"]
            self.vertices.from_hdf(hdf=hdf5_server, group_name="vertices")
            self.edges.from_hdf(hdf=hdf5_server, group_name="edges")
            self.starting_vertex = self.vertices[hdf5_server["startingvertexname"]]
            self.restarting_vertex = self.vertices[hdf5_server["restartingvertexname"]]
        if active_vertex_name is not None:
            self.active_vertex = self.vertices[active_vertex_name]
        else:
            self.active_vertex = self.restarting_vertex


class Vertices(dict):
    """
    A dictionary of vertices whose keys are the vertex name.
    """

    def __init__(self, **kwargs):
        super(Vertices, self).__init__(**kwargs)

    def __setattr__(self, key, val):
        if not isinstance(val, Vertex):
            raise ValueError
        self[key] = val

    def __getattr__(self, item):
        return self[item]

    def to_hdf(self, hdf, group_name="vertices"):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server["TYPE"] = str(type(self))
            for name, vertex in self.items():
                if isinstance(vertex, Vertex):
                    vertex.to_hdf(hdf=hdf5_server, group_name=name)
                else:
                    raise TypeError("Cannot save non-Vertex-like vertices")

    def from_hdf(self, hdf, group_name="vertices"):
        with hdf.open(group_name) as hdf5_server:
            for name, vertex in self.items():
                if isinstance(vertex, (CompoundVertex, Vertex)):
                    vertex.from_hdf(hdf=hdf5_server, group_name=name)
                else:
                    raise TypeError("Cannot load non-Vertex-like vertices")


class Edges(dict):
    """
    A collection of dictionaries connecting each state of a given vertex to another vertex.
    """

    def __init__(self, **kwargs):
        super(Edges, self).__init__(**kwargs)

    def __setattr__(self, key, val):
        self[key] = val

    def __getattr__(self, item):
        return self[item]

    def initialize(self, vertex):
        """
        Set an outbound edge to `None` for each allowable vertex state.

        Args:
            vertex (Vertex): The vertex to assign an edge to.
        """
        name = vertex.vertex_name
        if isinstance(vertex, Vertex):
            self[name] = {}
            for state in vertex.possible_vertex_states:
                self[name][state] = None
        else:
            raise TypeError("Vertices must inherit from `Vertex` .")

    def to_hdf(self, hdf, group_name="edges"):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server["TYPE"] = str(type(self))
            for name, edge in self.items():
                hdf5_server[name] = edge

    def from_hdf(self, hdf, group_name):
        return
