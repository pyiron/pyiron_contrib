import abc
import logging
from typing import Optional

from pyiron_contrib.tinybase.node import AbstractNode
from pyiron_contrib.tinybase.storage import (
        Storable,
        GenericStorage,
        pickle_load,
        pickle_dump
)
from pyiron_contrib.tinybase.executor import (
        Executor,
        BackgroundExecutor,
        ProcessExecutor
)
from pyiron_contrib.tinybase.database import (
    DatabaseEntry
)
from pyiron_contrib.tinybase.project import ProjectInterface, ProjectAdapter
from pyiron_base.state import state


class TinyJob(Storable, abc.ABC):
    """
    A tiny job unifies an executor, a node and its output.

    The job adds the node to the database and persists its input and output in a storage location.

    The input of the node is available from :attr:`~.input`. After the job has finished the output of the node can be
    accessed from :attr:`~.output` and the data written to storage from :attr:`.~storage`.

    This is an abstracat base class that works with any execution node without specifying it.  To create specialized
    jobs you can derive from it and overload :meth:`._get_node()` to return an instance of the node, e.g.

    >>> from somewhere import MyNode
    >>> class MyJob(TinyJob):
    ...     def _get_node(self):
    ...         return MyNode()

    The return value of :meth:`._get_node()` is persisted during the life time of the job.

    You can use :class:`.GenericTinyJob` to dynamically specify which node the job should execute.
    """

    _executors = {
            'foreground': Executor,
            'background': BackgroundExecutor,
            'process': ProcessExecutor
    }

    def __init__(self, project: ProjectInterface, job_name: str):
        """
        Create a new job.

        If the given `job_name` is already present in the `project` it is reloaded.  No checks are performed that the
        node type of the already present job and the current one match.  This is also not always necessary, e.g. when
        reloading a :class:`.GenericTinyJob` it will automatically read the correct node from storage.

        Args:
            project (:class:`.ProjectInterface`): the project the job should live in
            job_name (str): the name of the job.
        """
        if not isinstance(project, ProjectInterface):
            project = ProjectAdapter(project)
        self._project = project
        self._name = job_name
        self._node = None
        self._output = None
        self._storage = None
        self._executor = None
        self._id = None
        # FIXME: this should go into the job creation logic on the project
        if project.exists_storage(job_name):
            try:
                self.load()
            except Exception as e:
                raise RuntimeError(f"Failed to reload run job from storage: {e}") from None

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    def _update_id(self):
        self._id = self.project.get_job_id(self.name)

    @property
    def project(self):
        return self._project

    @abc.abstractmethod
    def _get_node(self) -> AbstractNode:
        """
        Return an instance of the :class:`.AbstractNode`.

        The value return from here is saved automatically in :prop:`.node`.
        """
        pass

    @property
    def node(self):
        if self._node is None:
            self._node = self._get_node()
        return self._node

    @property
    def jobtype(self):
        return self.node.__class__.__name__

    @property
    def input(self):
        return self.node.input

    @property
    def output(self):
        if self._output is not None:
            return self._output
        else:
            raise RuntimeError("Job not finished yet!")

    @property
    def storage(self):
        if self._storage is None:
            self._storage = self._project.create_storage(self.name)
        return self._storage

    def _set_output(self, data):
        self._output = data["output"][0]

    def _setup_executor_callbacks(self):
        self._executor._run_machine.observe("ready", lambda _: self.store(self.storage))
        self._executor._run_machine.observe("finished", self._set_output)
        self._executor._run_machine.observe("finished", lambda _: self.store(self.storage))

        self._executor._run_machine.observe("ready", self._add_to_database)
        self._executor._run_machine.observe("running", self._update_status("running"))
        self._executor._run_machine.observe("collect", self._update_status("collect"))
        self._executor._run_machine.observe("finished", self._update_status("finished"))

    def run(self, how='foreground') -> Optional[Executor]:
        """
        Start execution of the job.

        If the job already has a database id and is not in "ready" state, do nothing.

        Args:
            how (string): specifies which executor to use

        Returns:
            :class:`.Executor`: the executor that is running the node or nothing.
        """
        if self._id is None or self.project.database.get_item(self.id).status == "ready":
            exe = self._executor = self._executors[how](nodes=[self.node])
            self._setup_executor_callbacks()
            exe.run()
            return exe
        else:
            logging.info("Job already finished!")

    def remove(self):
        """
        Remove the job from the database and storage.

        Resets the internal state so that the job could be re-run from the same instance.
        """
        self.project.remove(self.id)
        self._id = None
        self._output = None
        self._storage = None
        self._executor = None

    def _add_to_database(self, _data):
        if self._id is None:
            entry = DatabaseEntry(
                    name=self.name,
                    project=self.project.path,
                    username=state.settings.login_user,
                    status="ready",
                    jobtype=self.jobtype,
            )
            self._id = self.project.database.add_item(entry)
        return self.id

    def _update_status(self, status):
        return lambda _data: self.project.database.update_status(self.id, status)

    # Storable Impl'
    def _store(self, storage):
        # preferred solution, but not everything that can be pickled can go into HDF atm
        # self._executor.output[-1].store(storage, "output")
        storage["node"] = pickle_dump(self.node)
        if self._output is not None:
            storage["output"] = pickle_dump(self._output)

    def load(self, storage: GenericStorage = None):
        """
        Load job from storage and reload id from database.

        If `storage` is not given, use the default provided by the project.

        Args:
            storage (:class:`.GenericStorage`): where to read from
        """
        self._update_id()
        if storage is None:
            storage = self.storage
        self._node = pickle_load(storage["node"])
        # this would be correct, but since we pickle output and put it into a HDF node it doesn't appear here yet!
        # if "output" in storage.list_groups():
        if "output" in storage.list_nodes():
            self._output = pickle_load(storage["output"])

    @classmethod
    def _restore(cls, storage, version):
        job = cls(project=storage.project, job_name=storage.name)
        job.load(storage=storage)
        return job


# I'm not perfectly happy with this, but three thoughts led to this class:
# 1. I want to be able to set any node on a tiny job with subclassing, to make the prototyping new jobs in the notebook
#    easy
# 2. I do *not* want people to accidently change the node instance/class while the job is running
class GenericTinyJob(TinyJob):
    """
    A generic tiny job is a tiny job that allows to set any node class after instantiating it.

    Set a node class via :attr:`.node_class`, e.g.

    >>> from somewhere import MyNode
    >>> job = GenericTinyJob(Project(...), "myjob")
    >>> job.node_class = MyNode
    >>> isinstance(job.input, type(MyNode.input))
    True
    """
    def __init__(self, project, job_name):
        super().__init__(project=project, job_name=job_name)
        self._node_class = None

    @property
    def node_class(self):
        return self._node_class

    @node_class.setter
    def node_class(self, cls):
        self._node_class = cls

    def _get_node(self):
        return self.node_class()
