import abc
import logging
from typing import Optional, Union

from pyiron_contrib.tinybase.task import AbstractTask
from pyiron_contrib.tinybase.storage import (
    Storable,
    GenericStorage,
    pickle_load,
    pickle_dump,
)
from pyiron_contrib.tinybase.executor import (
    Executor,
    ExecutionContext,
    BackgroundExecutor,
    ProcessExecutor,
)
from pyiron_contrib.tinybase.database import DatabaseEntry
from pyiron_contrib.tinybase.project import ProjectInterface, ProjectAdapter
from pyiron_base.state import state


class TinyJob(Storable):
    """
    A tiny job unifies an executor, a task and its output.

    The job adds the task to the database and persists its input and output in a storage location.

    The input of the task is available from :attr:`~.input`. After the job has
    finished the output of the task can be
    accessed from :attr:`~.output` and the data written to storage from :attr:`.~storage`.

    This is an abstracat base class that works with any execution task without specifying it.  To create specialized
    jobs you can derive from it and overload :meth:`._get_task()` to return an
    instance of the task, e.g.

    >>> from somewhere import MyTask
    >>> class MyJob(TinyJob):
    ...     def _get_task(self):
    ...         return MyTask()

    The return value of :meth:`._get_task()` is persisted during the life time of the job.

    You can use :class:`.GenericTinyJob` to dynamically specify which task the job should execute.
    """

    def __init__(self, task: AbstractTask, project: ProjectInterface, job_name: str):
        """
        Create a new job.

        Args:
            task (:class:`.AbstractTask`): the underlying task to run
            project (:class:`.ProjectInterface`): the project the job should live in
            job_name (str): the name of the job.
        """
        if not isinstance(project, ProjectInterface):
            project = ProjectAdapter(project)
        self._project = project
        self._name = job_name
        self._task = task
        self._output = None
        self._storage = None
        self._executor = None
        self._id = None

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

    @property
    def status(self):
        try:
            return self._project.database.get_item(self.id).status
        except ValueError:
            return "initialized"

    @property
    def task(self):
        return self._task

    @property
    def jobtype(self):
        return self.task.__class__.__name__

    @property
    def input(self):
        return self.task.input

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
        self._executor._run_machine.observe(
            "finished", lambda _: self.store(self.storage)
        )

        self._executor._run_machine.observe("ready", self._add_to_database)
        self._executor._run_machine.observe("running", self._update_status("running"))
        self._executor._run_machine.observe("collect", self._update_status("collect"))
        self._executor._run_machine.observe("finished", self._update_status("finished"))

    def run(
        self, executor: Union[Executor, str, None] = None
    ) -> Optional[ExecutionContext]:
        """
        Start execution of the job.

        If the job already has a database id and is not in "ready" state, do nothing.

        Args:
            executor (:class:`~.Executor`, str): specifies which executor to
                use, if `str` must be a method name of :class:`.ExecutorCreator`;
                if not given use the last created executor

        Returns:
            :class:`.Executor`: the executor that is running the task or nothing.
        """
        if (
            self._id is None
            or self.project.database.get_item(self.id).status == "ready"
        ):
            if executor is None:
                executor = "most_recent"
            if isinstance(executor, str):
                executor = getattr(self.project.create.executor, executor)()
            exe = self._executor = executor.submit(tasks=[self.task])
            self._setup_executor_callbacks()
            exe.run()
            return exe
        else:
            logging.info("Job already finished!")

    def wait(self, timeout: Optional[float] = None):
        """
        Wait until job is finished.

        Args:
            timeout (float, optional): maximum time to wait in seconds; wait
                indefinitely by default

        Raises:
            ValueError: if job status is not `finished` or `running`
        """
        if self.status == "finished":
            return
        if self.status != "running":
            raise ValueError("Job not running!")
        self._executor.wait(timeout=timeout)

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
        storage["task"] = pickle_dump(self.task)
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
        self._task = pickle_load(storage["task"])
        # this would be correct, but since we pickle output and put it into a
        # HDF task it doesn't appear here yet!
        # if "output" in storage.list_groups():
        if "output" in storage.list_nodes():
            self._output = pickle_load(storage["output"])

    @classmethod
    def _restore(cls, storage, version):
        task = pickle_load(storage["task"])
        job = cls(task=task, project=storage.project, job_name=storage.name)
        job.load(storage=storage)
        return job
