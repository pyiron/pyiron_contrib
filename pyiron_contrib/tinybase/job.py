import abc
import logging
from typing import Optional

from pyiron_contrib.tinybase.task import AbstractTask
from pyiron_contrib.tinybase.storage import (
    Storable,
    GenericStorage,
    pickle_load,
    pickle_dump,
)
from pyiron_contrib.tinybase.executor import (
    Executor,
    BackgroundExecutor,
    ProcessExecutor,
)
from pyiron_contrib.tinybase.database import DatabaseEntry
from pyiron_contrib.tinybase.project import ProjectInterface, ProjectAdapter
from pyiron_base.state import state


class TinyJob(Storable, abc.ABC):
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

    _executors = {
        "foreground": Executor(),
        "background": BackgroundExecutor(max_threads=4),
        "process": ProcessExecutor(max_processes=4),
    }

    def __init__(self, project: ProjectInterface, job_name: str):
        """
        Create a new job.

        If the given `job_name` is already present in the `project` it is reloaded.  No checks are performed that the
        task type of the already present job and the current one match.  This is also not always necessary, e.g. when
        reloading a :class:`.GenericTinyJob` it will automatically read the
        correct task from storage.

        Args:
            project (:class:`.ProjectInterface`): the project the job should live in
            job_name (str): the name of the job.
        """
        if not isinstance(project, ProjectInterface):
            project = ProjectAdapter(project)
        self._project = project
        self._name = job_name
        self._task = None
        self._output = None
        self._storage = None
        self._executor = None
        self._id = None
        # FIXME: this should go into the job creation logic on the project
        if project.exists_storage(job_name):
            try:
                self.load()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to reload run job from storage: {e}"
                ) from None

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
    def _get_task(self) -> AbstractTask:
        """
        Return an instance of the :class:`.AbstractTask`.

        The value return from here is saved automatically in :prop:`.task`.
        """
        pass

    @property
    def task(self):
        if self._task is None:
            self._task = self._get_task()
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

    def run(self, how="foreground") -> Optional[Executor]:
        """
        Start execution of the job.

        If the job already has a database id and is not in "ready" state, do nothing.

        Args:
            how (string): specifies which executor to use

        Returns:
            :class:`.Executor`: the executor that is running the task or nothing.
        """
        if (
            self._id is None
            or self.project.database.get_item(self.id).status == "ready"
        ):
            exe = self._executor = self._executors[how].submit(tasks=[self.task])
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
        job = cls(project=storage.project, job_name=storage.name)
        job.load(storage=storage)
        return job


# I'm not perfectly happy with this, but three thoughts led to this class:
# 1. I want to be able to set any task on a tiny job with subclassing, to make the prototyping new jobs in the notebook
#    easy
# 2. I do *not* want people to accidently change the task instance/class while the job is running
class GenericTinyJob(TinyJob):
    """
    A generic tiny job is a tiny job that allows to set any task class after instantiating it.

    Set a task class via :attr:`.task_class`, e.g.

    >>> from somewhere import MyTask
    >>> job = GenericTinyJob(Project(...), "myjob")
    >>> job.task_class = MyTask
    >>> isinstance(job.input, type(MyTask.input))
    True
    """

    def __init__(self, project, job_name):
        super().__init__(project=project, job_name=job_name)
        self._task_class = None

    @property
    def task_class(self):
        return self._task_class

    @task_class.setter
    def task_class(self, cls):
        self._task_class = cls

    def _get_task(self):
        return self.task_class()
