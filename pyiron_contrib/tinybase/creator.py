"""
Creators of various things.

They serve to give easy access to various classes, so that users do not have to
manually import them.

Rough spec:

    1. RootCreator is merely the entry point and delegates to sub creators
    based on name
    2. Sub creators should be type homogenous or related by functionality, i.e.
    one of jobs, one for tasks, one for structures, etc.
    3. Each project gets its own RootCreator, but should keep that instance
    alive so that sub creators may cache certain things
"""

import abc
import importlib
from typing import Union
from functools import wraps
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

try:
    from os import sched_getaffinity

    cpu_count = lambda: len(sched_getaffinity(0))
except ImportError:
    from multiprocessing import cpu_count

import pyiron_contrib.tinybase.job
from pyiron_contrib.tinybase.project import JobNotFoundError
from pyiron_contrib.tinybase.executor import Submitter, FuturesSubmitter
from pyiron_atomistics import ase_to_pyiron, Atoms

import ase.build

from pympipool import Executor as PyMPIPoolExecutor
from dask.distributed import Client, LocalCluster


class Creator(abc.ABC):
    @abc.abstractmethod
    def __init__(self, project, config):
        pass

    # TODO add all methods


def load_class(class_path):
    module, klass = class_path.rsplit(".", maxsplit=1)
    try:
        return getattr(importlib.import_module(module), klass)
    except ImportError as e:
        raise ValueError(
            f"Importing task class '{class_path}' failed: {e.args}!"
        ) from None


class JobCreator(Creator):
    def __init__(self, project, job_dict):
        self._project = project
        self._jobs = job_dict.copy()

    def __dir__(self):
        return tuple(self._jobs.keys())

    def __getattr__(self, task_type):
        if task_type not in self._jobs:
            raise ValueError(f"Unknown task type {task_type}!")
        task = self._jobs[task_type]
        if isinstance(task, str):
            task = self._jobs[task_type] = load_class(task)

        def create(
            name: str,
            delete_existing_job: bool = False,
            delete_aborted_job: bool = False,
        ):
            """
            Create or load a new job.

            Args:
                name (str): name of the job
                delete_existing_job (bool): if a job of this name and type
                    exists, delete it first
                delete_existing_job (bool): if a job if this name and type
                    exists and its status is aborted, delete it first
            """
            try:
                job = self._project.load(name)
                if delete_existing_job or (
                    delete_aborted_job and job.status == "aborted"
                ):
                    job.remove()
                else:
                    if not isinstance(job.task, task):
                        raise ValueError(
                            "Job with given name already exists, but is of different type!"
                        )
                    return job
            except JobNotFoundError:
                pass
            return pyiron_contrib.tinybase.job.TinyJob(task(), self._project, name)

        return create

    def register(self, task: Union["AbstractTask", str], name: str):
        """
        Register a new task.

        Args:
            task (AbstractTask): new task to register

        Returns:
            AbstractTask: the newly registered task
        """
        if name is None:
            name = type(task).__name__
        if name in self._jobs:
            if task != self._jobs[name]:
                raise ValueError(
                    "Refusing to register task: different task of the same name already exists!"
                )
            else:
                return self._jobs[name]
        self._jobs[name] = task
        return task


class TaskCreator(Creator):
    def __init__(self, project, config):
        self._tasks = config.copy()

    def __dir__(self):
        return tuple(self._tasks.keys())

    def __getattr__(self, name):
        task = self._tasks[name]
        if isinstance(task, str):
            task = self._tasks[name] = load_class(task)
        return task

    def register(self, task: Union["AbstractTask", str], name: str):
        """
        Register a new task.

        Args:
            task (AbstractTask): new task to register

        Returns:
            AbstractTask: the newly registered task
        """
        if name is None:
            name = type(task).__name__
        if name in self._jobs:
            if task != self._jobs[name]:
                raise ValueError(
                    "Refusing to register task: different task of the same name already exists!"
                )
            else:
                return self._jobs[name]
        self._jobs[name] = task
        return task


class StructureCreator(Creator):
    def __init__(self, project, config):
        pass

    @wraps(ase.build.bulk)
    def bulk(self, *args, **kwargs):
        return ase_to_pyiron(ase.build.bulk(*args, **kwargs))

    atoms = Atoms


class ExecutorCreator(Creator):
    _DEFAULT_CPUS = min(int(0.5 * cpu_count()), 8)

    def __init__(self, project, config):
        self._most_recent = None

    def most_recent(self):
        if self._most_recent is None:
            self._most_recent = Submitter()
        return self._most_recent

    def _save(func):
        @wraps(func)
        def f(self, *args, **kwargs):
            self._most_recent = func(self, *args, **kwargs)
            return self._most_recent

        return f

    @wraps(ProcessPoolExecutor)
    @_save
    def process(self, max_processes=_DEFAULT_CPUS, **kwargs):
        return FuturesSubmitter(
            ProcessPoolExecutor(max_workers=max_processes, **kwargs)
        )

    @_save
    def foreground(self):
        return Submitter()

    @wraps(ThreadPoolExecutor)
    @_save
    def background(self, max_workers=4, **kwargs):
        return FuturesSubmitter(ThreadPoolExecutor(max_workers=max_workers, **kwargs))

    @wraps(LocalCluster)
    @_save
    def dask_local(self, n_workers=_DEFAULT_CPUS, **kwargs):
        return FuturesSubmitter(LocalCluster(n_workers=n_workers, **kwargs))

    @wraps(Client)
    @_save
    def dask_cluster(self, cluster, **kwargs):
        return FuturesSubmitter(Client(cluster, **kwargs))

    @wraps(PyMPIPoolExecutor)
    @_save
    def pympipool(self, max_workers=_DEFAULT_CPUS, **kwargs):
        return FuturesSubmitter(PyMPIPoolExecutor(max_workers=max_workers, **kwargs))

    del _save


class RootCreator:
    def __init__(self, project, config):
        self._project = project
        self._subcreators = {}
        for name, (subcreator, subconfig) in config.items():
            self._subcreators[name] = subcreator(project, subconfig)

    def __dir__(self):
        return tuple(self._subcreators.keys())

    def register(self, name, subcreator, force=False):
        """
        Add a new creator.
        """
        if name not in self._subcreators or force:
            self._subcreators[name] = subcreator
        else:
            raise ValueError(f"Already registered a creator under name {name}!")

    def __getattr__(self, name):
        try:
            return self._subcreators[name]
        except KeyError:
            raise AttributeError(f"No creator of name {name} registered!")


TASK_CONFIG = {
    "AseStatic": "pyiron_contrib.tinybase.ase.AseStaticTask",
    "AseMinimize": "pyiron_contrib.tinybase.ase.AseMinimizeTask",
    "AseMD": "pyiron_contrib.tinybase.ase.AseMDTask",
    "Murnaghan": "pyiron_contrib.tinybase.murn.MurnaghanTask",
}


CREATOR_CONFIG = {
    "job": (JobCreator, TASK_CONFIG),
    "task": (TaskCreator, TASK_CONFIG),
    "structure": (StructureCreator, {}),
    "executor": (ExecutorCreator, {}),
}
