import abc
import os.path
from typing import Union

from pyiron_base import Project, DataContainer
from pyiron_contrib.tinybase.storage import (
    GenericStorage,
    ProjectHDFioStorageAdapter,
    DataContainerAdapter,
)
from pyiron_contrib.tinybase.database import TinyDB, GenericDatabase


class JobNotFoundError(Exception):
    pass


class ProjectInterface(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def open_location(cls, location) -> "ProjectInterface":
        pass

    @abc.abstractmethod
    def create_storage(self, name) -> GenericStorage:
        pass

    @abc.abstractmethod
    def exists_storage(self, name) -> bool:
        pass

    @abc.abstractmethod
    def remove_storage(self, name):
        pass

    @abc.abstractmethod
    def _get_database(self) -> GenericDatabase:
        pass

    @property
    def database(self):
        return self._get_database()

    @property
    def create(self):
        if not hasattr(self, "_creator"):
            from pyiron_contrib.tinybase.creator import RootCreator, CREATOR_CONFIG

            self._creator = RootCreator(self, CREATOR_CONFIG)
        return self._creator

    def load(self, name_or_id: Union[int, str]) -> "TinyJob":
        """
        Load a job from storage.

        If the job name is given, it must be a child of this project and not
        any of its sub projects.

        Args:
            name_or_id (int, str): either the job name or its id

        Returns:
            :class:`.TinyJob`: the loaded job

        Raises:
            :class:`.JobNotFoundError`: if no job of the given name or id exists
        """
        if isinstance(name_or_id, str):
            pr = self
            name = name_or_id
            if not pr.exists_storage(name):
                raise JobNotFoundError(f"No job with name {name} found!")
        else:
            try:
                entry = self.database.get_item(name_or_id)
            except ValueError as e:
                raise JobNotFoundError(*e.args)
            pr = self.open_location(entry.project)
            name = entry.name
        return pr.create_storage(name).to_object()

    @property
    @abc.abstractmethod
    def path(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    def job_table(self):
        return self.database.job_table()

    def get_job_id(self, name):
        project_id = self.database.get_project_id(self.path)
        return self.database.get_item_id(name, project_id)

    def remove(self, job_id):
        entry = self.database.remove_item(job_id)
        if entry.project == self.path:
            pr = self
        else:
            pr = self.open_location(entry.project)
        pr.remove_storage(entry.name)

    # TODO:
    # def copy_to/move_to across types of ProjectInterface


class ProjectAdapter(ProjectInterface):
    def __init__(self, project):
        self._project = project
        self._database = None

    @classmethod
    def open_location(cls, location):
        return cls(Project(location))

    def create_storage(self, name):
        return ProjectHDFioStorageAdapter(
            self, self._project.create_hdf(self._project.path, name)
        )

    def exists_storage(self, name) -> bool:
        return self._project.create_hdf(self._project.path, name).file_exists

    def remove_storage(self, name):
        self._project.create_hdf(self._project.path, name).remove_file()

    def _get_database(self):
        if self._database is None:
            self._database = TinyDB(os.path.join(self._project.path, "pyiron.db"))
        return self._database

    @property
    def name(self):
        return self._project.name

    @property
    def path(self):
        return self._project.path


class InMemoryProject(ProjectInterface):
    def __init__(self, location, db=None, storage=None):
        if db is None:
            db = TinyDB(":memory:")
        self._db = db
        self._location = location
        if storage is None:
            storage = {}
        self._storage = storage
        if location not in storage:
            self._storage[location] = DataContainer()

    def open_location(self, location):
        return self.__class__(location, db=self.database, storage=self._storage)

    def create_storage(self, name) -> GenericStorage:
        return DataContainerAdapter(
            self, self._storage[self._location], "/"
        ).create_group(name)

    def exists_storage(self, name) -> bool:
        return name in self._storage[self._location].list_groups()

    def remove_storage(self, name):
        self._storage[self._location].pop(name)

    def _get_database(self) -> GenericDatabase:
        return self._db

    @property
    def path(self):
        return self._location

    @property
    def name(self):
        return os.path.basename(self.path)
