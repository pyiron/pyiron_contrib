import abc
import os.path

from pyiron_base import Project
from pyiron_contrib.tinybase.storage import GenericStorage, ProjectHDFioStorageAdapter
from pyiron_contrib.tinybase.database import TinyDB, GenericDatabase

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

    def load(self, name_or_id: int | str) -> "TinyJob":
        # if a name is given, job must be in the current project
        if isinstance(name_or_id, str):
            pr = self
            name = name_or_id
        else:
            entry = self.database.get_item(name_or_id)
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

class ProjectAdapter(ProjectInterface):

    def __init__(self, project):
        self._project = project
        self._database = None

    @classmethod
    def open_location(cls, location):
        return cls(Project(location))

    def create_storage(self, name):
        return ProjectHDFioStorageAdapter(
                self,
                self._project.create_hdf(self._project.path, name)
        )

    def exists_storage(self, name) -> bool:
        return self._project.create_hdf(self._project.path, name).file_exists

    def remove_storage(self, name):
        self._project.create_hdf(self._project.path, name).remove_file()

    def remove(self, job_id):
        entry = self.database.remove_item(job_id)
        self.remove_storage(entry.name)

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

    def job_table(self):
        return self.database.job_table()

    def get_job_id(self, name):
        project_id = self.database.get_project_id(self.path)
        return self.database.get_item_id(name, project_id)
