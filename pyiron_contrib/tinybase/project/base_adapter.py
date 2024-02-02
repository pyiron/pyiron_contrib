import os.path

from pyiron_base import Project
from pyiron_contrib.tinybase.storage import ProjectHDFioStorageAdapter
from pyiron_contrib.tinybase.storage import GenericStorage
from pyiron_contrib.tinybase.project.interface import ProjectInterface
from pyiron_contrib.tinybase.database import GenericDatabase, TinyDB


class ProjectAdapter(ProjectInterface):
    def __init__(self, project):
        self._project = project
        self._database = None

    @classmethod
    def open_location(cls, location) -> "ProjectAdapter":
        return cls(Project(location))

    def create_storage(self, name) -> ProjectHDFioStorageAdapter:
        return ProjectHDFioStorageAdapter(
            self, self._project.create_hdf(self._project.path, name)
        )

    def exists_storage(self, name) -> bool:
        return self._project.create_hdf(self._project.path, name).file_exists

    def remove_storage(self, name):
        self._project.create_hdf(self._project.path, name).remove_file()

    def request_directory(self, name):
        path = os.path.join(self._project.path, name + "_hdf5", name)
        os.makedirs(path, exist_ok=True)
        return path

    def _get_database(self) -> GenericDatabase:
        if self._database is None:
            self._database = TinyDB(os.path.join(self._project.path, "pyiron.db"))
        return self._database

    @property
    def name(self) -> str:
        return self._project.name

    @property
    def path(self) -> str:
        return self._project.path
