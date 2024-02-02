import os
import os.path

from pyiron_contrib.tinybase.project.interface import ProjectInterface
from pyiron_contrib.tinybase.storage import (
    GenericStorage,
    DataContainerAdapter,
    H5ioStorage,
)
from pyiron_contrib.tinybase.database import GenericDatabase, TinyDB

from h5io_browser import Pointer as Hdf5Pointer


class FilesystemProject(ProjectInterface):
    """
    A plain project that stores data in HDF5 files on the filesystem and uses TinyDB.

    The database file will be created in the first location opened, but sub projects created from this object will share
    a database.  A global database is not yet supported.
    """

    def __init__(self, path):
        """
        Args:
            path (str): path to the project folder; will be created if non-existing.
        """
        self._path = path
        self._database = None

    @classmethod
    def open_location(cls, path):
        return cls(path)

    def _get_job_file(self, name):
        return os.path.join(self._path, name, "storage.h5")

    def create_storage(self, name):
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        return H5ioStorage.from_file(self, self._get_job_file(name), name)

    def exists_storage(self, name):
        return os.path.exists(self._get_job_file(name))

    def remove_storage(self, name):
        try:
            os.remove(self._get_job_file(name))
        except FileNotFoundError:
            pass

    def request_directory(self, name):
        path = os.path.join(self.path, name, "files")
        os.makedirs(path, exist_ok=True)
        return path

    def _get_database(self):
        if self._database is None:
            self._database = TinyDB(os.path.join(self.path, "pyiron.db"))
        return self._database

    @property
    def name(self):
        return os.path.basename(self._path)

    @property
    def path(self):
        return self._path


class SingleHdfProject(FilesystemProject):
    """
    Behaves likes a :class:`~.FilesystemProject` but stores all jobs in a single HDF5 file.
    """

    def _get_hdf(self):
        return H5ioStorage.from_file(self, os.path.join(self.path, "project.h5"))

    def create_storage(self, name):
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        return self._get_hdf()[name]

    def exists_storage(self, name):
        return name in self._get_hdf()[".."].list_groups()

    def remove_storage(self, name):
        del self._get_hdf()[name]
