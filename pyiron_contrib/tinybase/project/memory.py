import tempfile

from pyiron_base import DataContainer
from pyiron_contrib.tinybase.project.interface import ProjectInterface
from pyiron_contrib.tinybase.storage import GenericStorage, DataContainerAdapter
from pyiron_contrib.tinybase.database import GenericDatabase, TinyDB


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

    def open_location(self, location) -> "InMemoryProject":
        return self.__class__(location, db=self.database, storage=self._storage)

    def create_storage(self, name) -> GenericStorage:
        return DataContainerAdapter(
            self, self._storage[self._location], "/"
        ).create_group(name)

    def exists_storage(self, name) -> bool:
        return name in self._storage[self._location].list_groups()

    def remove_storage(self, name):
        self._storage[self._location].pop(name)

    def request_directory(self, name):
        return tempfile.mkdtemp()

    def _get_database(self) -> GenericDatabase:
        return self._db

    @property
    def path(self) -> str:
        return self._location

    @property
    def name(self) -> str:
        return os.path.basename(self.path)
