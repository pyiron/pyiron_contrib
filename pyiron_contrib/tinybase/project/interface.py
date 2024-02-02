import abc

from typing import Union

from pyiron_contrib.tinybase.storage import GenericStorage
from pyiron_contrib.tinybase.database import GenericDatabase


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
    def request_directory(self, name):
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
