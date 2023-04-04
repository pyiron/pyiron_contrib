import warnings

from pyiron_base import Toolkit

from pyiron_contrib.generic.coscineIo import CoscineProject, CoscineResource
from pyiron_contrib.generic.s3io import FileS3IO


class StorageInterfaceCreator:
    def __init__(self):
        pass

    @staticmethod
    def s3(*args, **kwargs):
        return FileS3IO(*args, **kwargs)

    @staticmethod
    def coscine(*args, **kwargs):
        return CoscineProject(*args, **kwargs)


class StorageInterfaceConnector:
    def __init__(self, project):
        self._store = {}
        if project is None:
            return

        if 'StorageIterface' in project.data:
            self._data = project.data.StorageInterface
        else:
            self._data = {}

        self._connect_storages()

    @classmethod
    def from_dict(cls, info_dict):
        self = cls(None)
        name = info_dict.get('name', 'new')
        self._data = {name: info_dict}
        self._connect_storages()

        return self

    def _connect_storages(self):
        for key, info_dict in self._data:
            if info_dict['type'] == str(CoscineResource):
                self[key] = CoscineResource(info_dict['info'])
            elif info_dict['type'] == str(FileS3IO):
                self[key] = FileS3IO.from_dict(info_dict['info'])

    def __setitem__(self, key, value):
        self._store[key] = value

    def __getitem__(self, item):
        return self._store[item]

    def __repr__(self):
        return f"Storage Access for {list(self._store.keys())}."


class StorageInterfaceFactory(Toolkit):

    def __init__(self, project):
        super().__init__(project)
        self._creator = StorageInterfaceCreator()
        self._storage_interface = None

    @property
    def create(self):
        return self._creator

    def attach(self, storage_name, storage_interface):
        info_dict = {'name': storage_name,
                     'type': str(type(storage_interface)),
                                            'info': storage_interface.connection_info}
        try:
            new = StorageInterfaceConnector.from_dict(info_dict)
        except ValueError:
            warnings.warn("Credential information insufficient to auto-connect - storage interface not saved!")
        else:
            if 'StorageInterface' not in self._project.data:
                self._project.data.create_group('StorageInterface')
            self._project.data.StorageInterface[storage_name] = info_dict
            if self._storage_interface is None:
                self._storage_interface = new
            else:
                self._storage_interface[storage_name] = new[storage_name]

    @property
    def storage(self):
        if self._storage_interface is None:
            self._storage_interface = StorageInterfaceConnector(self._project)
        return self._storage_interface
