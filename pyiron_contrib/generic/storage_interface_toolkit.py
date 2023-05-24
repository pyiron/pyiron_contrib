import warnings

from pyiron_base import Toolkit
from pyiron_base import ImportAlarm

try:
    from pyiron_contrib.generic.coscineIo import CoscineProject, CoscineResource

    import_alarm = ImportAlarm()
except ImportError:
    import_alarm = ImportAlarm("Connecting to CoScInE requires the coscine package.")

    class CoscineProject:
        def __init__(self, *args, **kwargs):
            raise ImportError("Interacting with coscine requires the coscine package")

    class CoscineResource:
        def __init__(self, *args, **kwargs):
            raise ImportError("Interacting with coscine requires the coscine package")


from pyiron_contrib.generic.s3io import FileS3IO


class StorageInterfaceCreator:
    @import_alarm
    def __init__(self):
        pass

    @staticmethod
    def s3(*args, **kwargs):
        return FileS3IO(*args, **kwargs)

    @staticmethod
    def coscine(*args, **kwargs):
        return CoscineProject(*args, **kwargs)


class StorageInterfaceConnector:
    _known_storage_classes = {str(CoscineResource): "coscine", str(FileS3IO): "s3"}

    @import_alarm
    def __init__(self, project):
        self._store = {}
        if project is None:
            return

        if "StorageInterface" in project.data:
            self._data = project.data.StorageInterface.copy()
        else:
            self._data = {}

    @classmethod
    def from_dict(cls, info_dict):
        self = cls(None)
        name = info_dict.get("name", "new")
        self._data = {name: info_dict}
        self._connect_storage(name)

        return self

    def _connect_storage(self, name):
        info_dict = self._data[name]
        if info_dict["type"] == str(CoscineResource):
            self._store[name] = CoscineResource(info_dict["info"])
        elif info_dict["type"] == str(FileS3IO):
            self._store[name] = FileS3IO.from_dict(info_dict["info"])

    def connect_all_storages(self):
        for key in self._data:
            self._connect_storage(key)

    def attach_temporary(self, key, value, data_dict):
        self._store[key] = value
        self._data[key] = data_dict

    def __getitem__(self, item):
        if item not in self._store and item in self._data:
            self._connect_storage(item)
        elif item not in self._store:
            raise KeyError(item)
        return self._store[item]

    @property
    def info(self):
        result = {}
        for key in self._data:
            if key in self._store:
                result[key] = {"type": self._data[key]["type"], "connected": True}
            else:
                result[key] = {"type": self._data[key]["type"], "connected": False}
        return result

    def __repr__(self):
        result = []
        for key, value in self.info.items():
            conn = "connected" if value["connected"] else "inactive"
            storage_class = value["type"]
            storage_type = (
                self._known_storage_classes[storage_class]
                if storage_class in self._known_storage_classes
                else storage_class
            )
            result.append(f"{key}({storage_type}, {conn})")
        return f"Storage Access for {result}."


class StorageInterfaceFactory(Toolkit):
    def __init__(self, project):
        super().__init__(project)
        self._creator = None
        self._storage_interface = None

    @property
    def create(self):
        if self._creator is None:
            self._creator = StorageInterfaceCreator()
        return self._creator

    def attach(self, storage_name, storage_interface):
        info_dict = {
            "name": storage_name,
            "type": str(type(storage_interface)),
            "info": storage_interface.connection_info,
        }
        try:
            new = StorageInterfaceConnector.from_dict(info_dict)
        except ValueError as e:
            raise ValueError(
                "Credential information insufficient to auto-connect - storage interface not saved!"
            ) from e
        else:
            if "StorageInterface" not in self._project.data:
                self._project.data.create_group("StorageInterface")
            self._project.data.StorageInterface[storage_name] = info_dict
            self._project.data.write()
            if self._storage_interface is None:
                self._storage_interface = new
            else:
                self._storage_interface.attach_temporary(
                    storage_name, new[storage_name], info_dict
                )

    @property
    def storage(self):
        if self._storage_interface is None:
            self._storage_interface = StorageInterfaceConnector(self._project)
        return self._storage_interface
