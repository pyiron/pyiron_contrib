import abc
import importlib
from typing import Any, Union, Optional

from pyiron_base import DataContainer
from pyiron_contrib.tinybase import __version__ as base__version__

import pickle
import codecs


# utility functions until ASE can be HDF'd
def pickle_dump(obj):
    return codecs.encode(pickle.dumps(obj), "base64").decode()


def pickle_load(buf):
    return pickle.loads(codecs.decode(buf.encode(), "base64"))


class GenericStorage(abc.ABC):
    """
    Generic interface to store things.

    The canonical implementation and model is ProjectHDFio from base, but other implementations are thinkable (S3, in
    memory, etc.)

    The concepts are borrowed mostly from HDF5 files.  There are groups, :meth:`.list_groups()`, and nodes,
    :meth:`.list_nodes()`, inside any storage.  Every group has a name, :attr:`.name`.

    Implementations must allow multiple objects of this class to refer to the same underlying storage group at the same
    time and access via the methods here must be atomic.
    """

    @abc.abstractmethod
    def __getitem__(self, item: str) -> Union["GenericStorage", Any]:
        """
        Return a value from storage.

        If `item` is in :meth:`.list_groups()` this must return another :class:`.GenericStorage`.

        Args:
            item (str): name of value

        Returns:
            :class:`.GenericStorage`: if `item` refers to a sub group
            object: value that is stored under `item`

        Raises:
            KeyError: `item` is neither a node or a sub group of this group
        """
        pass

    @abc.abstractmethod
    def __setitem__(self, item: str, value: Any):
        """
        Set a value to storage.

        Args:
            item (str): name of the value
            value (object): value to store
        """
        pass

    @abc.abstractmethod
    def create_group(self, name):
        """
        Create a new sub group.

        Args:
            name (str): name of the new group
        """
        pass

    def open(self, name: str) -> "GenericStorage":
        """
        Descend into a sub group.

        If `name` does not exist yet, create a new group.  Calling :meth:`~.close` on the returned object returns this
        object.

        Args:
            name (str): name of sub group

        Returns:
            :class:`.GenericStorage`: sub group
        """
        # FIXME: what if name in self.list_nodes()
        new = self.create_group(name)
        new._prev = self
        return new

    def close(self) -> "GenericStorage":
        """
        Surface from a sub group.

        If this object was not returned from a previous call to :meth:`.open` it returns itself silently.
        """
        try:
            return self._prev
        except AttributeError:
            return self

    @abc.abstractmethod
    def list_nodes(self) -> list[str]:
        """
        List names of values inside group.
        """
        pass

    @abc.abstractmethod
    def list_groups(self) -> list[str]:
        """
        List name of sub groups.
        """
        pass

    # DESIGN: this mostly exists to help to_object()ing GenericTinyJob, but it introduces a circular-ish connection.
    # Maybe there's another way to do it?
    @property
    @abc.abstractmethod
    def project(self):
        """
        The project that this storage belongs to.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self):
        """
        The name of the group that this object is currently pointing to.

        (In ProjectHDFio speak, this is the os.path.basename of h5_path)
        """
        pass

    def to_object(self) -> "Storable":
        """
        Instantiate an object serialized to this group.

        Returns:
            :class:`.Storable`: deserialized object

        Raises:
            ValueError: no was object serialized in this group (either NAME, MODULE or VERSION values missing)
            RuntimeError: failed to import serialized object (probably some libraries not installed)
        """
        try:
            name = self["NAME"]
            module = self["MODULE"]
            version = self["VERSION"]
        except KeyError:
            raise ValueError("No object was serialized in this group!")
        try:
            cls = getattr(importlib.import_module(module), name)
        except ImportError:
            raise RuntimeError("Failed to import serialized object!")
        return cls.restore(self, version=version)


class ProjectHDFioStorageAdapter(GenericStorage):
    """
    Adapter class around ProjectHDFio to let it be used as a GenericStorage.
    """

    def __init__(self, project, hdf):
        self._project = project
        self._hdf = hdf

    def __getitem__(self, item):
        return self._hdf[item]

    def __setitem__(self, item, value):
        self._hdf[item] = value

    def create_group(self, name):
        return ProjectHDFioStorageAdapter(self._project, self._hdf.create_group(name))

    def list_nodes(self):
        return self._hdf.list_nodes()

    def list_groups(self):
        return self._hdf.list_groups()

    # compat with small bug in base ProjectHDFio
    list_dirs = list_groups

    @property
    def project(self):
        return self._project

    @property
    def name(self):
        return self._hdf.name


class DataContainerAdapter(GenericStorage):
    """
    Provides in memory location to store objects.
    """

    def __init__(self, project, cont: DataContainer, name):
        self._project = project
        self._cont = cont
        self._name = name

    def __getitem__(self, item: str) -> Union["GenericStorage", Any]:
        v = self._cont[item]
        if isinstance(v, DataContainer):
            return self.__class__(self._project, v, item)
        else:
            return v

    def __setitem__(self, item: str, value: Any):
        self._cont[item] = value

    def create_group(self, name):
        if name not in self._cont:
            d = self._cont.create_group(name)
        else:
            d = self._cont[name]
        return self.__class__(self._project, d, name)

    def list_nodes(self):
        return self._cont.list_nodes()

    def list_groups(self):
        return self._cont.list_groups()

    @property
    def project(self):
        return self._project

    @property
    def name(self):
        return self._name


# DESIGN: equivalent of HasHDF but with generalized language
class Storable(abc.ABC):
    """
    Interface for classes that can store themselves to a :class:`~.GenericStorage`

    Necessary overrides are :meth:`._store` and :meth:`._restore`.
    """

    @abc.abstractmethod
    def _store(self, storage):
        pass

    def _store_type(self, storage):
        storage["NAME"] = self.__class__.__name__
        storage["MODULE"] = self.__class__.__module__
        # DESIGN: what happens with objects defined in different parts of pyiron{_contrib,_atomistics,_base}?  Which
        # version do we save?
        storage["VERSION"] = base__version__

    def store(self, storage: GenericStorage, group_name: Optional[str] = None):
        """
        Store object into storage.

        Args:
            storage (:class:`.GenericStorage`): storage to write to
            group_name (str): if given descend into this subgroup first
        """
        if group_name is not None:
            storage = storage.create_group(group_name)
        self._store_type(storage)
        self._store(storage)

    @classmethod
    @abc.abstractmethod
    def _restore(cls, storage: GenericStorage, version: str) -> "Storable":
        pass

    @classmethod
    def restore(cls, storage: GenericStorage, version: str) -> "Storable":
        """
        Restore an object of type `cls` from storage.

        Args:
            storage (:class:`.GenericStorage`): storage to read from
            version (str): version string of pyiron that wrote the object

        Return:
            :class:`.Storable`: deserialized object

        Raises:
            ValueError: failed to restore object
        """
        try:
            return cls._restore(storage, version)
        except Exception as e:
            raise ValueError(f"Failed to restore object with {e}")


class HasHDFAdapaterMixin(Storable):
    """
    Implements :class:`.Storable` in terms of HasHDF.  Make any sub class of it a subclass :class:`.Storable` as well by
    mixing this class in.
    """

    def _store(self, storage):
        self._to_hdf(storage)

    @classmethod
    def _restore(cls, storage, version):
        kw = cls.from_hdf_args(storage)
        obj = cls(**kw)
        obj._from_hdf(storage, version)
        return obj
