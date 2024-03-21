import abc
import codecs
import importlib
import pickle
from typing import Any, Union, Optional

from pyiron_base import DataContainer
from pyiron_base.interfaces.has_groups import HasGroups
from pyiron_base.storage.hdfio import ProjectHDFio
from pyiron_contrib.tinybase import __version__ as base__version__

from h5io_browser import Pointer as Hdf5Pointer


class GenericStorage(HasGroups, abc.ABC):
    """
    Generic interface to store things.

    The canonical implementation and model is ProjectHDFio from base, but other implementations are thinkable (S3, in
    memory, etc.)

    The concepts are borrowed mostly from HDF5 files.  There are groups, :meth:`.list_groups()`, and nodes,
    :meth:`.list_nodes()`, inside any storage.  Every group has a name, :attr:`.name`.

    Implementations must allow multiple objects of this class to refer to the same underlying storage group at the same
    time and access via the methods here must be atomic.

    Mandatory overrides for all implementations are

    1. :meth:`.__getitem__` to read values,
    2. :meth:`._set` to write values,
    3. :meth:`.create_group` to create sub groups,
    4. :meth:`.list_nodes` and :meth:`.list_groups` to see the contained groups and nodes,
    5. :attr:`.project` which is a back reference to the project that originally created this storage,
    6. :attr:`.name` which is the name of the group that this object points to, e.g. `storage.create_group(name).name == name`.

    For values that implement :class:`.Storable` there is an intentional asymmetry in item writing and reading.  Writing
    it calls :meth:`.Storable.store` automatically, but reading will return the :class:`.GenericStorage` group that was
    created during writing *without* calling :meth:`.Storable.restore` automatically.  The original value can be
    obtained by calling :meth:`.GenericStorage.to_object` on the returned group.  This is so that power users and
    developers can access sub objects efficiently without having to load all the containing objects first.

    If :meth:`_set` raises a `TypeError` indicating that it does not know how to store an object of a certain type,
    :meth:`.__setitem__` will pickle it automatically and try to write it again.  Such an object can be retrieved with
    :meth:`.to_object`.  The same is done for lists that contain elements which are not trivially storable in the
    storage implementation, e.g. lists of :class:`.Atoms` or other complex objects.
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

    def get(self, item, default=None):
        """
        Same as item access, but return default if given.

        Args:
            item (str): name of value

        Returns:
            :class:`.GenericStorage`: if `item` refers to a sub group
            object: value that is stored under `item`

        Raises:
            KeyError: `item` is neither a node or a sub group of this group
        """
        try:
            value = self[item]
        except (KeyError, ValueError):
            if default is not None:
                return default
            else:
                raise

    @abc.abstractmethod
    def _set(self, item: str, value: Any):
        """
        Set a value to storage.

        If this method raises a `TypeError` when called by
        :meth:`~.__setitem__`, that method will pickle the value and try again.

        Args:
            item (str): name of the value
            value (object): value to store

        Raises:
            TypeError: if the underlying storage cannot store values of the given type naively
        """
        pass

    def __setitem__(self, item: str, value: Any):
        """
        Set a value to storage.

        Args:
            item (str): name of the value
            value (object): value to store
        """
        if item in self.list_groups():
            del self[item]
        if isinstance(value, Storable):
            value.store(self, group_name=item)
        else:
            try:
                self._set(item, value)
            except TypeError:
                if isinstance(value, list):
                    self[item] = ListStorable(value)
                else:
                    self[item] = PickleStorable(value)

    @abc.abstractmethod
    def __delitem__(self, item: str):
        """
        Remove a group or node.

        Args:
            item (str): name of the item to delete
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

    # compatibility with ProjectHDFio, so that implementations of GenericStorage can be used as a drop-in replacement
    # for it
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
        value = self._hdf[item]
        if isinstance(value, ProjectHDFio):
            return type(self)(self._project, value)
        else:
            return value

    def _set(self, item, value):
        try:
            self._hdf[item] = value
        except TypeError:  # HDF layer doesn't know how to write value
            # h5io bug, when triggering an error in the middle of a write
            # some residual data maybe left in the file
            del self._hdf[item]
            raise

    def __delitem__(self, item):
        del self._hdf[item]

    def create_group(self, name):
        return ProjectHDFioStorageAdapter(self._project, self._hdf.create_group(name))

    def _list_nodes(self):
        return self._hdf.list_nodes()

    def _list_groups(self):
        return self._hdf.list_groups()

    # compat with small bug in base ProjectHDFio
    list_dirs = _list_groups

    @property
    def project(self):
        return self._project

    @property
    def name(self):
        return self._hdf.name

    def to_object(self):
        try:
            # Since we try to store object with _hdf[item] = value, which might trigger HasHDF functionality, we have to
            # try here to restore the object via that functionality as well
            return self._hdf.to_object()
        except:
            return super().to_object()


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

    def _set(self, item: str, value: Any):
        self._cont[item] = value

    def __delitem__(self, item):
        del self._cont[item]

    def create_group(self, name):
        if name not in self._cont:
            d = self._cont.create_group(name)
        else:
            d = self._cont[name]
        return self.__class__(self._project, d, name)

    def _list_nodes(self):
        return self._cont.list_nodes()

    def _list_groups(self):
        return self._cont.list_groups()

    @property
    def project(self):
        return self._project

    @property
    def name(self):
        return self._name


class H5ioStorage(GenericStorage):
    """
    Store objects in HDF5 files.

    Maybe created with a non existing file path or HDF5 group.  Those will be created on first write access.
    """

    def __init__(self, pointer: Hdf5Pointer, project):
        """
        Args:
            pointer (:class:`h5io_browser.Pointer`): open pointer object to HDF5 storage
            project (:class:`.tinybase.ProjectInterface`): project this storage belongs to
        """
        if not isinstance(pointer, Hdf5Pointer):
            raise TypeError("pointer must be a h5io_browser.Pointer!")
        self._project = project
        self._pointer = pointer

    @classmethod
    def from_file(cls, project, file: str, path: str = None):
        """
        Open a storage from the given file and HDF group within.

        Args:
            project (:class:`.tinybase.ProjectInterface`): project this storage belongs to
            file (str): file path to the HDF5 file
            path (str): group path within the HDF5 file
        """
        pointer = Hdf5Pointer(file)
        if path is not None:
            pointer = pointer[path]
        return cls(pointer, project=project)

    def __getitem__(self, item):
        value = self._pointer[item]
        if isinstance(value, Hdf5Pointer):
            return type(self)(value, project=self._project)
        else:
            return value

    def _set(self, item, value):
        self._pointer[item] = value

    def __delitem__(self, item):
        del self._pointer[item]

    def create_group(self, name):
        return type(self)(self._pointer[name], project=self._project)

    def _list_nodes(self):
        return self._pointer.list_h5_path()["nodes"]

    def _list_groups(self):
        return self._pointer.list_h5_path()["groups"]

    @property
    def project(self):
        return self._project

    @property
    def name(self):
        return self._pointer.h5_path.rsplit("/", maxsplit=1)[1]


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

        The object returned may not be of type `cls` in special circumstances,
        such as :class:`.PickleStorable`, which returns its underlying value
        directly.

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
            raise ValueError(f"Failed to restore object with: {e}")


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


def pickle_dump(obj):
    return codecs.encode(pickle.dumps(obj), "base64").decode()


def pickle_load(buf):
    return pickle.loads(codecs.decode(buf.encode(), "base64"))


class PickleStorable(Storable):
    """
    Trivial implementation of :class:`.Storable` that pickles values.

    Used as a fallback by :class:`.GenericStorage` if value cannot
    be stored in HDF natively.
    """

    def __init__(self, value):
        self._value = value

    def _store(self, storage):
        storage["pickle"] = pickle_dump(self._value)

    @classmethod
    def _restore(cls, storage, version):
        return pickle_load(storage["pickle"])


class ListStorable(Storable):
    """
    Trivial implementation of :class:`.Storable` for lists with potentially complex objects inside.

    Used by :class:`.GenericStorage` as a fallback if storing the list with h5py/h5io as it is fails.
    """

    def __init__(self, value):
        self._value = value

    def _store(self, storage):
        for i, v in enumerate(self._value):
            storage[f"index_{i}"] = v

    @classmethod
    def _restore(cls, storage, version):
        keys = sorted(
            [v for v in storage.list_nodes() if v.startswith("index_")]
            + [v for v in storage.list_groups() if v.startswith("index_")],
            key=lambda k: int(k.split("_")[1]),
        )
        value = []
        for k in keys:
            v = storage[k]
            if isinstance(v, GenericStorage):
                v = v.to_object()
            value.append(v)
        return value
