"""Generic File Object."""

# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
import posixpath
import shutil

from abc import ABC, abstractmethod
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from pyiron_base import GenericJob, state
from pyiron_base.storage.filedata import (
    FileDataTemplate as BaseFileDataTemplate,
    load_file,
    FileData as FileDataBase,
)
from pyiron_base import HasGroups

try:
    import ipywidgets as widgets
    from IPython.display import display

    gui_elements_imported = True
except ImportError:
    gui_elements_imported = False

__author__ = "Niklas Siemer"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Niklas Siemer"
__email__ = "siemer@mpie.de"
__status__ = "development"
__date__ = "Feb 02, 2021"


class MetaDataTemplate(ABC):
    @abstractmethod
    def to_dict(self):
        """Provide a dict representation of the meta data."""

    def _repr_html_(self):
        df = pd.DataFrame(
            {
                "Metadata": list(self.to_dict().keys()),
                "  ": list(self.to_dict().values()),
            }
        )
        df.set_index("Metadata", inplace=True)
        df[df.isnull()] = ""
        return df._repr_html_()


class MetaData(MetaDataTemplate):
    def __init__(self, metadata_dict=None):
        if metadata_dict is None:
            metadata_dict = {}
        self._metadata = {key: val for key, val in metadata_dict.items()}

    def to_dict(self):
        return self._metadata.copy()

    def __repr__(self):
        return f"{self.__class__.__name__}({self._metadata})"


class FileDataTemplate(BaseFileDataTemplate, ABC):
    def __init__(self):
        self._metadata = None
        if gui_elements_imported:
            self._output = widgets.Output()
            self._box = widgets.VBox([self._output])

    @staticmethod
    def _get_filetype_from_filename(filename):
        filetype = Path(filename).suffix
        return None if filetype == "" else filetype[1:]

    def _get_metadata(self):
        return self._metadata

    def _set_metadata(self, metadata):
        if not isinstance(metadata, MetaDataTemplate):
            self._metadata = MetaData(metadata)
        else:
            self._metadata = metadata

    @property
    def metadata(self):
        return self._get_metadata()

    @metadata.setter
    def metadata(self, metadata):
        self._set_metadata(metadata)

    if gui_elements_imported:

        @property
        def gui(self):
            self._output.clear_output()
            with self._output:
                display(self.data, self.metadata)
            return self._box

        def _ipython_display_(self):
            display(self.gui)


class FileData(FileDataTemplate):
    """FileData stores an instance of a data file, e.g. a single Image from a measurement."""

    def __init__(self, file, data=None, metadata=None, filetype=None):
        """FileData class to store data and associated metadata.

        Args:
            file (str): path to the data file (if data is None) or filename associated with the data.
            data (object/None): object containing data
            metadata (dict/DataContainer): Dictionary of metadata associated with the data
            filetype (str): File extension associated with the type data,
                            If provided this overwrites the assumption based on the extension of the filename.
        """
        super().__init__()
        if data is None:
            self.filename = os.path.split(file)[1]
            self.source = file
            self._data = None
        else:
            self.filename = file
            self.source = None
            self._data = data
        if filetype is None:
            self.filetype = self._get_filetype_from_filename(self.filename)
        else:
            self.filetype = filetype
        if metadata is None:
            self.metadata = MetaData()
        elif isinstance(metadata, MetaDataTemplate):
            self.metadata = metadata
        else:
            self.metadata = MetaData(metadata)
        self._hasdata = self._data is not None

    @property
    @lru_cache()
    def data(self):
        """Return the associated data."""
        if self._hasdata:
            return self._data
        else:
            return load_file(self.source, filetype=self.filetype)


class StorageInterface(HasGroups, ABC):
    """File handling in different storage interfaces"""

    def __init__(self):
        self._path = None

    @property
    def path(self):
        """Represents the path of the storage interface (always posix-path)"""
        return self._path

    @path.setter
    def path(self, new_path):
        self._path = self._path_setter(new_path)

    def _path_setter(self, new_path):
        return new_path

    @abstractmethod
    def is_file(self, item):
        """Check if the given item is a node/file"""

    @abstractmethod
    def is_dir(self, item):
        """Check if the given item is a group/directory"""

    @abstractmethod
    def upload_file(self, file, metadata=None, filename=None):
        """Upload the provided files to the storage"""

    @abstractmethod
    def remove_file(self, file):
        """Removes specified files from the storage"""

    @abstractmethod
    def __getitem__(self, item):
        """Return stored file as Subclass of FileDataTemplate"""

    @property
    def requires_metadata(self):
        return False

    def setup_storage(self):
        pass

    def parse_metadata(self, metadata):
        return metadata

    @abstractmethod
    def validate_metadata(self, metadata, raise_error=True):
        """Check metadata for validity and provide valid metadata back.

        Args:
            metadata: the meta data object to check
            raise_error: if raise_error is True, errors are raised. Otherwise, silently returning None.
        Raises:
            ValueError: if the metadata is not valid and raise_error.
        Returns:
            object: valid meta data or None if metadata is not valid and not raise_error.
        """


class LocalFileStorage(StorageInterface):
    def upload_file(self, file, metadata=None, filename=None):
        self.put(file=file, filepath=filename)

    def __init__(self, path, root_path=None):
        super().__init__()
        self.path = path
        if root_path is not None:
            self._root_path = Path(root_path)
        else:
            self._root_path = root_path

    def _path_setter(self, new_path):
        path = Path(new_path).expanduser()
        if not path.is_dir():
            raise ValueError(f"Provided path '{new_path}' is not a directory.")
        return path.resolve()

    def create_group(self, name):
        new_path = self._join_path(name)
        if new_path.exists() and not new_path.is_dir():
            raise ValueError(f"{new_path} already exists and is not a directory.")
        elif not new_path.exists():
            new_path.mkdir(parents=True)

        return self.__class__(new_path)

    def is_file(self, item):
        return self._join_path(item).is_file()

    def is_dir(self, item):
        return self._join_path(item).is_dir()

    def _list_groups(self):
        return [g for g in os.listdir(self.path) if self.is_dir(g)]

    def _list_nodes(self):
        return [f for f in os.listdir(self.path) if self.is_file(f)]

    def _join_path(self, item):
        if os.path.isabs(item):
            return item
        return self._path.joinpath(item).resolve()

    def __getitem__(self, item):
        item_path = self._join_path(item)
        if item_path.is_file():
            return FileData(str(item_path))
        elif item_path.is_dir():
            return self.__class__(item_path)
        else:
            raise KeyError(f"No such file or directory: '{item}' in '{self._path}'")

    def validate_metadata(self, metadata, raise_error=True):
        state.logger.warn(
            "Storing metadata for LocalStorage is currently handled only on the job level."
        )
        return metadata

    def put(self, file, filepath=None, overwrite=False):
        if filepath is None:
            filepath = self._path
        else:
            filepath = self._join_path(filepath)

        if filepath.is_file() and not overwrite:
            raise ValueError(f"File already present and overwrite=False.")

        if isinstance(file, (str, Path)):
            shutil.copy2(file, filepath)
        else:
            # Expecting file to be a file handle
            if filepath.is_dir():
                filepath = filepath.joinpath(file.name)
            shutil.copyfileobj(file, filepath.open("wb"))

    def remove_file(self, filename, missing_ok=False):
        file = self._join_path(filename)
        if file.is_dir():
            raise ValueError(f"{filename} is a directory but expected a file.")
        file.unlink(missing_ok=missing_ok)


class JobFileStorage(LocalFileStorage):
    """The job storage operates on the usual working directory of the job"""

    def _path_setter(self, new_path):
        new_path = super()._path_setter(new_path)
        self._validate_path(new_path)
        self._path = new_path

    def __init__(self, working_directory):
        super().__init__(path=".", root_path=working_directory)

    def _join_path(self, item):
        new_path = super()._join_path(item)
        self._validate_path(new_path)
        return new_path

    def _validate_path(self, path):
        rel_path = os.path.relpath(path, self._root_path)
        if rel_path.startswith(".."):
            raise ValueError(
                f"New path not within the jobs working directory {self._root_path}"
            )

    def setup_storage(self):
        os.mkdir(self._root_path)
