"""Generic File Object."""

# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
import abc
import posixpath
import shutil

from abc import ABC
import os
from os import path

from pyiron_base import GenericJob
from pyiron_base.generic.filedata import FileDataTemplate as BaseFileDataTemplate, load_file, FileData

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

from pyiron_base.interfaces.has_groups import HasGroups


class FileDataTemplate(BaseFileDataTemplate, ABC):
    @staticmethod
    def _get_filetype_from_filename(filename):
        filetype = os.path.splitext(filename)[1]
        if filetype == '' or filetype == '.':
            filetype = None
        else:
            filetype = filetype[1:]
        return filetype


class StorageInterface(HasGroups, abc.ABC):
    """File handling in different storage interfaces"""

    @abc.abstractmethod
    def upload_file(self, file):
        """Upload the provided files to the storage"""

    @abc.abstractmethod
    def remove_file(self, file):
        """Removes specified files from the storage"""

    @abc.abstractmethod
    def __getitem__(self, item):
        """Return stored file as Subclass of FileDataTemplate"""

    @property
    def requires_metadata(self):
        return False

    def setup_storage(self):
        pass

    def _validate_metadata(self, metadata):
        return True

    def validate_metadata(self, metadata):
        if metadata is None and self.requires_metadata:
            return False
        elif metadata is not None:
            return self._validate_metadata(metadata)
        else:
            return True


class LocalStorage(StorageInterface):
    """The local storage operates on the usual working directory of the job"""
    def __init__(self, job: GenericJob):
        self._job = job

    def upload_file(self, file, filename=None):
        filename = filename or os.path.basename(file)
        shutil.copy(file, os.path.join(self._job.working_directory, filename))

    def remove_file(self, file):
        os.remove(os.path.join(self._job.working_directory, file))

    def setup_storage(self):
        self._job._create_working_directory()

    def __getitem__(self, item):
        if item in self.list_nodes():
            file_name = posixpath.join(self._job.working_directory, f"{item}")
            if hasattr(self._job, '_stored_files'):
                metadata = self._job._stored_files[item]
            else:
                metadata = None
            return FileData(file=file_name, metadata=metadata)
        pass

    def _list_groups(self):
        """Every files is expected to be stored in the working directory - thus, no nesting of groups."""
        return []

    def _list_nodes(self):
        return self._job.list_files()
