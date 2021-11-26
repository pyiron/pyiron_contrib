"""Generic File Object."""

# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from abc import ABC
import os

from pyiron_base.generic.filedata import FileDataTemplate as BaseFileDataTemplate, load_file

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


class FileDataTemplate(BaseFileDataTemplate, ABC):
    @staticmethod
    def _get_filetype_from_filename(filename):
        filetype = os.path.splitext(filename)[1]
        if filetype == '' or filetype == '.':
            filetype = None
        else:
            filetype = filetype[1:]
        return filetype

