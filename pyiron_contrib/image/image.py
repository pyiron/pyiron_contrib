from __future__ import print_function
# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import inspect
from pkgutil import iter_modules

import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage import io
from collections import UserDict

from pyiron_contrib.image.utils import ModuleScraper

"""
Code for storing images in hdf5 and leveraging the skimage library as a class attribute.
"""

__author__ = "Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Feb 6, 2020"

# Some decorators look at the signature of skimage methods to see if they take an image
# (presumed to be in numpy.ndarray format).
# This is done by searching the signature for the variable name below:
_IMAGE_VARIABLE = 'image'


def pass_image_data(image):
    """
    Decorator to see if the signature of the function starts with a particular variable (`_IMAGE_VARIABLE`). If so,
    automatically passes an attribute of the argument (`image.data`) as the first argument.

    Args:
        image (Image): The image whose data to use.

    Returns:
        (fnc): Decorated function.
    """

    def decorator(function):
        takes_image_data = list(inspect.signature(function).parameters.keys())[0] == _IMAGE_VARIABLE

        def wrapper(*args, **kwargs):
            if takes_image_data:
                return function(image.data, *args, **kwargs)
            else:
                return function(*args, **kwargs)

        wrapper.__doc__ = ""
        if takes_image_data:
            wrapper.__doc__ += "This function has been wrapped to automatically supply the image argument. \n" \
                               "Remaining arguments can be passed as normal.\n"
        wrapper.__doc__ += "The original docstring follows:\n\n"
        wrapper.__doc__ += function.__doc__ or ""
        return wrapper

    return decorator


def set_image_data(image):
    """
    Decorator which checks the returned value of the function. If that value is of type `numpy.ndarray`, uses it to set
    an attribute of the argument (`image.data`) instead of returning it.

    Args:
        image (Image): The image whose data to set.

    Returns:
        (fnc): Decorated function.
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            output = function(*args, **kwargs)
            if isinstance(output, np.ndarray):
                image._data = output
            else:
                return output

        wrapper.__doc__ = "This function has been wrapped; if it outputs a numpy array, it will be " \
                          "automatically passed to the image's data field.\n" + function.__doc__
        return wrapper

    return decorator


def pass_and_set_image_data(image):
    """
    Decorator which connects function input and output to `image.data`.

    Args:
        image (Image): The image whose data to set.

    Returns:
        (fnc): Decorated function.
    """

    def decorator(function):
        return set_image_data(image)(pass_image_data(image)(function))
    return decorator


class Image:
    """
    A base class for storing image data in the form of numpy arrays. Functionality of the skimage library can be
    leveraged using the sub-module name and an `activate` method.

    Attributes:
        source (str/numpy.ndarray): The raw data source.
        data (numpy.ndarray): The image data. Not loaded until called, so first call may be slow. Modifications are
            made to this field, leaving the source untouched. NOT saved to hdf.
        as_gray (bool): Whether to interpret the image as grayscale.
        metadata (Metadata): Metadata associated with the source.
    """

    def __init__(self, source, metadata=None, as_gray=False):
        """
        source (str/numpy.ndarray): The raw data source.
        as_gray (bool): Whether to interpret the image as grayscale. (Default is False)
        metadata (Metadata): Metadata associated with the source. (Default is None.)
        """

        # Set data
        self._source = source
        self._data = None
        self.as_gray = as_gray

        # Set metadata
        self._metadata = None
        self.metadata = metadata or Metadata()

        # Apply wrappers
        submodule_blacklist = [
            'data',
            'scripts',
            'future',
            'registration',

        ]
        for module in iter_modules(skimage.__path__):
            if module.name[0] == '_' or module.name in submodule_blacklist:
                continue
            setattr(
                self,
                module.name,
                ModuleScraper(
                    'skimage.' + module.name,
                    decorator=pass_and_set_image_data,
                    decorator_args=(self,)
                )
            )

    @property
    def source(self):
        return self._source

    def overwrite_source(self, new_source, new_metadata=None, as_gray=False):
        """
        Apply a new source of image data to the image object.

        Args:
            new_source (str/numpy.ndarray): The filepath to the data, or the raw array of data itself.
            new_metadata (Metadata): The metadata associated with the new source. (Default is None.)
            as_gray (bool): Whether to interpret the new data as grayscale. (Default is False.)
        """

        self._source = new_source
        self._data = None
        self.as_gray = as_gray
        self.metadata = new_metadata or Metadata()

    @property
    def data(self):
        if self._data is None:
            self._load_data_from_source()
        return self._data

    @property
    def shape(self):
        return self.data.shape

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, new_metadata):
        if new_metadata is None or isinstance(new_metadata, Metadata):
            self._metadata = new_metadata
        elif isinstance(new_metadata, dict):
            self._metadata = Metadata(new_metadata)
        else:
            raise ValueError("Metadata field expected a `dict`, `Metadata`, or `None`, but got {}".format(
                type(new_metadata))
            )

    def __len__(self):
        return self.data.__len__()

    def _load_data_from_source(self):
        if isinstance(self.source, np.ndarray):
            self._data = self.source.copy()
            if len(self._data.shape) == 3:
                self.convert_to_grayscale()
        elif isinstance(self.source, str):
            self._data = io.imread(self.source, as_gray=self.as_gray)
        else:
            raise ValueError("Data source not understood, should be numpy.ndarray or string pointing to image file.")

    def reload_data(self):
        """
        Reverts the `data` attribute to the source, i.e. the most recently read file (if set by reading data), or the
        originally assigned array (if set by direct array assignment).
        """

        self._load_data_from_source()

    def convert_to_grayscale(self):
        """
        Flattens (NxMx3) data into (NxM) grayscale data.
        """
        if self._data is not None:
            if len(self.data.shape) == 3 and self.data.shape[-1] == 3:
                self._data = np.mean(self._data, axis=-1)
                self.as_gray = True
            else:
                raise ValueError("Can only convert data with shape NxMx3 to grayscale")
        else:
            self.as_gray = True

    def plot(self, ax=None, subplots_kwargs=None, imshow_kwargs=None, hide_axes=True):
        """
        Make a simple matplotlib `imshow` plot of the data.

        Args:
            ax (matplotlib.axes.Axes): The axis to plot on. (Default is None, make a new figure.)
            subplots_kwargs (dict): Keyword arguments to pass to the figure generation. Only used if no axis is
                provided. (Default is None.)
            imshow_kwargs (dict): Keyword arguments to pass to the `imshow` plotting command. (Default is None.)
            hide_axes (bool): Whether to hide axis ticks and labels. (Default is True.)

        Returns:
            (matplotlib.figure.Figure): The figure the plot is in.
            (matplotlib.axes.Axes): The axis the plot is on.
        """

        subplots_kwargs = subplots_kwargs or {}
        imshow_kwargs = imshow_kwargs or {}

        if ax is None:
            fig, ax = plt.subplots(**subplots_kwargs)
        else:
            fig = ax.figure

        ax.imshow(self.data, **imshow_kwargs)

        if hide_axes:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        return fig, ax

    def to_hdf(self, hdf, group_name=None):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server["TYPE"] = str(type(self))
            hdf5_server["source"] = self.source
            hdf5_server["as_gray"] = self.as_gray
            self.metadata.to_hdf(hdf=hdf5_server, group_name="metadata")

    def from_hdf(self, hdf, group_name=None):
        with hdf.open(group_name) as hdf5_server:
            source = hdf5_server["source"]
            as_gray = hdf5_server["as_gray"]
            metadata = Metadata()
            metadata.from_hdf(hdf=hdf5_server, group_name="metadata")
        self.overwrite_source(source, new_metadata=metadata, as_gray=as_gray)


class Metadata(UserDict):
    """
    TODO: Leverage the generic to and from hdf functions written by Dominik over in pyiron_contrib/protocol
    """

    def __getattr__(self, item):
        return self.data[item]
        # return super(Metadata, self).__getitem__(item)

    def __setattr__(self, key, value):
        if key == "data":
            self.__dict__[key] = value
        else:
            self.__dict__['data'][key] = value

    def to_hdf(self, hdf, group_name=None):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server["TYPE"] = str(type(self))
            hdf5_server["KEYS"] = list(self.keys())
            for k, v in self.items():
                hdf5_server[k] = v

    def from_hdf(self, hdf, group_name=None):
        with hdf.open(group_name) as hdf5_server:
            keys = hdf5_server["KEYS"]
            for k in keys:
                self[k] = hdf5_server[k]
