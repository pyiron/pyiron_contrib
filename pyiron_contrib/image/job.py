from __future__ import print_function
# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron.base.job.generic import GenericJob
import numpy as np
import skimage
from skimage import io
import matplotlib.pyplot as plt
import inspect
from pyiron_contrib.image.utils import ModuleScraper
from pkgutil import iter_modules

"""
Store and process image data within the pyiron framework. Functionality of the `skimage` library is automatically 
scraped, along with some convenience decorators to switch their function-based library to a class-method-based library.
"""

__author__ = "Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Jan 30, 2020"

# Some decorators look at the signature of skimage methods to see if they take an image
# (presumed to be in numpy.ndarray format).
# This is done by searching the signature for the variable name below:
_IMAGE_VARIABLE = 'image'


class ImageJob(GenericJob):
    """
    A basic job type for storing image data.
    """

    def __init__(self, project, job_name):
        super(ImageJob, self).__init__(project, job_name)
        self.__name__ = "ImageJob"
        self.images = []

    @staticmethod
    def get_factors(n):
        i = int(n**0.5 + 0.5)
        while n % i != 0:
            i -= 1
        return i, int(n/i)

    def plot(self, subplots_kwargs=None, imshow_kwargs=None):
        subplots_kwargs = subplots_kwargs or {}
        imshow_kwargs = imshow_kwargs or {}
        nrows, ncols = self.get_factors(len(self.images))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **subplots_kwargs)
        axes = np.atleast_2d(axes)
        for n, img in enumerate(self.images):
            i = int(np.floor(n / ncols))
            j = n % ncols
            ax = axes[i, j]
            img.plot(ax=ax, imshow_kwargs=imshow_kwargs)

        fig.tight_layout()
        return fig, axes

    def run(self, run_again=False, repair=False, debug=False, run_mode=None):
        # This is just a place holder to stop the job from even *starting* to run until run_static is implemented
        raise NotImplementedError

    def run_static(self):
        # TODO: Take a modifier chain as input and apply all modifiers on run
        raise NotImplementedError

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the job in an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(ImageJob, self).to_hdf(hdf=hdf, group_name=group_name)
        if hdf is None:
            hdf = self.project_hdf5
        with hdf.open("images") as hdf5_server:
            for n, image in enumerate(self.images):
                image.to_hdf(hdf=hdf5_server, group_name="img{}".format(n))
        hdf["n_images"] = n + 1

    def from_hdf(self, hdf=None, group_name=None):
        """
        Load the Protocol from an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(ImageJob, self).to_hdf(hdf=hdf, group_name=group_name)
        if hdf is None:
            hdf = self.project_hdf5
        with hdf.open("images") as hdf5_server:
            for n in np.arange(hdf["n_images"], dtype=int):
                img = Image()
                img.from_hdf(hdf=hdf5_server, group_name="img{}".format(n))
                self.images.append(img)


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
    """

    def __init__(self, source=None, metadata=None, as_grey=False):
        # Set data
        self._source = source
        self._data = None
        self.as_grey = as_grey

        # Set metadata
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

    def overwrite_source(self, new_source, new_metadata=None, as_grey=False):
        self._source = new_source
        self._data = None
        self.as_grey = as_grey
        self.metadata = new_metadata or Metadata()

    @property
    def data(self):
        if self._data is None:
            self._load_data_from_source()
        return self._data

    @property
    def shape(self):
        return self.data.shape

    def __len__(self):
        return self.data.__len__()

    def _load_data_from_source(self):
        if isinstance(self.source, np.ndarray):
            self._data = self.source.copy()
        elif isinstance(self.source, str):
            self._data = io.imread(self.source, as_grey=self.as_grey)
        else:
            raise ValueError("Data source not understood, should be numpy.ndarray or string pointing to image file.")

    def reload_data(self):
        """
        Reverts the `data` attribute to the source, i.e. the most recently read file (if set by reading data), or the
        originally assigned array (if set by direct array assignment).
        """
        self._load_data_from_source()

    def convert_to_greyscale(self):
        if self._data is not None:
            if len(self.data.shape) == 3 and self.data.shape[-1] == 3:
                self._data = np.mean(self._data, axis=-1)
                self.as_grey = True
            else:
                raise ValueError("Can only convert data with shape NxMx3 to greyscale")
        else:
            self.as_grey = True

    def plot(self, ax=None, subplots_kwargs=None, imshow_kwargs=None, hide_axes=True):
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
            hdf5_server["as_grey"] = self.as_grey
            self.metadata.to_hdf(hdf=hdf5_server, group_name="metadata")

    def from_hdf(self, hdf, group_name=None):
        with hdf.open(group_name) as hdf5_server:
            source = hdf5_server["source"]
            as_grey = hdf5_server["as_grey"]
            metadata = Metadata()
            metadata.from_hdf(hdf=hdf5_server, group_name="metadata")
        self.overwrite_source(source, new_metadata=metadata, as_grey=as_grey)


class Metadata:
    def __init__(self, text=None):
        self.text = text

    def to_hdf(self, hdf, group_name=None):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server["TYPE"] = str(type(self))
            hdf5_server["text"] = self.text

    def from_hdf(self, hdf, group_name=None):
        with hdf.open(group_name) as hdf5_server:
            self.text = hdf5_server["text"]
