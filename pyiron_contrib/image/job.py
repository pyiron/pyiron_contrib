from __future__ import print_function
# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron.base.job.generic import GenericJob
import numpy as np
import skimage as ski
from skimage import io, filters, exposure
import matplotlib.pyplot as plt
import inspect
from pyiron_contrib.image.utils import ModuleScraper

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


class Images(GenericJob):
    pass


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
                image.set_data(output, image.is_greyscale)
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

    def __init__(self, data=None, metadata=None, as_grey=False):
        # Set data
        self._source = None
        self._data = None
        self.is_greyscale = None
        self.set_data(data, as_grey=as_grey)

        # Set metadata
        self.metadata = metadata  # TODO

        # Apply wrappers
        # TODO:
        #  Set up some sort of metaclass so that the scraping and wrapping is done at import. It will be too expensive
        #  to do this every time we instantiate...
        for module_name in [
            'filters',
            'exposure'
        ]:
            # setattr(
            #     self,
            #     module_name,
            #     self._ModuleScraper(self, getattr(ski, module_name))
            # )
            setattr(
                self,
                module_name,
                ModuleScraper(
                    'skimage.' + module_name,
                    decorator=pass_and_set_image_data,
                    decorator_args=(self,)
                )
            )

    @property
    def data(self):
        return self._data

    def set_data(self, new_data, as_grey=False):
        self.is_greyscale = as_grey

        if isinstance(new_data, np.ndarray):
            self._data = new_data
            if self._source is None:
                self._source = new_data.copy()
        elif isinstance(new_data, str):
            self._data = ski.io.imread(new_data, as_grey=as_grey)
            self._source = new_data
        elif new_data is None:
            pass
        else:
            raise ValueError("Data type not understood, should be numpy.ndarray or string pointing to image file.")

    def reset_data(self):
        """
        Reverts the `data` attribute to the most recently read file (if set by reading data), or the originally
        assigned array (if set by direct array assignment).
        """
        self.set_data(self._source, self.is_greyscale)

    def convert_to_greyscale(self):
        if self.data is not None and len(self.data.shape) == 3 and self.data.shape[-1] == 3:
            self.set_data(self.data, as_grey=True)
        else:
            raise ValueError("Can only convert data with shape NxMx3 to greyscale")

    def imshow(self, subplots_kwargs=None, ax_kwargs=None):
        subplots_kwargs = subplots_kwargs or {}
        ax_kwargs = ax_kwargs or {}
        fig, ax = plt.subplots(**subplots_kwargs)
        ax.imshow(self.data, **ax_kwargs)
        return fig, ax


class Metadata:
    def __init__(self, data=None, note=None):
        self._data = data
        self.note = note

    @property
    def shape(self):
        return self._data.shape