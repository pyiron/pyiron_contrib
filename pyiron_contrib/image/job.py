from __future__ import print_function
# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron.base.job.generic import GenericJob
import numpy as np
from types import FunctionType
import skimage as ski
import matplotlib.pyplot as plt
import inspect

"""
Store and process image data within the pyiron framework.
"""

__author__ = "Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Jan 30, 2020"


class Images(GenericJob):
    pass


class Image:
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
            setattr(
                self,
                module_name,
                self._ModuleScraper(self, getattr(ski, module_name))
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

    class _ModuleScraper:
        """
        Scrapes all the methods and attributes from a module. Methods are wrapped to automatically pass and collect
        image data if the signature/output type indicates this is reasonable. Sub-modules are collected using by
        recursively instantiating this class. A limited subset of 'primitive' datatypes are also captured.

        Warning:
            There is currently no automated check that the signatures from these functions are compatible with this
            behaviour.
        """

        # Variable name from the skimage library. Assumed to be the first positional argument when it occurs.
        _IMAGE_VARIABLE = 'image'

        def __init__(self, image, module, scrape_submodules=False):
            """
            Args:
                image (Image): The image to apply modifying functions to.
                module (module): The module from which to scrape.
            """
            for name, obj in inspect.getmembers(module):
                if name[0] == '_':
                    continue

                primitives = (int, float, bool, np.ndarray)
                if isinstance(obj, FunctionType):
                    setattr(self, name, self.set_image_data(image)(self.pass_image_data(image)(obj)))
                elif isinstance(obj, type(module)) and scrape_submodules and obj.__package__ == module.__package__:
                    print("Scraping subclass {}".format(name))
                    setattr(self, name, self.__class__(image, obj))
                elif isinstance(obj, primitives):
                    setattr(self, name, obj)

        def pass_image_data(self, image):
            def decorator(function):
                takes_image_data = list(inspect.signature(function).parameters.keys())[0] == self._IMAGE_VARIABLE

                def wrapper(*args, **kwargs):
                    if takes_image_data:
                        return function(image.data, *args, **kwargs)
                    else:
                        return function(image.data, *args, **kwargs)

                wrapper.__doc__ = ""
                if takes_image_data:
                    wrapper.__doc__ += "This function has been wrapped to automatically supply the image argument. \n" \
                                       "Remaining arguments can be passed as normal.\n"
                wrapper.__doc__ += "The original docstring follows:\n\n"
                wrapper.__doc__ += function.__doc__ or ""
                return wrapper
            return decorator

        def set_image_data(self, image):
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

        # [Alegedly](https://hynek.me/articles/decorators/) , the wrapt library will let me preserve docstrings *and*
        # signatures, but I can't figure it out


class Metadata:
    def __init__(self, data=None, note=None):
        self._data = data
        self.note = note

    @property
    def shape(self):
        return self._data.shape