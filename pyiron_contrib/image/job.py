from __future__ import print_function
# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base import GenericJob
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile
from glob import iglob
from os.path import abspath

from pyiron_contrib.image.image import Image
from pyiron_contrib.image.utils import DistributingList
from pyiron_base.generic.inputlist import InputList

from pyiron_contrib.image.custom_filters import brightness_filter

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


class ImageJob(GenericJob):
    """
    A job type for storing and processing image data.

    TODO: Consider allowing the `data` field of each image to be saved to hdf5...

    Attributes:
        images (DistributingList): A list of `Image` objects.
    """

    def __init__(self, project, job_name):
        super(ImageJob, self).__init__(project, job_name)
        self.__name__ = "ImageJob"
        self._images = DistributingList()
        self.input = InputList(table_name ="input")
        self.output = InputList(table_name ="output")


    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, val):
        if isinstance(val, DistributingList):
            self._images = val
        elif isinstance(val, (tuple, list, np.ndarray)):
            if not all([isinstance(obj, Image) for obj in val]):
                raise ValueError("Only `Image`-type objects can be set to the `images` attribute.")
            self._images = DistributingList(val)
        else:
            raise ValueError("Images was expecting a list-like object, but got {}".format(type(val)))

    @staticmethod
    def _get_factors(n):
        i = int(n**0.5 + 0.5)
        while n % i != 0:
            i -= 1
        return i, int(n/i)

    def plot(self, mask=None, subplots_kwargs=None, imshow_kwargs=None, hide_axes=True):
        """
        Make a simple matplotlib `imshow` plot for each of the images on a grid.

        Args:
            mask (list/numpy.ndarray): An integer index mask for selecting a subset of the images to plot.
            subplots_kwargs (dict): Keyword arguments to pass to the figure generation. (Default is None.)
            imshow_kwargs (dict): Keyword arguments to pass to the `imshow` plotting command. (Default is None.)
            hide_axes (bool): Whether to hide axis ticks and labels. (Default is True.)

        Returns:
            (matplotlib.figure.Figure): The figure the plots are in.
            (list): The axes the plot is on.
        """

        if mask is not None:
            images = self.images[mask]
        else:
            images = self.images

        subplots_kwargs = subplots_kwargs or {}
        imshow_kwargs = imshow_kwargs or {}
        nrows, ncols = self._get_factors(len(images))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **subplots_kwargs)
        axes = np.atleast_2d(axes)
        for n, img in enumerate(images):
            i = int(np.floor(n / ncols))
            j = n % ncols
            ax = axes[i, j]
            img.plot(ax=ax, imshow_kwargs=imshow_kwargs, hide_axes=hide_axes)

        fig.tight_layout()
        return fig, axes

    def add_image(self, source, metadata=None, as_gray=False, relative_path=True):
        """
        Add an image to the job.

        Args:
            source (str/numpy.ndarray): The filepath to the data, or the raw array of data itself.
            metadata (Metadata): The metadata associated with the source. (Default is None.)
            as_gray (bool): Whether to interpret the new data as grayscale. (Default is False.)
            relative_path (bool): Whether the path provided is relative. (Default is True, automatically converts to an
                absolute path before setting the `source` value of the image.)
        """

        if not isfile(source) and not isinstance(source, np.ndarray):
            raise ValueError("Could not find a file at {}, nor is source an array.".format(source))
        if isinstance(source, str) and relative_path:
            source = abspath(source)
        self.images.append(Image(source=source, metadata=metadata, as_gray=as_gray))

    def add_images(self, sources, metadata=None, as_gray=False):
        """
        Add multiple images to the job.

        Args:
            sources (str/list/tuple/numpy.ndarray): When a string, uses the `glob` module to look for matching files.
                When list-like, iteratively uses each element as a new source.
            metadata (Metadata): The metadata associated with all these sources. (Default is None.)
            as_gray (bool): Whether to interpret all this data as grayscale. (Default is False.)
            relative_path (bool): Whether the path provided is relative. (Default is True, automatically converts to an
                absolute path before setting the `source` value of the image.)
        """

        if isinstance(sources, str):
            for match in iglob(sources):
                self.add_image(match, metadata=metadata, as_gray=as_gray)
        elif isinstance(sources, (list, tuple, np.ndarray)):
            for source in sources:
                self.add_image(source, metadata=metadata, as_gray=as_gray)

    def run(self, run_again=False, repair=False, debug=False, run_mode=None):
        super(ImageJob, self).run(run_again=run_again, repair=repair, debug=debug, run_mode=run_mode)

    def run_static(self):
        """This is just a toy example right now."""
        self.status.running = True
        if hasattr(self.input, 'filter') and self.input.filter == 'brightness_filter':
            fractions = []
            cutoffs = []
            masks = []
            for img in self.images:
                frac, cut, mask = brightness_filter(img)
                fractions.append(frac)
                cutoffs.append(cut)
                masks.append(mask)
            self.output.fractions = np.array(fractions)
            self.output.cutoffs = np.array(cutoffs)
            self.output.masks = np.array(masks)

        else:
            self.logger.warning("Didn't run anything. Check input.")
        self.status.collect = True
        self.run()

    def write_input(self):
        """Must define abstract method"""
        pass

    def collect_output(self):
        """Must define abstract method"""
        self.to_hdf()

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the job in an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(ImageJob, self).to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(hdf=self._hdf5, group_name=None)
        self.output.to_hdf(hdf=self._hdf5, group_name=None)
        with self._hdf5.open("images") as hdf5_server:
            for n, image in enumerate(self.images):
                image.to_hdf(hdf=hdf5_server, group_name="img{}".format(n))
        self._hdf5["n_images"] = n + 1

    def from_hdf(self, hdf=None, group_name=None):
        """
        Load the Protocol from an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(ImageJob, self).from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(hdf=self._hdf5, group_name=None)
        self.output.from_hdf(hdf=self._hdf5, group_name=None)
        with self._hdf5.open("images") as hdf5_server:
            for n in np.arange(self._hdf5["n_images"], dtype=int):
                img = Image(source=None)
                img.from_hdf(hdf=hdf5_server, group_name="img{}".format(n))
                self.images.append(img)

