from __future__ import print_function
# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron.base.job.generic import GenericJob
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile
from glob import iglob

from pyiron_contrib.image.image import Image
from pyiron_contrib.image.utils import DistributingList

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
    A basic job type for storing image data.
    """

    def __init__(self, project, job_name):
        super(ImageJob, self).__init__(project, job_name)
        self.__name__ = "ImageJob"
        self._images = DistributingList()

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, val):
        if isinstance(val, DistributingList):
            self._images = val
        elif isinstance(val, (tuple, list, np.ndarray)):
            self._images = DistributingList(val)
        else:
            raise ValueError("Images was expecting a list-like object, but got {}".format(type(val)))

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

    def add_image(self, source, metadata=None, as_grey=False):
        if not isfile(source):
            raise ValueError("Could not find a file at {}".format(source))
        self.images.append(Image(source=source, metadata=metadata, as_grey=as_grey))

    def add_images(self, sources, metadata=None, as_grey=False):
        if isinstance(sources, str):
            for match in iglob(sources):
                self.add_image(match, metadata=metadata, as_grey=as_grey)
        elif isinstance(sources, (list, tuple, np.ndarray)):
            for source in sources:
                self.add_image(source, metadata=metadata, as_grey=as_grey)

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
