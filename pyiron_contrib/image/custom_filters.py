from __future__ import print_function
# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Jan 30, 2020"


def brightness_filter(image, sigma=10, bins=100, deg=10, plot=False):
    """
    Automatically detect a threshold between 'dark' and 'light' pixel values by looking for a minima in the pixel value
    histogram, fitting a polynomial, and finding the trough between the two highest peaks.

    Just exists as a near-trivial example

    Args:
        image (image.Image): The image to filter.
        sigma (float): The amount of gaussian smearing to apply to the image data before binning brightness. (Default is
            10.)
        bins (int): How many bins to put the pixel brightness into. (Default is 100)
        deg (int): Degree of polynomial to fit to the binned brightness data. (Default is 10.)
        plot (bool): Whether to plot a summary of the filtering process. (Default is False.)

    Returns:
        (float): The fraction of pixels darker than the threshold.
        (float): The threshold used.
        (numpy.ndarray): The mask of values from the smoothed image below the threshold.
    """
    # Smooth out to avoid checker-board of secondary phase
    image.reload_data()
    raw_data = image.data.copy()
    image.filters.gaussian(sigma=sigma)
    signal = image.data.flatten()

    # Find 'dark' and 'light' signal peaks
    counts, edges = np.histogram(image.data.flatten(), bins=bins)
    bincenters = 0.5 * (edges[:-1] + edges[1:])
    coeffs = np.polyfit(bincenters, counts, deg=deg)
    poly = np.poly1d(coeffs)
    extrema = poly.deriv().r[1:-1]
    extremal_bin_ids = np.array([np.argmin(np.abs((bincenters - x))) for x in extrema], dtype=int)

    # Find the trough location between the two biggest peaks
    ultimate_id, penultimate_id = np.argsort(poly(bincenters[extremal_bin_ids]))[-2:]
    trough_id = extremal_bin_ids[int(0.5 * (ultimate_id + penultimate_id))]
    peak_id, pen_peak_id = extremal_bin_ids[ultimate_id], extremal_bin_ids[penultimate_id]
    if not (bincenters[peak_id] < bincenters[trough_id] < bincenters[pen_peak_id]) and \
            not (bincenters[peak_id] > bincenters[trough_id] > bincenters[pen_peak_id]):
        print("WARNING: Had trouble finding a dividing trough for {}".format(image.source))
    threshold_signal = bincenters[trough_id]

    # Separate phases based on lightness/darkness
    phase_mask = image.data < threshold_signal
    phase_fraction = np.mean(phase_mask)

    if plot:
        # Plot the signal distribution
        _, ax = plt.subplots(figsize=(12, 6))
        sns.distplot(signal, ax=ax, norm_hist=False, kde=False, label='brightness distribution')
        ax.plot(bincenters, poly(bincenters), label='fit')

        ax.axvline(bincenters[peak_id], color='w', linestyle='--')
        ax.axvline(bincenters[trough_id], color='w', label='threshold')
        ax.axvline(bincenters[pen_peak_id], color='w', linestyle='--')
        plt.legend()
        plt.xlabel('Brightness of smoothed image')
        plt.ylabel('Pixel count')
        plt.show()

        # And the resulting map
        _, axes = plt.subplots(ncols=3, figsize=(12, 6))
        axes[0].imshow(raw_data)
        axes[0].set_title('Original')
        axes[1].imshow(image.data)
        axes[1].set_title('Smoothed')
        axes[2].imshow(phase_mask)
        axes[2].set_title('Phase mask')
        plt.show()

    return phase_fraction, threshold_signal, phase_mask