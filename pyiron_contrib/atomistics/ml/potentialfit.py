"""
Abstract base class for fitting interactomic potentials.
"""

from typing import List, Optional

import abc

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np

from pyiron_base import FlattenedStorage
from pyiron_contrib.atomistics.atomistics.job.trainingcontainer import (
    TrainingContainer,
    TrainingStorage,
)


class PotentialFit(abc.ABC):
    """
    Abstract mixin that defines a general interface to potential fitting codes.

    Training data can be added to the job with :method:`~.add_training_data`.  This should be atom structures with
    (at least) corresponding energies and forces, but additional (per structure or atom) maybe added.  Subclasses of
    :class:`~.TrainingContainer` that define and handle such data are explicitly allowed.

    :property:`~.training_data` and :property:`~.predicted_data` can be used to access the initial training data and the
    predicted data on them after the fit.
    """

    @abc.abstractmethod
    def _add_training_data(self, container: TrainingContainer) -> None:
        pass

    def add_training_data(self, container: TrainingContainer) -> None:
        """
        Add data to the fit.

        Calling this multiple times appends data to internal storage.

        Args:
            container (:class:`.TrainingContainer`): container holding data to fit
        """
        if self.status.initialized:
            self._add_training_data(container)
        else:
            raise ValueError("Data can only be added before fitting is started!")

    @abc.abstractmethod
    def _get_training_data(self) -> TrainingStorage:
        pass

    @property
    def training_data(self) -> TrainingStorage:
        """
        Return all training data added so far.

        Returns:
            :class:`pyiron_contrib.atomistics.atomistics.job.trainingcontainer.TrainingStorage`: container holding all training data
        """
        return self._get_training_data()

    @abc.abstractmethod
    def _get_predicted_data(self) -> FlattenedStorage:
        pass

    @property
    def predicted_data(self) -> FlattenedStorage:
        """
        Predicted properties of the training data after the fit.

        In contrast to :property:`~.training_data` this may not contain the original atomic structures, but must be in
        the same order.  Certain properties in the training data may be omitted from this data set, if the inconvenient
        or impossible to predict.  This should be documented on the subclass for each specific code.

        Returns:
            :class:`pyiron_base.FlattenedStorage`: container holding all predictions of the fitted potential on the
                                                   training data
        """
        if self.status.finished:
            return self._get_predicted_data()
        else:
            raise ValueError("Data can only be accessed after successful fit!")

    @property
    def plot(self):
        """
        Plots correlation and (training) error histograms.
        """
        return PotentialPlots(self.training_data, self.predicted_data)

    @abc.abstractmethod
    def get_lammps_potential(self) -> pd.DataFrame:
        """
        Return a pyiron compatible dataframe that defines a potential to be used with a Lammps job (or subclass
        thereof).

        Returns:
            DataFrame: contains potential information to be used with a Lammps job.
        """
        pass


class PotentialPlots:
    def __init__(self, training_data, predicted_data):
        self._training_data = training_data
        self._predicted_data = predicted_data

    def energy_scatter_histogram(self, logy=False):
        """
        Plots correlation and (training) error histograms.

        Args:
            logy (bool): Use log scale for histogram heights
        """
        energy_train = self._training_data["energy"] / self._training_data["length"]
        energy_pred = self._predicted_data["energy"] / self._predicted_data["length"]
        plt.subplot(1, 2, 1)
        plt.scatter(energy_train, energy_pred)

        plt.xlabel("True Energy Per Atom [eV / atom]")
        plt.ylabel("Predicted Energy Per Atom [eV / atom]")
        plt.plot()

        plt.subplot(1, 2, 2)
        plt.hist(energy_train - energy_pred, log=logy)
        plt.xlabel("Training Error [eV / atom]")

    def energy_log_histogram(self, bins=20, logy=False):
        """
        Plots a histogram of logarithmic training errors.

        Bins are created automatically using the minimum and maximum absolute
        errors with the given number of bins.

        Arguments:
            bins (int, optional): number of bins for the histogram
            logy (bool, optional): if True use a log scale also for the y-axis
        """

        energy_train = self._training_data["energy"] / self._training_data["length"]
        energy_pred = self._predicted_data["energy"] / self._predicted_data["length"]
        de = abs(energy_train - energy_pred)
        rmse = np.sqrt((de**2).mean())
        mae = de.mean()
        high = de.max()
        low = de.min()

        ax = plt.gca()
        trafo = ax.get_xaxis_transform()

        def annotated_vline(x, text, linestyle="--"):
            plt.axvline(x, color="k", linestyle=linestyle)
            plt.text(
                x=x,
                y=0.5,
                s=text,
                transform=trafo,
                rotation="vertical",
                horizontalalignment="center",
                path_effects=[withStroke(linewidth=4, foreground="w")],
            )

        plt.hist(de, bins=np.logspace(np.log10(low), np.log10(high), bins), log=logy)
        plt.xscale("log")
        annotated_vline(rmse, f"RMSE = {rmse:.02}")
        annotated_vline(mae, f"MAE = {mae:.02}")
        annotated_vline(high, f"HIGH = {high:.02}", linestyle="-")
        annotated_vline(low, f"LOW = {low:.02}", linestyle="-")
        plt.xlabel("Training Error [eV/atom]")

    def force_scatter_histogram(self, axis=None, logy=False):
        """
        Plots correlation and (training) error histograms.

        Args:
            axis (None, int): Whether to plot for an axis or norm
            logy (bool): Use log scale for histogram heights
        """
        force_train = self._training_data["forces"]
        force_pred = self._predicted_data["forces"]

        if axis is None:
            ft = np.linalg.norm(force_train, axis=1)
            fp = np.linalg.norm(force_pred, axis=1)
        else:
            ft = force_train[:, axis]
            fp = force_pred[:, axis]

        plt.subplot(1, 2, 1)
        plt.scatter(ft, fp)
        plt.xlabel("True Forces [eV/$\mathrm{\AA}$]")
        plt.ylabel("Predicted Forces [eV/$\AA$]")
        plt.subplot(1, 2, 2)
        plt.hist(ft - fp, log=logy)
        plt.xlabel("Training Error [eV/$\AA$]")

    def force_log_histogram(
        self, bins: int = 20, logy: bool = False, axis: Optional[int] = None
    ):
        """
        Plots a histogram of logarithmic training errors.

        Bins are created automatically using the minimum and maximum absolute
        errors with the given number of bins.

        Arguments:
            bins (int, optional): number of bins for the histogram
            logy (bool, optional): if True use a log scale also for the y-axis
            axis (int, optional): which axis of the forces to plot; if not given plot force magnitude
        """

        force_train = self._training_data["forces"]
        force_pred = self._predicted_data["forces"]

        if axis is None:
            ft = np.linalg.norm(force_train, axis=1)
            fp = np.linalg.norm(force_pred, axis=1)
        else:
            ft = force_train[:, axis]
            fp = force_pred[:, axis]

        df = abs(ft - fp)
        rmse = np.sqrt((df**2).mean())
        mae = df.mean()
        high = df.max()
        low = df.min()

        ax = plt.gca()
        trafo = ax.get_xaxis_transform()

        def annotated_vline(x, text, linestyle="--"):
            plt.axvline(x, color="k", linestyle=linestyle)
            plt.text(
                x=x,
                y=0.5,
                s=text,
                transform=trafo,
                rotation="vertical",
                horizontalalignment="center",
                path_effects=[withStroke(linewidth=4, foreground="w")],
            )

        plt.hist(
            df, bins=np.logspace(np.log10(low + 1e-8), np.log10(high), bins), log=logy
        )
        plt.xscale("log")
        annotated_vline(rmse, f"RMSE = {rmse:.02}")
        annotated_vline(mae, f"MAE = {mae:.02}")
        annotated_vline(high, f"HIGH = {high:.02}", linestyle="-")
        annotated_vline(low, f"LOW = {low:.02}", linestyle="-")
        plt.xlabel("Training Error [eV/$\mathrm{\AA}$]")

    def force_angle_histogram(
        self,
        bins: int = 180,
        logy: bool = True,
        tol: float = 1e-6,
        angle_in_degrees=True,
        cumulative=False,
    ):
        """
        Plot histogram of the angle between training and predicted forces.

        Args:
            bins (int): number of bins
            logy (bool): Use log scale for histogram heights
            tol (float): consider forces smaller than this zero (and obmit them from the histogram)
            angle_in_degrees (bool): if True use degrees, otherwise radians
        """
        force_train = self._training_data["forces"]
        force_pred = self._predicted_data["forces"]

        force_norm_train = np.linalg.norm(force_train, axis=-1).reshape(-1, 1)
        force_norm_pred = np.linalg.norm(force_pred, axis=-1).reshape(-1, 1)

        I = ((force_norm_train > tol) & (force_norm_train > tol)).reshape(-1)

        force_dir_train = force_train[I] / force_norm_train[I]
        force_dir_pred = force_pred[I] / force_norm_pred[I]

        err = np.arccos((force_dir_train * force_dir_pred).sum(axis=-1).round(8))
        if angle_in_degrees:
            err = np.rad2deg(err)
        if cumulative:
            logy = False
        plt.hist(err, bins=bins, log=logy, cumulative=cumulative)
        plt.xlabel(
            "Angular Deviation of Force [" + ["rad", "deg"][angle_in_degrees] + "]"
        )
        plt.ylabel("Count")
