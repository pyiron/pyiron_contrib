"""
Abstract base class for fitting interactomic potentials.
"""

import abc

import pandas as pd
import matplotlib.pyplot as plt
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
        self.training_data = training_data
        self.predicted_data = predicted_data

    def energy_scatter_histogram(self):
        """
        Plots correlation and (training) error histograms.
        """
        energy_train = self.training_data["energy"] / self.training_data["length"]
        energy_pred = self.predicted_data["energy"] / self.predicted_data["length"]
        plt.subplot(1, 2, 1)
        plt.scatter(energy_train, energy_pred)
        plt.xlabel("True Energy Per Atom [eV]")
        plt.ylabel("Predicted Energy Per Atom [eV]")
        plt.plot()
        plt.subplot(1, 2, 2)
        plt.hist(energy_train - energy_pred)
        plt.xlabel("Training Error [eV]")

    def force_scatter_histogram(self, axis=None):
        """
        Plots correlation and (training) error histograms.

        Args:
            axis (None, int): Whether to plot for an axis or norm

        """
        force_train = self.training_data["forces"]
        force_pred = self.predicted_data["forces"]

        if axis is None:
            ft = np.linalg.norm(force_train, axis=1)
            fp = np.linalg.norm(force_pred, axis=1)
        else:
            ft = force_train[:, axis]
            fp = force_pred[:, axis]

        plt.subplot(1, 2, 1)
        plt.scatter(ft, fp)
        plt.xlabel("True Forces [eV$\AA$]")
        plt.ylabel("Predicted Forces [eV/$\AA$]")
        plt.subplot(1, 2, 2)
        plt.hist(ft - fp)
        plt.xlabel("Training Error [eV/$\AA$]")
