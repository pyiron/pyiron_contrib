"""
Abstract base class for fitting interactomic potentials.
"""

import abc

import pandas as pd

from pyiron_base import FlattenedStorage
from pyiron_contrib.atomistics.atomistics.job.trainingcontainer import TrainingContainer

class PotentialFit(abc.ABC):
    """
    Abstract mixin that defines a general interface to potential fitting codes.

    Training data can be added to the job with :method:`~.add_training_data`.  This should be atom structures with
    (at least) corresponding energies and forces, but additional (per structure or atom) maybe added.  Subclasses of
    :class:`~.TrainingContainer` that define and handle such data are explicitly allowed.

    :property:`~.training_data` and :property:`~.predicted_data` can be used to access 
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

    @abstract
    def _get_training_data(self) -> FlattenedStorage:
        pass

    @property
    def training_data(self) -> FlattenedStorage:
        """
        Return all training data added so far.

        Returns:
            :class:`pyiron_base.FlattenedStorage`: container holding all training data
        """
        return self._get_training_data()

    @abstract
    def _get_predicted_data(self) -> FlattenedStorage:
        pass

    @property
    def predicted_data(self) -> FlattenedStorage:
        """
        Predicted properties of the training data after the fit.

        Returns:
            :class:`pyiron_base.FlattenedStorage`: container holding all predictions of the fitted potential on the
                                                   training data
        """
        if self.status.finished:
            return self._get_predicted_data()
        else:
            raise ValueError("Data can only be accessed after successful fit!")

    def plot(self):
        """
        Plots correlation and (training) error histograms.
        """
        if not self.status.finished:
            raise ValueError("Results can only be plotted after job finished successfully!")

        energy_train = self.training_data["energy"] / self.training_data["length"]
        energy_pred = self.predicted_data["energy"] / self.predicted_data["length"]

        plt.subplot(1, 2, 1)
        plt.scatter(energy_train, energy_pred)
        plt.xlabel("True Energy Per Atom [eV]")
        plt.ylabel("Predicted Energy Per Atom [eV]")

        plt.subplot(1, 2, 2)
        plt.hist(energy_train - energy_pred)
        plt.xlabel("Training Error [eV]")

    @abstract
    def get_lammps_potential(self) -> pd.DataFrame:
        """
        Return a pyiron compatible dataframe that defines a potential to be used with a Lammps job (or subclass
        thereof).

        Returns:
            DataFrame: contains potential information to be used with a Lammps job.
        """
        pass
