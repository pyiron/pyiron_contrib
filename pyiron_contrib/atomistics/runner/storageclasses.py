# coding: utf-8
# Copyright (c) Georg-August-Universität Göttingen - Behler Group
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""Class Overrides for all `runnerase` storage classes.

The ASE calculator for RuNNer, provided by the Python package `runnerase`,
stores data like the values of symmetry functions or the weights of atomic
neural networks in custom classes. However, these classes generally do not know
how to write themselves to HDF5 storage format. Therefore, these classes are
extended in this module with additional functions, typically `to_hdf(...)` and
`from_hdf(...)`.

Attributes:
     HDFSymmetryFunctionValues (FlattenedStorage): Storage container for
        symmetry function values.
    RunneraseHDFMixin (HasHDF): Abstract mixin for all classes that store
        RuNNer results to HDF.
    HDFSymmetryFunctionSet (SymmetryFunctionSet, RunneraseHDFMixin): Storage
        container for a set of symmetry functions.
    HDFSplitTrainTest (RunnerSplitTrainTest, RunneraseHDFMixin): Storage
        container for the splitting between training and testing dataset.
    HDFFitResults (RunnerFitResults, RunneraseHDFMixin): Storage container
        for the results of a RuNNer fit.
    HDFWeights (RunnerWeights, RunneraseHDFMixin): Storage container for the
        weights of atomic neural networks.
    HDFScaling (RunnerScaling, RunneraseHDFMixin): Storage container for the
        symmetry function scaling data.

.. _RuNNer online documentation:
   https://theochem.gitlab.io/runner
"""

from typing import Optional, Union
from abc import abstractmethod

import numpy as np

from runnerase.symmetryfunctions import SymmetryFunctionSet
from runnerase.storageclasses import (
    RunnerSymmetryFunctionValues,
    RunnerStructureSymmetryFunctionValues,
    RunnerSplitTrainTest,
    RunnerFitResults,
    RunnerWeights,
    RunnerScaling,
)
from pyiron_base import ProjectHDFio
from pyiron_base import HasHDF
from pyiron_base import FlattenedStorage

from .utils import pad, unpad


class HDFSymmetryFunctionValues(FlattenedStorage):
    """Extend runnerase RunnerSymmetryFunctionValues with HDF5 compatibility."""

    __hdf_version__ = "0.3.0"

    def from_runnerase(self, runnerase_sfvalues: RunnerSymmetryFunctionValues) -> None:
        """Fill `self` with information of the corresponding `runnerase` object.

        `runnerase` stores the symmetry function values of each structure in
        a separate `RunnerStructureSymmetryFunctionValues` object. However,
        it is very inefficient to read and write all of this information
        separately into HDF5 storage. Therefore, we use a `FlattenedStorage`
        where all the symmetry function values of all structures are stored
        in one large flat array.

        Matters are complicated further because `FlattenedStorage` only accepts
        data chunks (read here: structures) where all different arrays of
        information have the same length. Unfortunately, one can have different
        numbers of symmetry functions for each element and varying numbers of
        elements for each structure in a training dataset.

        The solution here is to `pad` the sfvalues of each group of elements in
        one structure with rows filled with `np.NaN`. This way, all arrays for
        one structure have length `number_of_atoms` and can be efficiently
        stored and retrieved even for large  datasets.

        Parameters
        ----------
        runnerase_sfvalues : RunnerSymmetryFunctionValues
            A `runnerase` class object containing the symmetry function values
            for a whole dataset.
        """
        for structure_sfvalues in runnerase_sfvalues.data:
            # Preprocess the symmetry function value arrays.
            sfvalues_arrays = {}
            for element, sfvalues in structure_sfvalues.data.items():
                name = f"sfvalues_{element}"
                shape = (sfvalues.shape[1],)
                # Add arrays for storing the symmetry function values of all
                # atoms of the same `element`, unless they are already present.
                if name not in self.list_arrays():
                    self.add_array(
                        name, shape=shape, dtype=np.float64, per="element", fill=np.NaN
                    )

                # Pad the sfvalues of the current `element` with `np.NaN` rows,
                # so that the total length is equal to the number of atoms in
                # the structure.
                sfvalues_padded = pad(sfvalues, len(structure_sfvalues))

                sfvalues_arrays[name] = sfvalues_padded

            self.add_chunk(
                len(structure_sfvalues),
                energy_total=structure_sfvalues.energy_total,
                energy_short=structure_sfvalues.energy_short,
                energy_elec=structure_sfvalues.energy_elec,
                charge=structure_sfvalues.charge,
                **sfvalues_arrays,
            )

    def to_runnerase(self) -> RunnerSymmetryFunctionValues:
        """Create the corresponding `runnerase` object from `self`.

        Returns
        ----------
        runnerase_sfvalues : RunnerSymmetryFunctionValues
            A `runnerase` class object containing the symmetry function values
            for a whole dataset.
        """
        runnerase_sfvalues = RunnerSymmetryFunctionValues()

        for chunk_idx in range(len(self)):
            # Create a new object for storing the sfvalues of one structure.
            struct_sfvalues = RunnerStructureSymmetryFunctionValues()

            # Fill the object with per-chunk properties.
            struct_sfvalues.energy_total = self["energy_total", chunk_idx]
            struct_sfvalues.energy_short = self["energy_short", chunk_idx]
            struct_sfvalues.energy_elec = self["energy_elec", chunk_idx]
            struct_sfvalues.charge = self["charge", chunk_idx]

            # Read the symmetry function values.
            for arrayname in self.list_arrays():
                if arrayname.startswith("sfvalues"):
                    element = arrayname.split("_")[1]
                    sfvalues = self[f"sfvalues_{element}", chunk_idx]
                    struct_sfvalues.data[element] = unpad(sfvalues)

            # Append the structure to the container object
            # `RunnerSymmetryFunctionValues`.
            runnerase_sfvalues.data.append(struct_sfvalues)

        return runnerase_sfvalues


class RunneraseHDFMixin(HasHDF):
    """Abstract Mixin to add HDF5 compatibility to runnerase classes."""

    __hdf_version__ = "0.3.0"

    @property
    @abstractmethod
    def runnerase_properties(self):
        """Define the class properties stored in `self.baseclass` objects."""
        ...

    @property
    @abstractmethod
    def baseclass(self):
        """Define the runnerase class which is wrapped by this HDF class."""
        ...

    def from_runnerase(
        self,
        runnerase_class: Union[
            RunnerSplitTrainTest,
            RunnerWeights,
            RunnerScaling,
            RunnerFitResults,
            RunnerSymmetryFunctionValues,
        ],
    ) -> None:
        """Fill `self` with information of the corresponding `runnerase` object.

        Args:
            runnerase_class (runnerase storage class): The runnerase class whose
                information will be wrapped.
        """
        for prop in self.runnerase_properties:
            self.__dict__[prop] = runnerase_class.__dict__[prop]

    def to_runnerase(
        self,
    ) -> Union[
        RunnerSplitTrainTest,
        RunnerWeights,
        RunnerScaling,
        RunnerFitResults,
        RunnerSymmetryFunctionValues,
    ]:
        """Create the corresponding `runnerase` object from `self`.

        Returns:
            runnerase_class (runnerase storage class): The runnerase class whose
                information was wrapped.
        """
        runnerase_class = self.baseclass()

        for prop in self.runnerase_properties:
            runnerase_class.__dict__[prop] = self.__dict__[prop]

        return runnerase_class

    def _to_hdf(self, hdf: ProjectHDFio) -> None:
        """Write `self` to HDF5 storage.

        Args:
            hdf (ProjectHDFio): The HDF file where `self` will be stored.
        """
        for prop in self.runnerase_properties:
            hdf[f"{prop}"] = self.__dict__[prop]

    def _from_hdf(self, hdf: ProjectHDFio, version: Optional[str] = None) -> Union[
        RunnerSplitTrainTest,
        RunnerWeights,
        RunnerScaling,
        RunnerFitResults,
        RunnerSymmetryFunctionValues,
    ]:
        """Read `self` from HDF5 storage.

        Args:
            hdf (ProjectHDFio): The HDF file where `self` will be stored.
            version (str): The HDF version of the storage file.
        """
        if version != self.__hdf_version__:
            raise RuntimeError(
                "Invalid HDF5 version found while reading " + self.__class__.__name__
            )

        # Open HDF file at the right group with a context manager.
        for node in hdf.list_nodes():
            for prop in self.runnerase_properties:
                if prop in node:
                    self.__dict__[prop] = hdf[node]

        return self

    def _get_hdf_group_name(self):
        """Get the name of the group where this object is stored in HDF."""
        return self.__class__.__name__


class HDFSymmetryFunctionSet(SymmetryFunctionSet, RunneraseHDFMixin):
    """Extend runnerase SymmetryFunctionSet with HDF5 compatibility."""

    __hdf_version__ = "0.3.0"

    @property
    def runnerase_properties(self):
        """Show class properties stored in `SymmetryFunctionSet` objects."""
        return ["_sets", "_symmetryfunctions", "min_distances"]

    @property
    def baseclass(self):
        """Define the base class which is wrapped by this HDF class."""
        return SymmetryFunctionSet

    def _to_hdf(self, hdf: ProjectHDFio) -> None:
        """Write `self` to HDF5 storage.

        `runnerase`s SymmetryFunctionSet has the convenient property that
        all symmetry functions can be written to and read from a list
        representation. Therefore, they are also stored as lists in HDF format.

        Args:
            hdf (ProjectHDFio): The HDF file where `self` will be stored.
        """
        for idx, sfset in enumerate(self.sets):
            hdfset = HDFSymmetryFunctionSet()
            hdfset.from_runnerase(sfset)
            hdfset.to_hdf(hdf=hdf, group_name=f"set__index_{idx}")

        symmetryfunctions = [sf.to_list() for sf in self.symmetryfunctions]
        hdf["symmetryfunctions__index_0"] = symmetryfunctions

    def _from_hdf(
        self, hdf: ProjectHDFio, version: Optional[str] = None
    ) -> "HDFSplitTrainTest":
        """Read `self` from HDF5 storage.

        Args:
            hdf (ProjectHDFio): The HDF file where `self` will be stored.
            version (str): The HDF version of the storage file.
        """
        if version != self.__hdf_version__:
            raise RuntimeError(
                "Invalid HDF5 version found while reading " + self.__class__.__name__
            )

        # Reload symmetry function sets.
        for group in hdf.list_groups():
            if group.startswith("set"):
                new_set = hdf.__getitem__(group).to_object()
                self.append(new_set)

        for node in hdf.list_nodes():
            # Reload symmetry functions.
            if node.startswith("symmetryfunctions"):
                sflist = hdf.__getitem__(node)
                self.from_list(sflist)

        return self


class HDFSplitTrainTest(RunnerSplitTrainTest, RunneraseHDFMixin):
    """Mix HDF5 compatibility into RunnerSplitTrainTest`."""

    @property
    def runnerase_properties(self):
        """Show class properties stored in `RunnerSplitTrainTest` objects."""
        return ["train", "test"]

    @property
    def baseclass(self):
        """Define the base class which is wrapped by this HDF class."""
        return RunnerSplitTrainTest


class HDFFitResults(RunnerFitResults, RunneraseHDFMixin):
    """Mix HDF5 compatibility into RunnerFitResults`."""

    @property
    def runnerase_properties(self):
        """Show class properties stored in `RunnerFitResults` objects."""
        return [
            "epochs",
            "rmse_energy",
            "rmse_forces",
            "rmse_charge",
            "opt_rmse_epoch",
            "units",
        ]

    @property
    def baseclass(self):
        """Define the base class which is wrapped by this HDF class."""
        return RunnerFitResults


class HDFWeights(RunnerWeights, RunneraseHDFMixin):
    """Mix HDF5 compatibility into RunnerWeights`."""

    @property
    def runnerase_properties(self):
        """Show class properties stored in `RunnerWeights` objects."""
        return ["data"]

    @property
    def baseclass(self):
        """Define the base class which is wrapped by this HDF class."""
        return RunnerWeights


class HDFScaling(RunnerScaling, RunneraseHDFMixin):
    """Mix HDF5 compatibility into RunnerScaling`."""

    @property
    def runnerase_properties(self):
        """Show class properties stored in `RunnerScaling` objects."""
        return ["data", "target_min", "target_max"]

    @property
    def baseclass(self):
        """Define the base class which is wrapped by this HDF class."""
        return RunnerScaling
