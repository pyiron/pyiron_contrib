# encoding: utf-8
# Copyright (c) Georg-August-Universität Göttingen - Behler Group
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""Utility functions for use with the pyiron Hamiltonian of RuNNer.

.. _RuNNer online documentation:
   https://theochem.gitlab.io/runner
"""

from typing import List

import numpy as np

from ase.atoms import Atoms

from pyiron_atomistics.atomistics.structure.atoms import pyiron_to_ase

from runnerase.singlepoint import RunnerSinglePointCalculator

from ..atomistics.job.trainingcontainer import TrainingContainer


def container_to_ase(container: TrainingContainer) -> List[Atoms]:
    """Convert a `TrainingContainer` into a list of ASE Atoms objects.

    Args:
        container (TrainingContainer): The training data to be converted.

    Returns:
        structures (List[Atoms]): A list of ASE Atoms objects.
    """
    structure_lst = []

    arrays = container.to_dict()
    arraynames = arrays.keys()

    # Iterate over the structures by zipping the dictionary values.
    for properties in zip(*arrays.values()):
        zipped = dict(zip(arraynames, properties))

        # Retrieve atomic positions, cell vectors, etc.
        atoms = pyiron_to_ase(zipped["structure"])

        # Attach charges to the Atoms object.
        if "charges" in zipped:
            atoms.set_initial_charges(zipped["charges"])

        # Store all properties that will be saved on the calculator below.
        calc_properties = {"energy": None, "forces": None, "totalcharge": None}
        for prop in calc_properties:
            if prop in zipped:
                calc_properties[prop] = zipped[prop]

        # Overwrite the totalcharge if the property was not present.
        if calc_properties["totalcharge"] is None:
            totalcharge = np.sum(atoms.get_initial_charges())
            calc_properties["totalcharge"] = totalcharge

        # Storage energies, forces, and totalcharge on a calculator object.
        atoms.calc = RunnerSinglePointCalculator(
            atoms=atoms,
            energy=calc_properties["energy"],
            forces=calc_properties["forces"],
            totalcharge=calc_properties["totalcharge"],
        )

        structure_lst.append(atoms)

    return structure_lst


def ase_to_container(structures: List[Atoms], container: TrainingContainer) -> None:
    """Append `structures` to `TrainingContainer`.

    Args:
        structures (List[Atoms]): A list of ASE Atoms objects to be stored on
            the given `container`.
        container (TrainingContainer): The container to which the data will be
            appended.
    """
    for structure in structures:
        properties = {}

        # If the structure has a calculator attached to it, get energies,
        # forces, etc.
        if structure.calc is not None:
            properties["energy"] = structure.get_potential_energy()
            properties["forces"] = structure.get_forces()
            properties["charges"] = structure.get_initial_charges()

            if isinstance(structure.calc, RunnerSinglePointCalculator):
                properties["totalcharge"] = structure.calc.get_property("totalcharge")

            else:
                properties["totalcharge"] = np.sum(properties["charges"])

        # StructureStorage needs a `spins` property, but not all ASE Atoms
        # objects have that.
        if not hasattr(structure, "spins"):
            structure.spins = None

        container.include_structure(structure, **properties)


def pad(array: np.ndarray, desired_length: int) -> np.ndarray:
    """Pad `array` with `np.NaN` rows up to `desired_length`.

    This routine pads an array of symmetry function values with np.NaN in those
    places where the index (first column of sfvalue arrays) is missing.

    Args:
        array (np.ndarray): The array to be padded. The first column must
            contain a continuous index.
        desired_length (int): The final desired length of the array.

    Returns:
        array_padded (np.ndarray): The padded array.
    """
    # Create a sequence of missing indices.
    all_indices = np.arange(0, desired_length, 1)
    contained_indices = array[:, 0].astype(int)
    missing_indices = np.delete(all_indices, contained_indices)

    # Create an np.NaN-filled array for the padded sfvalues data.
    array_padded = np.empty(shape=(desired_length, array.shape[1]))
    array_padded[:] = np.NaN

    # Insert first the data for this atom, second the np.NaN values.
    array_padded[: len(contained_indices), :] = array
    array_padded[len(contained_indices) :, 0] = missing_indices

    return array_padded


def unpad(array: np.ndarray):
    """Remove all rows containing NaN values from an ndarray."""
    return array[~np.isnan(array).any(axis=1), :]
