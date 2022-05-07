# encoding: utf-8
# Copyright (c) Georg-August-Universität Göttingen - Behler Group
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""Utility functions for use with the pyiron Hamiltonian of RuNNer.

Attributes
----------
    RunnerTrainingContainer : TrainingContainer
        Extension of the TrainingContainer class with a function to convert
        to_ase.

Reference
---------
    [RuNNer online documentation](https://theochem.gitlab.io/runner)
"""

from typing import List, Optional

import numpy as np

from ase.atoms import Atoms

from pyiron_atomistics.atomistics.structure.atoms import pyiron_to_ase

from runnerase.singlepoint import RunnerSinglePointCalculator

from ..atomistics.job.trainingcontainer import TrainingContainer


def container_to_ase(container: TrainingContainer) -> List[Atoms]:
    """Convert a `TrainingContainer` into a list of ASE Atoms objects."""
    structure_lst = []

    for row in list(zip(*container.to_list())):

        # Retrieve all properties, i.e. energy, forces, etc.
        structure, energy, forces, totalcharge, charges, _ = row

        # Retrieve atomic positions, cell vectors, etc.
        atoms = pyiron_to_ase(structure)

        # Attach properties to the Atoms object.
        atoms.set_initial_charges(charges)
        atoms.calc = RunnerSinglePointCalculator(
            atoms=atoms,
            energy=energy,
            forces=forces,
            totalcharge=totalcharge
        )
        structure_lst.append(atoms)

    return structure_lst


def ase_to_container(
    structures: List[Atoms],
    container: Optional[TrainingContainer]
) -> None:
    """Add `structures` to `TrainingContainer`."""
    for structure in structures:

        properties = {}

        # If the structure has a calculator attached to it, get energies,
        # forces, etc.
        if structure.calc is not None:
            properties['energy'] = structure.get_potential_energy()
            properties['forces'] = structure.get_forces()
            properties['charges'] = structure.get_initial_charges()

            if isinstance(structure.calc, RunnerSinglePointCalculator):
                properties['totalcharge'] = structure.calc.get_property(
                    'totalcharge'
                )
            else:
                properties['totalcharge'] = np.sum(properties['charges'])

        # StructureStorage needs a `spins` property, but not all ASE Atoms
        # objects have that.
        if not hasattr(structure, 'spins'):
            structure.spins = None

        container.include_structure(structure, **properties)



def pad(array: np.ndarray, desired_length: int) -> np.ndarray:
    """Pad `sfvalues` with `np.NaN` rows up to `num_atoms`."""
    # Create a sequence of missing indices.
    all_indices = np.arange(0, desired_length, 1)
    contained_indices = array[:, 0].astype(int)
    missing_indices = np.delete(all_indices, contained_indices)

    # Create an np.NaN-filled array for the padded sfvalues data.
    array_padded = np.empty(shape=(desired_length, array.shape[1]))
    array_padded[:] = np.NaN

    # Insert first the data for this atom, second the np.NaN values.
    array_padded[:len(contained_indices), :] = array
    array_padded[len(contained_indices):, 0] = missing_indices

    return array_padded


def unpad(array: np.ndarray):
    """Remove all rows containing NaN values from an ndarray."""
    return array[~np.isnan(array).any(axis=1), :]
