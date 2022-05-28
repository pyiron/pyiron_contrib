# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Store structures together with energies and forces for potential fitting applications.

Basic usage:

>>> pr = Project("training")
>>> container = pr.create.job.TrainingContainer("small_structures")

Let's make a structure and invent some forces

>>> structure = pr.create.structure.ase_bulk("Fe")
>>> forces = numpy.array([-1, 1, -1])
>>> container.include_structure(structure, energy=-1.234, forces=forces, name="Fe_bcc")

If you have a lot of precomputed structures you may also add them in bulk from a pandas DataFrame

>>> df = pandas.DataFrame({ "name": "Fe_bcc", "atoms": structure, "energy": -1.234, "forces": forces })
>>> container.include_dataset(df)

You can retrieve the full database with :method:`~.TrainingContainer.to_pandas()` like this

>>> container.to_pandas()
name    atoms   energy  forces  number_of_atoms
Fe_bcc  ...
"""

from typing import Callable, Optional
from warnings import catch_warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyiron_contrib.atomistics.atomistics.job.structurestorage import StructureStorage
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure
from pyiron_atomistics.atomistics.structure.neighbors import NeighborsTrajectory
from pyiron_base import GenericJob, DataContainer, deprecate


class TrainingContainer(GenericJob, HasStructure):
    """
    Stores ASE structures with energies and forces.
    """

    def __init__(self, project, job_name):
        super().__init__(project=project, job_name=job_name)
        self.__name__ = "TrainingContainer"
        self.__hdf_version__ = "0.3.0"
        self._container = TrainingStorage()

        self.input = DataContainer(
            {"save_neighbors": True, "num_neighbors": 12}, table_name="parameters"
        )

    def include_job(self, job, iteration_step=-1):
        """
        Add structure, energy and forces from job.

        Args:
            job (:class:`.AtomisticGenericJob`): job to take structure from
            iteration_step (int, optional): if job has multiple steps, this selects which to add
        """
        self._container.include_job(job, iteration_step)

    @deprecate("Use add_structure instead")
    def include_structure(self, structure, energy, forces=None, stress=None, name=None):
        """
        Add new structure to structure list and save energy and forces with it.

        For consistency with the rest of pyiron, energy should be in units of eV and forces in eV/A, but no conversion
        is performed.

        Args:
            structure_or_job (:class:`~.Atoms`): structure to add
            energy (float): energy of the whole structure
            forces (Nx3 array of float, optional): per atom forces, where N is the number of atoms in the structure
            stress (6 array of float, optional): per structure stresses in voigt notation
            name (str, optional): name describing the structure
        """
        self._container.include_structure(structure, energy, forces, stress, name)

    def add_structure(
        self, structure, energy, forces=None, stress=None, name=None, **arrays
    ):
        """
        Add new structure to structure list and save energy and forces with it.

        For consistency with the rest of pyiron, energy should be in units of eV and forces in eV/A, but no conversion
        is performed.

        Args:
            structure_or_job (:class:`~.Atoms`): structure to add
            energy (float): energy of the whole structure
            forces (Nx3 array of float, optional): per atom forces, where N is the number of atoms in the structure
            stress (6 array of float, optional): per structure stresses in voigt notation
            name (str, optional): name describing the structure
        """
        self._container.add_structure(
            structure, energy, name=name, forces=forces, stress=stress, **arrays
        )

    def include_dataset(self, dataset):
        """
        Add a pandas DataFrame to the saved structures.

        The dataframe should have the following columns:
            - name: human readable name of the structure
            - atoms(:class:`ase.Atoms`): the atomic structure
            - energy(float): energy of the whole structure
            - forces (Nx3 array of float): per atom forces, where N is the number of atoms in the structure
            - stress (6 array of float): per structure stress in voigt notation
        """
        self._container.include_dataset(dataset)

    def _get_structure(self, frame=-1, wrap_atoms=True):
        return self._container.get_structure(frame=frame, wrap_atoms=wrap_atoms)

    def _number_of_structures(self):
        return self._container.number_of_structures

    def get_neighbors(self, num_neighbors=None):
        """
        Calculate and add neighbor information in each structure.

        If input.save_neighbors is True the data is automatically added to the internal storage and will be saved
        together with the normal structure data.

        Args:
            num_neighbors (int, optional): Number of neighbors to collect, if not given use value from input

        Returns:
            NeighborsTrajectory: neighbor information
        """
        if num_neighbors is None:
            num_neighbors = self.input.num_neighbors
        n = NeighborsTrajectory(
            has_structure=self,
            store=self._container if self.input.save_neighbors else None,
            num_neighbors=num_neighbors,
        )
        n.compute_neighbors()
        return n

    def get_elements(self):
        """
        Return a list of chemical elements in the training set.

        Returns:
            :class:`list`: list of unique elements in the training set as strings of their standard abbreviations
        """
        return self._container.get_elements()

    def to_pandas(self):
        """
        Export list of structure to pandas table for external fitting codes.

        The table contains the following columns:
            - 'name': human-readable name of the structure
            - 'ase_atoms': the structure as a :class:`.Atoms` object
            - 'energy': the energy of the full structure
            - 'forces': the per atom forces as a :class:`numpy.ndarray`, shape Nx3
            - 'stress': the per structure stress as a :class:`numpy.ndarray`, shape 6
            - 'number_of_atoms': the number of atoms in the structure, N

        Returns:
            :class:`pandas.DataFrame`: collected structures
        """
        return self._container.to_pandas()

    def to_list(self, filter_function=None):
        """
        Returns the data as lists of pyiron structures, energies, forces, and the number of atoms

        Args:
            filter_function (function): Function applied to the dataset (which is a pandas DataFrame) to filter it

        Returns:
            tuple: list of structures, energies, forces, and the number of atoms
        """
        return self._container.to_list(filter_function)

    def write_input(self):
        pass

    def collect_output(self):
        pass

    def run_static(self):
        if self.input.save_neighbors:
            self.get_neighbors()
        self.status.finished = True

    def run_if_interactive(self):
        if self.input.save_neighbors:
            self.get_neighbors()
        self.to_hdf()
        self.status.finished = True

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(self.project_hdf5)
        self._container.to_hdf(self.project_hdf5, "structures")

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        hdf_version = self.project_hdf5.get("HDF_VERSION", "0.1.0")
        if hdf_version == "0.1.0":
            table = pd.read_hdf(
                self.project_hdf5.file_name, self.name + "/output/structure_table"
            )
            self.include_dataset(table)
        else:
            self._container = TrainingStorage()
            self._container.from_hdf(self.project_hdf5, "structures")
            if hdf_version == "0.3.0":
                self.input.from_hdf(self.project_hdf5, "parameters")

    def sample(
        self, name: str, selector: Callable[[StructureStorage, int], bool]
    ) -> "TrainingContainer":
        """
        Create a new TrainingContainer with structures filtered by selector.

        `self` must have status `finished`.  `selector` is passed the underlying :class:`StructureStorage` of this
        container and the index of the structure and return a boolean whether to include the structure in the new
        container or not.  The new container is saved and run.

        Args:
            name (str): name of the new TrainingContainer
            selector (Callable[[StructureStorage, int], bool]): callable that selects structure to include

        Returns:
            :class:`.TrainingContainer`: new container with selected structures

        Raises:
            ValueError: if a job with the given `name` already exists.
        """
        cont = self.project.create.job.TrainingContainer(name)
        if not cont.status.initialized:
            raise ValueError(f"Job '{name}' already exists with status: {cont.status}!")
        cont._container = self._container.sample(selector)
        cont.run()
        return cont

    @property
    def plot(self):
        """
        :class:`.TrainingPlots`: plotting interface
        """
        return TrainingPlots(self._container)

    def iter(self, *arrays, wrap_atoms=True):
        """
        Iterate over all structures in this object and all arrays that are defined

        Args:
            wrap_atoms (bool): True if the atoms are to be wrapped back into the unit cell; passed to
                               :meth:`.get_structure()`
            *arrays (str): name of arrays that should be iterated over

        Yields:
            :class:`pyiron_atomistics.atomistitcs.structure.atoms.Atoms`, arrays: every structure attached to the object and queried arrays
        """
        yield from self._container.iter(*arrays, wrap_atoms=wrap_atoms)


class TrainingPlots:
    """
    Simple interface to plot various properties of the structures inside the given :class:`.TrainingContainer`.
    """

    __slots__ = "_train"

    def __init__(self, train):
        self._train = train

    def _calc_spacegroups(self, symprec=1e-3):
        """
        Calculate space groups of all structures.

        Args:
            symprec (float): symmetry precision given to spglib

        Returns:
            DataFrame: contains columns 'crystal_system' (str) and 'space_group' (int) for each structure
        """

        def get_crystal_system(num):
            if num in range(1, 3):
                return "triclinic"
            elif num in range(3, 16):
                return "monoclinic"
            elif num in range(16, 75):
                return "orthorombic"
            elif num in range(75, 143):
                return "trigonal"
            elif num in range(143, 168):
                return "tetragonal"
            elif num in range(168, 195):
                return "hexagonal"
            elif num in range(195, 230):
                return "cubic"

        def extract(s):
            spg = s.get_symmetry(symprec=symprec).spacegroup["Number"]
            return {"space_group": spg, "crystal_system": get_crystal_system(spg)}

        return pd.DataFrame(map(extract, self._train.iter_structures()))

    def cell(self, angle_in_degrees=True):
        """
        Plot histograms of cell parameters.

        Plotted are atomic volume, density, cell vector lengths and cell vector angles in separate subplots all on a
        log-scale.

        Args:
            angle_in_degrees (bool): whether unit for angles is degree or radians

        Returns:
            `DataFrame`: contains the plotted information in the columns:
                            - a: length of first vector
                            - b: length of second vector
                            - c: length of third vector
                            - alpha: angle between first and second vector
                            - beta: angle between second and third vector
                            - gamma: angle between third and first vector
                            - V: volume of the cell
                            - N: number of atoms in the cell
        """
        N = self._train.get_array("length")
        C = self._train.get_array("cell")

        def get_angle(cell, idx=0):
            return np.arccos(
                np.dot(cell[idx], cell[(idx + 1) % 3])
                / np.linalg.norm(cell[idx])
                / np.linalg.norm(cell[(idx + 1) % 3])
            )

        def extract(n, c):
            return {
                "a": np.linalg.norm(c[0]),
                "b": np.linalg.norm(c[1]),
                "c": np.linalg.norm(c[2]),
                "alpha": get_angle(c, 0),
                "beta": get_angle(c, 1),
                "gamma": get_angle(c, 2),
            }

        df = pd.DataFrame([extract(n, c) for n, c in zip(N, C)])
        df["V"] = np.linalg.det(C)
        df["N"] = N
        if angle_in_degrees:
            df["alpha"] = np.rad2deg(df["alpha"])
            df["beta"] = np.rad2deg(df["beta"])
            df["gamma"] = np.rad2deg(df["gamma"])

        plt.subplot(1, 4, 1)
        plt.title("Atomic Volume")
        plt.hist(df.V / df.N, bins=20, log=True)
        plt.xlabel(r"$V$ [$\AA^3$]")

        plt.subplot(1, 4, 2)
        plt.title("Density")
        plt.hist(df.N / df.V, bins=20, log=True)
        plt.xlabel(r"$\rho$ [$\AA^{-3}$]")

        plt.subplot(1, 4, 3)
        plt.title("Lattice Vector Lengths")
        plt.hist([df.a, df.b, df.c], log=True)
        plt.xlabel(r"$a,b,c$ [$\AA$]")

        plt.subplot(1, 4, 4)
        plt.title("Lattice Vector Angles")
        plt.hist([df.alpha, df.beta, df.gamma], log=True)
        if angle_in_degrees:
            label = r"$\alpha,\beta,\gamma$ [°]"
        else:
            label = r"$\alpha,\beta,\gamma$ [rad]"
        plt.xlabel(label)

        return df

    def spacegroups(self, symprec=1e-3):
        """
        Plot histograms of space groups and crystal systems.

        Spacegroups and crystal systems are plotted in separate subplots.

        Args:
            symprec (float): precision of the symmetry search (passed to spglib)

        Returns:
            DataFrame: contains two columns "space_group", "crystal_system"
                       for each structure in `train`
        """

        df = self._calc_spacegroups(symprec=symprec)
        plt.subplot(1, 2, 1)
        plt.hist(df.space_group, bins=230)
        plt.xlabel("Space Group")

        plt.subplot(1, 2, 2)
        l, h = np.unique(df.crystal_system, return_counts=True)
        sort_key = {
            "triclinic": 1,
            "monoclinic": 3,
            "orthorombic": 16,
            "trigonal": 75,
            "tetragonal": 143,
            "hexagonal": 168,
            "cubic": 195,
        }
        I = np.argsort([sort_key[ll] for ll in l])
        plt.bar(l[I], h[I])
        plt.xlabel("Crystal System")
        plt.xticks(rotation=35)
        return df

    def energy_volume(self, crystal_systems=False):
        """
        Plot volume vs. energy.

        Volume and energy are normalized per atom before plotting.

        Args:
            crystal_systems (bool): if True, plot & label structures of different crystal systems separately.

        Returns:
            DataFrame: contains atomic energy and volumes in the columns 'E' and 'V'; if `crystal_systems` is given,
                       also contain space groups and crystal systems of each structure
        """

        N = self._train.get_array("length")
        E = self._train.get_array("energy") / N
        C = self._train.get_array("cell")
        V = np.linalg.det(C) / N

        df = pd.DataFrame({"V": V, "E": E})

        if crystal_systems:
            spg = self._calc_spacegroups()
            df = df.join(spg)
            for cs, dd in df.groupby("crystal_system"):
                plt.scatter(dd.V, dd.E, label=cs)
            plt.legend()
        else:
            plt.scatter(df.V, df.E)
        plt.xlabel(r"Atomic Volume [$\AA^3$]")
        plt.ylabel(r"Atomic Energy [eV]")

        return df

    def coordination(self, num_shells=4, log=True):
        """
        Plot histogram of coordination in neighbor shells.

        Computes one histogram of the number of neighbors in each neighbor shell up to `num_shells` and then plots them
        together.

        Args:
            num_shells (int): maximum shell to plot
            log (float): plot histogram values on a log scale
        """
        if not self._train.has_array("shells"):
            raise ValueError(
                "TrainingContainer contains no neighbor information, call TrainingContainer.get_neighbors first!"
            )
        shells = self._train.get_array("shells")
        shell_index = (
            shells[np.newaxis, :, :]
            == np.arange(1, num_shells + 1)[:, np.newaxis, np.newaxis]
        )
        neigh_count = shell_index.sum(axis=-1)
        ticks = np.arange(neigh_count.min(), neigh_count.max() + 1)
        plt.hist(
            neigh_count.T,
            bins=ticks - 0.5,
            log=True,
            label=[f"{i}." for i in range(1, num_shells + 1)],
        )
        plt.xticks(ticks)
        plt.xlabel("Number of Neighbors")
        plt.legend(title="Shell")
        plt.title("Neighbor Coordination in Shells")

    def shell_distances(self, num_shells=4):
        """
        Plot a violin plot of the neighbor distances in shells up to `num_shells`.

        Args:
            num_shells (int): maximum shell to plot
        """
        if not self._train.has_array("shells") or not self._train.has_array(
            "distances"
        ):
            raise ValueError(
                "TrainingContainer contains no neighbor information, call TrainingContainer.get_neighbors first!"
            )
        dists = self._train.get_array("distances")
        R = dists.flatten()
        shells = self._train.get_array("shells")
        S = shells.ravel()
        d = pd.DataFrame(
            {"distance": R[S < num_shells + 1], "shells": S[S < num_shells + 1]}
        )
        sns.violinplot(y=d.shells, x=d.distance, scale="width", orient="h")
        plt.xlabel(r"Distance [$\AA$]")
        plt.ylabel("Shell")

    def forces(self, axis: Optional[int] = None):
        """
        Plot a histogram of all forces.

        Args:
            axis (int, optional): plot only forces along this axis, if not given plot all forces
        """
        f = self._train.get_array("forces")
        if axis is not None:
            f = f[:, axis]
        else:
            f = f.ravel()
        plt.hist(f, bins=20)
        plt.xlabel(r"Force [eV/$\mathrm{\AA}$]")


class TrainingStorage(StructureStorage):
    def __init__(self):
        super().__init__()
        self.add_array("energy", dtype=np.float64, per="chunk", fill=np.nan)
        self.add_array(
            "forces", shape=(3,), dtype=np.float64, per="element", fill=np.nan
        )
        # save stress in voigt notation
        self.add_array("stress", shape=(6,), dtype=np.float64, per="chunk", fill=np.nan)
        self._table_cache = None
        self.to_pandas()

    def to_pandas(self):
        """
        Export list of structure to pandas table for external fitting codes.

        The table contains the following columns:
            - 'name': human-readable name of the structure
            - 'ase_atoms': the structure as a :class:`.Atoms` object
            - 'energy': the energy of the full structure
            - 'forces': the per atom forces as a :class:`numpy.ndarray`, shape Nx3
            - 'stress': the per structure stress as a :class:`numpy.ndarray`, shape 6
            - 'number_of_atoms': the number of atoms in the structure, N

        Returns:
            :class:`pandas.DataFrame`: collected structures
        """
        if self._table_cache is None or len(self._table_cache) != len(self):
            self._table_cache = pd.DataFrame(
                {
                    "name": [self.get_array("identifier", i) for i in range(len(self))],
                    "atoms": [self.get_structure(i) for i in range(len(self))],
                    "energy": [self.get_array("energy", i) for i in range(len(self))],
                    "forces": [self.get_array("forces", i) for i in range(len(self))],
                    "stress": [self.get_array("stress", i) for i in range(len(self))],
                }
            )
            self._table_cache["number_of_atoms"] = [
                len(s) for s in self._table_cache.atoms
            ]
        return self._table_cache

    def include_job(self, job, iteration_step=-1):
        """
        Add structure, energy and forces from job.

        Args:
            job (:class:`.AtomisticGenericJob`): job to take structure from
            iteration_step (int, optional): if job has multiple steps, this selects which to add
        """
        energy = job.output.energy_pot[iteration_step]
        ff = job.output.forces
        if ff is not None:
            forces = ff[iteration_step]
        else:
            forces = None
        # HACK: VASP work-around, current contents of pressures are meaningless, correct values are in
        # output/generic/stresses
        pp = job["output/generic/stresses"]
        if pp is None:
            pp = job.output.pressures
        if pp is not None:
            stress = pp[iteration_step]
        else:
            stress = None
        if stress is not None:
            stress = np.asarray(stress)
            if stress.shape == (3, 3):
                stress = np.array(
                    [
                        stress[0, 0],
                        stress[1, 1],
                        stress[2, 2],
                        stress[1, 2],
                        stress[0, 2],
                        stress[0, 1],
                    ]
                )
        self.include_structure(
            job.get_structure(iteration_step=iteration_step),
            energy=energy,
            forces=forces,
            stress=stress,
            name=job.name,
        )

    @deprecate("Use add_structure instead")
    def include_structure(self, structure, energy, forces=None, stress=None, name=None):
        """
        Add new structure to structure list and save energy and forces with it.

        For consistency with the rest of pyiron, energy should be in units of eV and forces in eV/A, but no conversion
        is performed.

        Args:
            structure_or_job (:class:`~.Atoms`): structure to add
            energy (float): energy of the whole structure
            forces (Nx3 array of float, optional): per atom forces, where N is the number of atoms in the structure
            stress (6 array of float, optional): per structure stresses in voigt notation
            name (str, optional): name describing the structure
        """
        self.add_structure(
            structure, energy, identifier=name, forces=forces, stress=stress
        )

    def add_structure(
        self, structure, energy, identifier=None, forces=None, stress=None, **arrays
    ):
        data = {"energy": energy}
        if forces is not None:
            data["forces"] = forces
        if stress is not None:
            data["stress"] = stress
        super().add_structure(structure, identifier, **data)

    def include_dataset(self, dataset):
        """
        Add a pandas DataFrame to the saved structures.

        The dataframe should have the following columns:
            - name: human readable name of the structure
            - atoms(:class:`ase.Atoms`): the atomic structure
            - energy(float): energy of the whole structure
            - forces (Nx3 array of float): per atom forces, where N is the number of atoms in the structure
            - stress (6 array of float): per structure stress in voigt notation
        """
        if (
            "name" not in dataset.columns
            or "atoms" not in dataset.columns
            or "energy" not in dataset.columns
        ):
            raise ValueError(
                "At least columns 'name', 'atoms' and 'energy' must be present in dataset!"
            )
        for row in dataset.itertuples(index=False):
            kwargs = {}
            if hasattr(row, "forces"):
                kwargs["forces"] = row.forces
            if hasattr(row, "stress"):
                kwargs["stress"] = row.stress
            self.add_structure(
                row.atoms, energy=row.energy, identifier=row.name, **kwargs
            )

    def to_list(self, filter_function=None):
        """
        Returns the data as lists of pyiron structures, energies, forces, and the number of atoms

        Args:
            filter_function (function): Function applied to the dataset (which is a pandas DataFrame) to filter it

        Returns:
            tuple: list of structures, energies, forces, and the number of atoms
        """
        data_table = self.to_pandas()
        if filter_function is not None:
            data_table = filter_function(data_table)
        structure_list = data_table.atoms.to_list()
        energy_list = data_table.energy.to_list()
        force_list = data_table.forces.to_list()
        num_atoms_list = data_table.number_of_atoms.to_list()
        return structure_list, energy_list, force_list, num_atoms_list

    def iter(self, *arrays, wrap_atoms=True):
        """
        Iterate over all structures in this object and all arrays that are defined

        Args:
            wrap_atoms (bool): True if the atoms are to be wrapped back into the unit cell; passed to
                               :meth:`.get_structure()`
            *arrays (str): name of arrays that should be iterated over

        Yields:
            :class:`pyiron_atomistics.atomistitcs.structure.atoms.Atoms`, arrays: every structure attached to the object and queried arrays
        """
        array_vals = (self.get_array_ragged(a) for a in arrays)
        yield from zip(self.iter_structures(), *array_vals)

    @property
    def plot(self):
        """
        :class:`.TrainingPlots`: plotting interface
        """
        return TrainingPlots(self)
