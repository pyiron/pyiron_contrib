# coding: utf-8
# Copyright (c) Georg-August-Universität Göttingen - Behler Group
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""Pyiron Hamiltonian for machine-learning with RuNNer.

The RuNNer Neural Network Energy Representation is a framework for the
construction of high-dimensional neural network potentials developed in the
group of Prof. Dr. Jörg Behler at Georg-August-Universität Göttingen.

Attributes:
     RunnerFit : GenericJob, HasStorage, PotentialFit
        Job class for generating and evaluating potential energy surfaces using
        RuNNer.

.. _RuNNer online documentation:
   https://theochem.gitlab.io/runner
"""

from typing import Optional, List
from copy import deepcopy

import numpy as np
import pandas as pd

from ase.data import atomic_numbers
from ase.units import Bohr

from runnerase.io.ase import read_results_mode1, read_results_mode2, read_results_mode3
from runnerase import Runner
from runnerase.defaultoptions import DEFAULT_PARAMETERS

from pyiron import Project
from pyiron_base import ProjectHDFio, FlattenedStorage
from pyiron_base import state, Executable, GenericJob, DataContainer
from pyiron_base import HasStorage

from pyiron_contrib.atomistics.ml.potentialfit import PotentialFit
from pyiron_contrib.atomistics.atomistics.job import TrainingContainer, TrainingStorage

from .utils import container_to_ase
from .storageclasses import (
    HDFSymmetryFunctionSet,
    HDFSymmetryFunctionValues,
    HDFSplitTrainTest,
    HDFFitResults,
    HDFWeights,
    HDFScaling,
)

__author__ = "Alexander Knoll"
__maintainer__ = "Alexander Knoll"
__email__ = "alexander.knoll@chemie.uni-goettingen.de"
__copyright__ = "Copyright 2022, Georg-August-Universität Göttingen - Behler " "Group"
__version__ = "0.1.1"
__status__ = "development"
__date__ = "May 17, 2022"


class RunnerFit(GenericJob, HasStorage, PotentialFit):
    """Generate a potential energy surface using RuNNer.

    The RuNNer Neural Network Energy Representation (RuNNer) is a Fortran code
    for the generation of high-dimensional neural network potentials actively
    developed in the group of Prof. Dr. Jörg Behler at Georg-August-Universität
    Göttingen.

    RuNNer operates in three different modes:

        - Mode 1: Calculation of symmetry function values. Symmetry functions
          are many-body descriptors for the chemical environment of an atom.
        - Mode 2: Fitting of the potential energy surface.
        - Mode 3: Prediction. Use the previously generated high-dimensional
          potential energy surface to predict the energy and force of an
          unknown chemical configuration.

    The different modes generate a lot of output:

        - Mode 1:
            - sfvalues: The values of the symmetry functions for each atom.
            - splittraintest: which structures belong to the training and which
              to the testing set. ASE needs this information to generate the
              relevant input files for RuNNer Mode 2.

        - Mode 2:
            - weights: The neural network weights.
            - scaling: The symmetry function scaling factors.

        - Mode 3:
            - energy
            - forces

    Examples:
        Starting a new calculation (Mode 1):

        ```python
        from pyiron import Project
        from job import RunnerFit
        from runnerase import read_runnerase

        # Create an empty sample project and a new job.
        pr = Project(path='example')
        job = pr.create_job(RunnerFit, 'mode1')

        # Import RuNNer settings from RuNNer input.nn file using ASE's I/O
        # routines.
        options = read_runnerconfig('./')

        # Read input structures from a TrainingContainer that was saved before.
        container = pr['H2O_MD']

        # Attach the information to the job.
        job.add_training_data = container
        job.parameters.update(options)

        # Set the RuNNer Mode to 1.
        job.input.runner_mode = 1

        job.run()
        ```

        Restarting Mode 1 and running Mode 2:

        ```python
        job = Project('example')['mode1'].restart('mode2')
        job.parameters.runner_mode = 2
        job.run()
        ```

        Restarting Mode 2 and running Mode 3:

        ```python
        job = Project('runnertest')['mode2'].restart('mode3')
        job.parameters.runner_mode = 3
        job.run()
        ```
    """

    __name__ = "RuNNerFit"

    # These properties are needed by RuNNer as input data (depending on the
    # chosen RuNNer mode).
    _input_properties = ["scaling", "weights", "sfvalues", "splittraintest"]

    # Define a default executable.
    _executable = Executable(
        codename="runner",
        module="runner",
        path_binary_codes=state.settings.resource_paths,
    )

    def __init__(self, project: Project, job_name: str) -> None:
        """Initialize the class.

        Args:
            project (Project): The project container where the job is created.
            job_name (str): The label of the job (used for all directories).
        """
        # Initialize first the job, then the job storage.
        GenericJob.__init__(self, project=project, job_name=job_name)
        HasStorage.__init__(self)

        # Create groups for storing calculation inputs and outputs.
        self.storage.create_group("input")
        self.storage.create_group("output")

        # Create a group for storing the RuNNer configuration parameters.
        self.storage.input.create_group("parameters")

        self.storage.input.parameters.update(deepcopy(DEFAULT_PARAMETERS))

        # Store training data (structures, energies, ...) in a separate node.
        self.storage.input.training_data = TrainingStorage()

        state.publications.add(self.publication)

    @property
    def publication(self):
        """Define relevant publications."""
        return {
            "runner": [
                {
                    "title": "First Principles Neural Network Potentials for "
                    "Reactive Simulations of Large Molecular and "
                    "Condensed Systems",
                    "journal": "Angewandte Chemie International Edition",
                    "volume": "56",
                    "number": "42",
                    "year": "2017",
                    "issn": "1521-3773",
                    "doi": "10.1002/anie.201703114",
                    "url": "https://doi.org/10.1002/anie.201703114",
                    "author": ["Jörg Behler"],
                },
                {
                    "title": "Constructing high‐dimensional neural network"
                    "potentials: A tutorial review",
                    "journal": "International Journal of Quantum Chemistry",
                    "volume": "115",
                    "number": "16",
                    "year": "2015",
                    "issn": "1097-461X",
                    "doi": "10.1002/qua.24890",
                    "url": "https://doi.org/10.1002/qua.24890",
                    "author": ["Jörg Behler"],
                },
                {
                    "title": "Generalized Neural-Network Representation of "
                    "High-Dimensional Potential-Energy Surfaces",
                    "journal": "Physical Review Letters",
                    "volume": "98",
                    "number": "14",
                    "year": "2007",
                    "issn": "1079-7114",
                    "doi": "10.1103/PhysRevLett.98.146401",
                    "url": "https://doi.org/10.1103/PhysRevLett.98.146401",
                    "author": ["Jörg Behler", "Michelle Parrinello"],
                },
            ]
        }

    @property
    def input(self) -> DataContainer:
        """Show all input properties in storage.

        Returns:
            self.storage.input (DataContainer): The data container with all
                input properties.
        """
        return self.storage.input

    @property
    def output(self) -> DataContainer:
        """Show all calculation output in storage.

        Returns:
            self.storage.output (DataContainer): The data container with all
                output properties.
        """
        return self.storage.output

    @property
    def parameters(self) -> DataContainer:
        """Show the input parameters/settings in storage.

        Returns:
            self.storage.input.parameters (DataContainer): The data container
                with all input parameters for RuNNer.
        """
        return self.storage.input.parameters

    @property
    def scaling(self) -> Optional[HDFScaling]:
        """Show the symmetry function scaling data in storage.

        Returns:
            self.output.scaling (HDFScaling, None): If defined, the symmetry
                function scaling data in storage.
        """
        if "scaling" in self.output:
            return self.output.scaling

        return None

    @scaling.setter
    def scaling(self, scaling: HDFScaling) -> None:
        """Set the symmetry function scaling data in storage.

        Args:
            scaling (HDFScaling): RuNNer symmetry function scaling data wrapped
                in a HDFScaling storage container.
        """
        self.output.scaling = scaling

    @property
    def weights(self) -> Optional[HDFWeights]:
        """Show the atomic neural network weights data in storage.

        Returns:
            self.output.weights (HDFWeights, None): If defined, the weights in
                storage.
        """
        if "weights" in self.output:
            return self.output.weights

        return None

    @weights.setter
    def weights(self, weights: HDFWeights) -> None:
        """Set the weights of the atomic neural networks in storage.

        Args:
            weights (HDFWeights): Atomic neural network weights wrapped in a
                HDWeights storage container.
        """
        self.output.weights = weights

    @property
    def sfvalues(self) -> Optional[HDFSymmetryFunctionValues]:
        """Show the symmetry function value data in storage.

        Returns:
            self.output.sfvalues (HDFSymmetryFunctionValues, None): If defined,
                the symmetry function values in storage.
        """
        if "sfvalues" in self.output:
            return self.output.sfvalues

        return None

    @sfvalues.setter
    def sfvalues(self, sfvalues: HDFSymmetryFunctionValues) -> None:
        """Set the symmetry function value data in storage.

        Args:
            sfvalues (HDFSymmetryFunctionValues): Symmetry function values
                wrapped in a HDF storage container.
        """
        self.output.sfvalues = sfvalues

    @property
    def splittraintest(self) -> Optional[HDFSplitTrainTest]:
        """Show the split between training and testing data in storage.

        Returns:
            self.output.splittraintest (HDFSplitTrainTest, None): If defined,
                the splitting data in storage.
        """
        if "splittraintest" in self.output:
            return self.output.splittraintest

        return None

    @splittraintest.setter
    def splittraintest(self, splittraintest: HDFSplitTrainTest) -> None:
        """Set the split between training and testing data in storage.

        Args:
            splittraintest (HDFSplitTrainTest): Split between training and
                testing data wrapped in a HDF storage container.
        """
        self.output.splittraintest = splittraintest

    def _add_training_data(self, container: TrainingContainer) -> None:
        """Add a set of training data to storage.

        Args:
            container (TrainingContainer): The training data that will be added
                to `self`.
        """
        # Get a dictionary of all property arrays saved in this container.
        arrays = container.to_dict()
        arraynames = arrays.keys()

        # Iterate over the structures by zipping, i.e. transposing the
        # dictionary values.
        for properties in zip(*arrays.values()):
            zipped = dict(zip(arraynames, properties))
            self.storage.input.training_data.add_structure(**zipped)

    def _get_training_data(self) -> TrainingStorage:
        """Show the stored training data.

        Returns:
            self.storage.input.training_data (TrainingContainer): The stored
                training data.
        """
        return self.storage.input.training_data

    def _get_predicted_data(self) -> FlattenedStorage:
        """Show the predicted data after a successful fit.

        Returns:
            predicted_data (TrainingContainer): The predicted data in storage.
                At the moment, pyiron can interpret the energies and forces
                predicted by RuNNer.
        """
        # Energies and forces will only be available after RuNNer Mode 3.
        if "energy" not in self.output:
            raise RuntimeError(
                "You have to run RuNNer prediction mode "
                + "(Mode 3) before you can access predictions."
            )

        # Get a list of structures.
        structures = list(self.training_data.iter_structures())

        # Get the values of all properties RuNNer can predict for a structure.
        pred_properties = {
            "energy": np.full((len(structures),), np.nan),
            "forces": np.full((3, len(structures)), np.nan),
        }
        for prop in pred_properties:
            if prop in self.output:
                pred_properties[prop] = self.output[prop]

        predicted_data = FlattenedStorage()

        for structure, energy, force in zip(
            structures, pred_properties["energy"], pred_properties["forces"]
        ):
            predicted_data.add_chunk(len(structure), energy=energy, forces=force)

        return predicted_data

    def get_lammps_potential(
        self, elements: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Return a pandas dataframe with information for setting up LAMMPS.

        The nnp pair_style for LAMMPS is provided by the external package n2p2,
        that is maintained by Andreas Singgraber. Please take a look at their
        [documentation](https://compphysvienna.github.io/n2p2/interfaces/pair_\
        nnp.html)
        to understand more about the configuration options.

        Args:
            elements (List[str], optional): A list of elements for which the
                potential will be returned.

        Returns:
            df (pd.DataFrame): A dataframe containing all the information
                required to set up a LAMMPS job with RuNNer.
        """
        if not self.status.finished:
            raise RuntimeError(
                "LAMMPS potential can only be generated after a " + "successful fit."
            )

        if "weights" not in self.output or "scaling" not in self.output:
            raise RuntimeError("This potential has not been trained yet.")

        # Get all elements in the training dataset.
        if elements is None:
            elements = self.training_data.get_elements()

        # Create a list of all files needed by the potential.
        files = [
            f"{self.working_directory}/input.nn",
            f"{self.working_directory}/scaling.data",
        ]

        # Add the weight files.
        for elem in elements:
            atomic_number = atomic_numbers[elem]
            filename = f"weights.{atomic_number:03}.data"
            files.append(f"{self.working_directory}/{filename}")

        # Save the mapping of elements between LAMMPS and n2p2.
        emap = " ".join(el for el in elements)

        # Get the cutoff radius of the symmetry functions.
        cutoffs = self.parameters.symfunction_short.cutoffs
        cutoff = cutoffs[0]

        if len(cutoffs) > 1:
            raise RuntimeError(
                "LAMMPS potential can only be generated for a "
                + "uniform cutoff radius."
            )

        return pd.DataFrame(
            {
                "Name": [f"RuNNer-{''.join(elements)}"],
                "Filename": [files],
                "Model": ["RuNNer"],
                "Species": [elements],
                "Config": [
                    [
                        f'pair_style hdnnp {cutoff * Bohr} dir "./" '
                        + "showew yes showewsum 0 resetew no maxew 100 "
                        + "cflength 1.8897261328 cfenergy 0.0367493254\n",
                        f"pair_coeff * * {emap}\n",
                    ]
                ],
            }
        )

    def write_input(self) -> None:
        """Write the relevant job input files.

        This routine writes the input files for the job using the ASE Runner
        calculator.
        """
        input_properties = {
            "sfvalues": None,
            "splittraintest": None,
            "weights": None,
            "scaling": None,
        }

        for prop in input_properties:
            if prop in self.output and self.output[prop] is not None:
                input_properties[prop] = self.output[prop].to_runnerase()

        # Create an ASE Runner calculator object.
        # Pay attention to the different name: `structures` --> `dataset`.
        calc = Runner(
            label="pyiron",
            dataset=container_to_ase(self.training_data),
            scaling=input_properties["scaling"],
            weights=input_properties["weights"],
            sfvalues=input_properties["sfvalues"],
            splittraintest=input_properties["splittraintest"],
            **self.parameters.to_builtin(),
        )

        # Set the correct elements of the system.
        calc.set_elements()

        # If no seed was specified yet, choose a random value.
        if "random_seed" not in calc.parameters:
            calc.set(random_seed=np.random.randint(1, 1000))

        # If no dataset was attached to the calculator, the single structure
        # stored as the atoms property will be written instead.
        atoms = calc.dataset or calc.atoms

        # Set the correct calculation directory and file prefix.
        calc.directory = self.working_directory
        calc.prefix = f"mode{self.parameters.runner_mode}"

        # Set the 'flags' which ASE uses to see which input files need to
        # be written.
        targets = {1: "sfvalues", 2: "fit", 3: "energy"}

        calc.write_input(
            atoms, targets[self.parameters.runner_mode], system_changes=None
        )

    def _executable_activate(self, enforce: bool = False) -> None:
        """Link to the RuNNer executable."""
        if self._executable is None or enforce:
            self._executable = Executable(
                codename="runner",
                module="runner",
                path_binary_codes=state.settings.resource_paths,
            )

    def collect_output(self) -> None:
        """Read and store the job results."""
        # Compose job label (needed by the ASE I/O routines) and store dir.
        label = f"{self.working_directory}/mode{self.parameters.runner_mode}"
        directory = self.working_directory

        # If successful, RuNNer Mode 1 returns symmetry function values for
        # each structure and the information, which structure belongs to the
        # training and which to the testing set.
        if self.parameters.runner_mode == 1:
            results = read_results_mode1(label, directory)

            # Transform sfvalues into the pyiron class for HDF5 storage and
            # store it in the output dictionary.
            sfvalues = HDFSymmetryFunctionValues()
            sfvalues.from_runnerase(results["sfvalues"])
            self.output.sfvalues = sfvalues

            # Transform split data between training and testing set into the
            # pyiron class for HDF5 storage and store it in the output
            # dictionary.
            splittraintest = HDFSplitTrainTest()
            splittraintest.from_runnerase(results["splittraintest"])
            self.output.splittraintest = splittraintest

        # If successful, RuNNer Mode 2 returns the weights of the atomic neural
        # networks, the symmetry function scaling data, and the results of the
        # fitting process.
        elif self.parameters.runner_mode == 2:
            results = read_results_mode2(label, directory)

            fitresults = HDFFitResults()
            fitresults.from_runnerase(results["fitresults"])
            self.output.fitresults = fitresults

            weights = HDFWeights()
            weights.from_runnerase(results["weights"])
            self.output.weights = weights

            scaling = HDFScaling()
            scaling.from_runnerase(results["scaling"])
            self.output.scaling = scaling

        # If successful, RuNNer Mode 3 returns the energy and forces of the
        # structure for which it was executed.
        elif self.parameters.runner_mode == 3:
            results = read_results_mode3(directory)
            self.output.energy = results["energy"]
            self.output.forces = results["forces"]

        # Store all calculation results in the project's HDF5 file.
        self.to_hdf()

    def to_hdf(
        self, hdf: Optional[ProjectHDFio] = None, group_name: Optional[str] = None
    ) -> None:
        """Store all job information in HDF5 format.

        Args:
            hdf (ProjectHDFio): HDF5-object which contains the project data.
            group_name (str): Subcontainer name.
        """
        # Replace the runnerase class `SymmetryFunctionSet` by the extended
        # class from the `storageclasses` module which knows how to write itself
        # to hdf.
        sfset = self.parameters.pop("symfunction_short")
        new_sfset = HDFSymmetryFunctionSet()
        new_sfset.from_runnerase(sfset)
        self.parameters.symfunction_short = new_sfset

        GenericJob.to_hdf(self, hdf=hdf, group_name=group_name)
        HasStorage.to_hdf(self, hdf=self.project_hdf5, group_name="")

    def from_hdf(
        self, hdf: Optional[ProjectHDFio] = None, group_name: Optional[str] = None
    ) -> None:
        """Reload all job information from HDF5 format.

        Args:
            hdf (ProjectHDFio): HDF5-object which contains the project data.
            group_name (str): Subcontainer name.
        """
        GenericJob.from_hdf(self, hdf=hdf, group_name=group_name)
        HasStorage.from_hdf(self, hdf=self.project_hdf5, group_name="")
