# coding: utf-8
# Copyright (c) Georg-August-Universität Göttingen - Behler Group
# Distributed under the terms of "New BSD License", see the LICENSE file.
"""Pyiron Hamiltonian for machine-learning with RuNNer.

The RuNNer Neural Network Energy Representation is a framework for the
construction of high-dimensional neural network potentials developed in the
group of Prof. Dr. Jörg Behler at Georg-August-Universität Göttingen.

Attributes
----------
     RunnerJob : GenericJob, HasStorage
        Job class for generating and evaluating potential energy surfaces using
        RuNNer.

Reference
---------
    [RuNNer online documentation](https://theochem.gitlab.io/runner)

FIXME
-----
Paket bitte auch via Conda Forge anbieten
"""

from typing import Optional

import numpy as np
import pandas as pd

from ase.data import atomic_numbers
# from ase.atoms import Atoms as ASEAtoms

from runnerase.io.ase import (read_results_mode1, read_results_mode2,
                              read_results_mode3)
from runnerase import Runner
from runnerase.defaultoptions import DEFAULT_PARAMETERS
# from runnerase.singlepoint import RunnerSinglePointCalculator

from pyiron import Project
from pyiron_base import ProjectHDFio, FlattenedStorage
from pyiron_base import state, Executable, GenericJob, DataContainer
from pyiron_base.generic.object import HasStorage

# from pyiron_atomistics.atomistics.structure.atoms import Atoms

from pyiron_contrib.atomistics.ml.potentialfit import PotentialFit
from pyiron_contrib.atomistics.atomistics.job import (TrainingContainer,
                                                      TrainingStorage)

from .utils import container_to_ase
from .storageclasses import (HDFSymmetryFunctionSet, HDFSymmetryFunctionValues,
                             HDFSplitTrainTest, HDFFitResults, HDFWeights,
                             HDFScaling)

__author__ = 'Alexander Knoll'
__maintainer__ = 'Alexander Knoll'
__email__ = 'alexander.knoll@chemie.uni-goettingen.de'
__copyright__ = 'Copyright 2022, Georg-August-Universität Göttingen - Behler '\
                'Group'
__version__ = '0.1.0'
__status__ = 'development'
__date__ = 'May 11, 2022'


class RunnerJob(GenericJob, HasStorage, PotentialFit):
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

    Examples
    --------
    Starting a new calculation (Mode 1):

    ```python
    from pyiron import Project
    from job import RunnerFit
    from runnerase import read_runnerase

    # Create an empty sample project and a new job.
    pr = Project(path='example')
    job = pr.create_job(RunnerJob, 'mode1')

    # Import RuNNer settings from RuNNer input.nn file using ASE's I/O routines.
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

    __name__ = 'RuNNerJob'

    # These properties are needed by RuNNer as input data (depending on the
    # chosen RuNNer mode).
    _input_properties = ['scaling', 'weights', 'sfvalues', 'splittraintest']

    # Define a default executable.
    _executable = Executable(
        codename='runner',
        module='runner',
        path_binary_codes=state.settings.resource_paths
    )

    def __init__(self, project: Project, job_name: str) -> None:
        """Initialize the class.

        Parameters
        ----------
            project : Project
                The project container where the job is created.
            job_name : str
                The label of the job (used for all directories).
        """
        # Initialize the base class.
        GenericJob.__init__(self, project=project, job_name=job_name)
        HasStorage.__init__(self)

        # Create groups for storing calculation inputs and outputs.
        self.storage.create_group('input')
        self.storage.create_group('output')

        # Create a group for storing the RuNNer configuration parameters.
        self.storage.input.create_group('parameters')
        self.storage.input.parameters.update(DEFAULT_PARAMETERS)

        # Store training data (structures, energies, ...) in a separate node.
        self.storage.input.training_data = TrainingStorage()

        state.publications.add(self.publication)

    @property
    def publication(self):
        """Define relevant publications."""
        return {
            'runner': [
                {
                    'title': 'First Principles Neural Network Potentials for '
                             'Reactive Simulations of Large Molecular and '
                             'Condensed Systems',
                    'journal': 'Angewandte Chemie International Edition',
                    'volume': '56',
                    'number': '42',
                    'year': '2017',
                    'issn': '1521-3773',
                    'doi': '10.1002/anie.201703114',
                    'url': 'https://doi.org/10.1002/anie.201703114',
                    'author': ['Jörg Behler'],
                },
                {
                    'title': 'Constructing high‐dimensional neural network'
                             'potentials: A tutorial review',
                    'journal': 'International Journal of Quantum Chemistry',
                    'volume': '115',
                    'number': '16',
                    'year': '2015',
                    'issn': '1097-461X',
                    'doi': '10.1002/qua.24890',
                    'url': 'https://doi.org/10.1002/qua.24890',
                    'author': ['Jörg Behler'],
                },
                {
                    'title': 'Generalized Neural-Network Representation of '
                             'High-Dimensional Potential-Energy Surfaces',
                    'journal': 'Physical Review Letters',
                    'volume': '98',
                    'number': '14',
                    'year': '2007',
                    'issn': '1079-7114',
                    'doi': '10.1103/PhysRevLett.98.146401',
                    'url': 'https://doi.org/10.1103/PhysRevLett.98.146401',
                    'author': ['Jörg Behler', 'Michelle Parrinello'],
                },
            ]
        }

    @property
    def input(self) -> DataContainer:
        """Show all input properties in storage."""
        return self.storage.input

    @property
    def output(self):
        """Show all calculation output in storage."""
        return self.storage.output

    @property
    def parameters(self):
        """Show the input parameters/settings in storage."""
        return self.storage.input.parameters

    @property
    def scaling(self) -> Optional[HDFScaling]:
        """Show the symmetry function scaling data in storage."""
        if 'scaling' in self.output:
            return self.output.scaling

        return None

    @scaling.setter
    def scaling(self, scaling: HDFScaling) -> None:
        """Set the symmetry function scaling data in storage."""
        self.output.scaling = scaling

    @property
    def weights(self) -> Optional[HDFWeights]:
        """Show the atomic neural network weights data in storage."""
        if 'weights' in self.output:
            return self.output.weights

        return None

    @weights.setter
    def weights(self, weights: HDFWeights) -> None:
        """Set the weights of the atomic neural networks in storage."""
        self.output.weights = weights

    @property
    def sfvalues(self) -> Optional[HDFSymmetryFunctionValues]:
        """Show the symmetry function value data in storage."""
        if 'sfvalues' in self.output:
            return self.output.sfvalues

        return None

    @sfvalues.setter
    def sfvalues(self, sfvalues: HDFSymmetryFunctionValues) -> None:
        """Set the symmetry function value data in storage."""
        self.output.sfvalues = sfvalues

    @property
    def splittraintest(self):
        """Show the split between training and testing data in storage."""
        if 'splittraintest' in self.output:
            return self.output.splittraintest

        return None

    @splittraintest.setter
    def splittraintest(self, splittraintest: HDFSplitTrainTest) -> None:
        """Set the split between training and testing data in storage."""
        self.output.splittraintest = splittraintest

    def _add_training_data(self, container: TrainingContainer) -> None:
        """Add a set of training data to storage."""
        # Get a dictionary of all property arrays saved in this container.
        arrays = container.to_dict()
        arraynames = arrays.keys()

        # Iterate over the structures by zipping the dictionary values together.
        for properties in zip(*arrays.values()):
            zipped = dict(zip(arraynames, properties))
            self.storage.input.training_data.add_structure(**zipped)

    def _get_training_data(self) -> TrainingStorage:
        """Show the stored training data."""
        return self.storage.input.training_data

    def _get_predicted_data(self) -> FlattenedStorage:
        """Show the predicted data after a successful fit."""
        # Energies and forces will only be available after RuNNer Mode 3.
        if 'energy' not in self.output:
            raise RuntimeError('You have to run RuNNer prediction mode '
                               + '(Mode 3) before you can access predictions.')

        pred_properties = {'energy': None, 'forces': None}

        # Get a list of structures and energies.
        structures = list(self.training_data.iter_structures())

        # Get the values of all properties RuNNer can predict for a structure.
        for prop in pred_properties:
            if prop in self.output:
                pred_properties[prop] = self.output[prop]

        predicted_data = FlattenedStorage()

        for structure, energy, force in zip(structures,
                                            pred_properties['energy'],
                                            pred_properties['forces']):
            predicted_data.add_chunk(len(structure), energy=energy,
                                     forces=force)

        return predicted_data

    def get_lammps_potential(self) -> pd.DataFrame:
        """Return a pandas dataframe with information for setting up LAMMPS."""
        if not self.status.finished:
            raise RuntimeError('LAMMPS potential can only be generated after a '
                               + 'successful fit.')

        if 'weights' not in self.output or 'scaling' not in self.output:
            raise RuntimeError('This potential has not been trained yet.')

        # Get all elements in the training dataset.
        elements = self.training_data.get_elements()

        # Create a list of all files needed by the potential.
        files = [f'{self.working_directory}/input.nn',
                 f'{self.working_directory}/scaling.data']

        # Add the weight files.
        for elem in elements:
            atomic_number = atomic_numbers[elem]
            filename = f'weights.{atomic_number:03}.data'
            files.append(f'{self.working_directory}/{filename}')

        # Save the mapping of elements between LAMMPS and n2p2.
        emap = ','.join([f'{i}:{el}' for i, el in enumerate(elements)])

        # Get the cutoff radius of the symmetry functions.
        cutoffs = self.parameters.symfunction_short.cutoffs
        cutoff = cutoffs[0]

        if len(cutoffs) > 1:
            raise RuntimeError('LAMMPS potential can only be generated for a '
                               + 'a uniform cutoff radius.')

        return pd.DataFrame({
            'Name': [f"RuNNer-{''.join(elements)}"],
            'Filename': [files],
            'Model': ['RuNNer'],
            'Species': [elements],
            'Config': [['pair_style nnp dir "./" '
                        + 'showew no showewsum 0 resetew no maxew 100 '
                        + 'cflength 1.8897261328 cfenergy 0.0367493254 '
                        + f'emap "{emap}"\n',
                        f'pair_coeff * * {cutoff}\n']]
        })

    def write_input(self) -> None:
        """Write the relevant job input files.

        This routine writes the input files for the job using the ASE Runner
        calculator.
        """
        input_properties = {'sfvalues': None, 'splittraintest': None,
                            'weights': None, 'scaling': None}

        for prop in input_properties:
            if prop in self.output and self.output[prop] is not None:
                input_properties[prop] = self.output[prop].to_runnerase()

        # Create an ASE Runner calculator object.
        # Pay attention to the different name: `structures` --> `dataset`.
        calc = Runner(
            label='pyiron',
            dataset=container_to_ase(self.training_data),
            scaling=input_properties['scaling'],
            weights=input_properties['weights'],
            sfvalues=input_properties['sfvalues'],
            splittraintest=input_properties['splittraintest'],
            **self.parameters.to_builtin()
        )

        # Set the correct elements of the system.
        calc.set_elements()

        # If no seed was specified yet, choose a random value.
        if 'random_seed' not in calc.parameters:
            calc.set(random_seed=np.random.randint(1, 1000))

        # If no dataset was attached to the calculator, the single structure
        # stored as the atoms property will be written instead.
        atoms = calc.dataset or calc.atoms

        # Set the correct calculation directory and file prefix.
        calc.directory = self.working_directory
        calc.prefix = f'mode{self.parameters.runner_mode}'

        # Set the 'flags' which ASE uses to see which input files need to
        # be written.
        targets = {1: 'sfvalues', 2: 'fit', 3: 'energy'}

        calc.write_input(
            atoms,
            targets[self.parameters.runner_mode],
            system_changes=None
        )

    def _executable_activate(self, enforce: bool = False) -> None:
        """Link to the RuNNer executable."""
        if self._executable is None or enforce:
            self._executable = Executable(
                codename='runner',
                module='runner',
                path_binary_codes=state.settings.resource_paths
            )

    def collect_output(self) -> None:
        """Read and store the job results."""
        # Compose job label (needed by the ASE I/O routines) and store dir.
        label = f'{self.working_directory}/mode{self.parameters.runner_mode}'
        directory = self.working_directory

        # If successful, RuNNer Mode 1 returns symmetry function values for
        # each structure and the information, which structure belongs to the
        # training and which to the testing set.
        if self.parameters.runner_mode == 1:
            results = read_results_mode1(label, directory)

            # Transform sfvalues into the pyiron class for HDF5 storage and
            # store it in the output dictionary.
            sfvalues = HDFSymmetryFunctionValues()
            sfvalues.from_runnerase(results['sfvalues'])
            self.output.sfvalues = sfvalues

            # Transform split data between training and testing set into the
            # pyiron class for HDF5 storage and store it in the output
            # dictionary.
            splittraintest = HDFSplitTrainTest()
            splittraintest.from_runnerase(results['splittraintest'])
            self.output.splittraintest = splittraintest

        # If successful, RuNNer Mode 2 returns the weights of the atomic neural
        # networks, the symmetry function scaling data, and the results of the
        # fitting process.
        elif self.parameters.runner_mode == 2:
            results = read_results_mode2(label, directory)

            fitresults = HDFFitResults()
            fitresults.from_runnerase(results['fitresults'])
            self.output.fitresults = fitresults

            weights = HDFWeights()
            weights.from_runnerase(results['weights'])
            self.output.weights = weights

            scaling = HDFScaling()
            scaling.from_runnerase(results['scaling'])
            self.output.scaling = scaling

        # If successful, RuNNer Mode 3 returns the energy and forces of the
        # structure for which it was executed.
        elif self.parameters.runner_mode == 3:
            results = read_results_mode3(directory)
            self.output.energy = results['energy']
            self.output.forces = results['forces']

        # Store all calculation results in the project's HDF5 file.
        self.to_hdf()

    def to_hdf(
        self,
        hdf: Optional[ProjectHDFio] = None,
        group_name: Optional[str] = None
    ) -> None:
        """Store all job information in HDF5 format.

        Parameters
        ----------
            hdf : ProjectHDFio
                HDF5-object which contains the project data.
            group_name : str
                Subcontainer name.
        """
        # Replace the runnerase class `SymmetryFunctionSet` by the extended
        # class from the `storageclasses` module which knows how to write itself
        # to hdf.
        sfset = self.parameters.pop('symfunction_short')
        new_sfset = HDFSymmetryFunctionSet()
        new_sfset.from_runnerase(sfset)
        self.parameters.symfunction_short = new_sfset

        GenericJob.to_hdf(self, hdf=hdf, group_name=group_name)
        HasStorage.to_hdf(self, hdf=self.project_hdf5, group_name='')

    def from_hdf(
        self,
        hdf: Optional[ProjectHDFio] = None,
        group_name: Optional[str] = None
    ) -> None:
        """Reload all job information from HDF5 format.

        Parameters
        ----------
            hdf : ProjectHDFio
                HDF5-object which contains the project data.
            group_name : str
                Subcontainer name.
        """
        GenericJob.from_hdf(self, hdf=hdf, group_name=group_name)
        HasStorage.from_hdf(self, hdf=self.project_hdf5, group_name='')
