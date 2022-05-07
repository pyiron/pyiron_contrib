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
Group name wieder rausschmeißen
SymmetryFunctionValues no Class, instead make it a function each.
np.nan init ist nicht noetig, 0.0 geht auch.
theoretisch wäre das besser das nicht direkt zu machen.
ersetze trainingcontainer durch trainingstorage
Paket bitte auch via Conda Forge anbieten
Try inhereting from PotentialFit class

use one trainingstorage, one for runner input structures and one for runner
output structures.
"""

from typing import Optional, List, Union

import numpy as np

from ase.atoms import Atoms as ASEAtoms

from runnerase.io.ase import (read_results_mode1, read_results_mode2,
                              read_results_mode3)
from runnerase import Runner
from runnerase.defaultoptions import DEFAULT_PARAMETERS
from runnerase.singlepoint import RunnerSinglePointCalculator

from pyiron import Project
from pyiron_base import ProjectHDFio
from pyiron_base import state, Executable, GenericJob, DataContainer
from pyiron_base.generic.object import HasStorage

from pyiron_atomistics.atomistics.structure.atoms import Atoms

from ..atomistics.job.trainingcontainer import TrainingContainer
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
__date__ = 'May 02, 2022'


class RunnerJob(GenericJob, HasStorage):
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

    # Import training dataset and RuNNer settings from RuNNer input.data
    # and input.nn files using ASE's I/O routines.
    structures, options = read_runnerase('./')

    # Attach the information to the job.
    job.structures = structures
    job.input.update(options)

    # Set the RuNNer Mode to 1.
    job.input.runner_mode = 1

    job.run()
    ```

    Restarting Mode 1 and running Mode 2:

    ```python
    job = Project('example')['mode1'].restart('mode2')
    job.input.runner_mode = 2
    job.run()
    ```

    Restarting Mode 2 and running Mode 3:

    ```python
    job = Project('runnertest')['mode2'].restart('mode3')
    job.input.runner_mode = 3
    job.run()
    ```
    """

    __name__ = 'RuNNerJob'

    # These properties are needed by RuNNer as input data (depending on the
    # chosen RuNNer mode).
    _input_properties = ['scaling', 'weights', 'sfvalues', 'splittraintest']

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
        # FIXME: use TrainingStorage instead.
        self.storage.structures = TrainingContainer(project, job_name)

        # Initialize optional RuNNer input data as class properties.
        self.storage.sfvalues = HDFSymmetryFunctionValues()
        self.storage.splittraintest = HDFSplitTrainTest()
        self.storage.scaling = HDFScaling()
        self.storage.weights = HDFWeights()

        # Set default parameters.
        self.storage.input.update(DEFAULT_PARAMETERS)

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
        """Show input options in storage."""
        return self.storage.input

    @property
    def output(self):
        """Show all calculation output in storage."""
        return self.storage.output

    @property
    def scaling(self):
        """Show the symmetry function scaling data in storage."""
        return self.storage.scaling

    @property
    def weights(self):
        """Show the atomic neural network weights data in storage."""
        return self.storage.weights

    @weights.setter
    def weights(self, weights: HDFWeights):
        """Set the weights of the atomic neural networks in storage."""
        self.storage.weights = weights

    @property
    def sfvalues(self):
        """Show the symmetry function value data in storage."""
        return self.storage.sfvalues

    @property
    def splittraintest(self):
        """Show the split between training and testing data in storage."""
        return self.storage.splittraintest

    @property
    def structures(self) -> TrainingContainer:
        """Store a dataset consisting of many chemical structures."""
        return self.storage.structures

    @structures.setter
    def structures(self, container: TrainingContainer) -> None:
        """Append structures to storage.

        Add a list of structures to storage.

        Parameters
        ----------
            structures : List[Atoms]
                A list of ASE Atoms objects or Pyiron Atoms objects which are
                to be stored.

        Examples
        --------
        ```python
        >>> job.structures = [structure1, structure2]
        >>> job.structures = [structure3, structure4]
        >>> print(job.structures)
        [structure1, structure2, structure3, structure4]
        ```
        """
        self.storage.structures = container

    def write_input(self) -> None:
        """Write the relevant job input files.

        This routine writes the input files for the job using the ASE Runner
        calculator.
        """
        # Create an ASE Runner calculator object.
        # Pay attention to the different name: `structures` --> `dataset`.
        calc = Runner(
            label='pyiron',
            dataset=container_to_ase(self.structures),
            scaling=self.storage.scaling.to_runnerase(),
            weights=self.storage.weights.to_runnerase(),
            sfvalues=self.storage.sfvalues.to_runnerase(),
            splittraintest=self.storage.splittraintest.to_runnerase(),
            **self.input.to_builtin()
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
        calc.prefix = f'mode{self.input.runner_mode}'

        # Set the 'flags' which ASE uses to see which input files need to
        # be written.
        targets = {1: 'sfvalues', 2: 'fit', 3: 'energy'}

        calc.write_input(
            atoms,
            targets[self.input.runner_mode],
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
        label = f'{self.working_directory}/mode{self.input.runner_mode}'
        directory = self.working_directory

        # If successful, RuNNer Mode 1 returns symmetry function values for
        # each structure and the information, which structure belongs to the
        # training and which to the testing set.
        if self.input.runner_mode == 1:
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
        elif self.input.runner_mode == 2:
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
        elif self.input.runner_mode == 3:
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
        sfset = self.storage.input.pop('symfunction_short')
        new_sfset = HDFSymmetryFunctionSet()
        new_sfset.from_runnerase(sfset)
        self.input.symfunction_short = new_sfset

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

    def restart(
        self,
        job_name: Optional[str] = None,
        job_type: Optional[str] = None
    ) -> 'RunnerJob':
        """Reload all calculation details.

        This procedure extends the base class `restart()` by setting results
        from previous calculations as input parameters to the new calculation.
        The recognized properties depend on the class variable
        self._input_properties (see class docstring for further details).

        Returns
        -------
            new_ham : RunnerJob
                the newly created RunnerJob object.
        """
        # Call the base class routine to generate the new Hamiltonian, which is
        # a RunnerJob class instance.
        # At this point, the Hamiltonian holds the input parameters,
        # structures, and outputs of the previous calculation. However, it
        # cannot access the relevant properties as input values which is
        # necessary for starting a new calculation.
        new_ham = super().restart(job_name, job_type)

        for prop in self._input_properties:
            if prop in self.output.keys():
                new_ham.storage[prop] = self.output[prop]

        return new_ham
