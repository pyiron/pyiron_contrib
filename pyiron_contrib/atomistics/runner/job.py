# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational
# Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""Pyiron Hamiltonian for machine-learning with RuNNer.

The RuNNer Neural Network Energy Representation is a framework for the
construction of high-dimensional neural network potentials developed in the
group of Prof. Dr. Jörg Behler at Georg-August-Universität Göttingen.

Provides:
    - PotentialFittingBase (GenericJob): Future base class for all fitting
                                         codes in pyiron. Originally provided
                                         in the AtomicRex module of
                                         pyiron_contrib.
    - RunnerStructureContainer (StructureStorage): Storage container for
                                                   RuNNer training datasets.

FIXME:
At this point, RunnerFit heavily relies on an unpublished version of an ASE
calculator for RuNNer.

Reference:
    - RuNNer online documentation](https://theochem.gitlab.io/runner)
"""

import numpy as np

from runnerase.io.ase import (read_results_mode1, read_results_mode2,
                              read_results_mode3)
from runnerase import Runner
from runnerase.defaultoptions import DEFAULT_PARAMETERS
from runnerase.singlepoint import RunnerSinglePointCalculator

from pyiron_base import state, Executable, DataContainer, GenericJob
from pyiron_base.generic.object import HasStorage
from pyiron_atomistics.atomistics.structure.atoms import pyiron_to_ase

from ..atomistics.job.trainingcontainer import TrainingContainer

__author__ = 'Alexander Knoll'
__copyright__ = 'Copyright 2021, Georg-August-Universität Göttingen - Behler '\
                'Group'
__version__ = '0.1.0'
__maintainer__ = 'Alexander Knoll'
__email__ = 'alexander.knoll@chemie.uni-goettingen.de'
__status__ = 'development'
__date__ = 'April 28, 2022'


class RunnerTrainingContainer(TrainingContainer):
    """Store chemical structures as a Runner training dataset."""

    def to_ase(self):
        """Convert all attached structures to a list of ASE Atoms objects."""
        structure_lst = []
        for idx, structure in enumerate(self.iter_structures()):
            # Retrieve all properties, i.e. energy, forces, etc.
            energy = self._container.get_array('energy', idx)
            forces = self._container.get_array('forces', idx)
            charges = self._container.get_array('charges', idx)
            totalcharge = self._container.get_array('totalcharge', idx)

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


class RunnerFit(GenericJob, HasStorage):
    """Generate a potential energy surface using RuNNer.

    The RuNNer Neural Network Energy Representation (RuNNer) is a Fortran code
    for the generation of high-dimensional neural network potentials actively
    developed in the group of Prof. Dr. Jörg Behler at Georg-August-Universität
    Göttingen.

    RuNNer operates in three different modes:
        - Mode 1: Calculation of symmetry function values. Symmetry functions
                  are many-body descriptors for the chemical environment of an
                  atom.
        - Mode 2: Fitting of the potential energy surface.
        - Mode 3: Prediction. Use the previously generated high-dimensional
                  potential energy surface to predict the energy and force
                  of an unknown chemical configuration.

    The different modes generate a lot of output:
        - Mode 1:
            - sfvalues:       The values of the symmetry functions for each
                              atom.
            - splittraintest: which structures belong to the training and which
                              to the testing set. ASE needs this
                              information to generate the relevant input files
                              for RuNNer Mode 2.
        - Mode 2:
            - weights:        The neural network weights.
            - scaling:        The symmetry function scaling factors.

        - Mode 3: predict the total energy and atomic forces for a structure.

    Examples:
    Starting a new calculation (Mode 1):

    ```python
        from pyiron import Project
        from job import RunnerFit
        import ase.io.runner.runner as io

        # Create an empty sample project and a new job.
        pr = Project(path='example')
        job = pr.create_job(RunnerFit, 'mode1')

        # Import training dataset and RuNNer settings from RuNNer input.data
        # and input.nn files using ASE's I/O routines.
        structures, options = io.read_runnerase('./')

        # Attach the information to the job.
        job.structures = structures
        job.input.update(options)

        # Set the RuNNer Mode to 1.
        job.input.runner_mode = 1

        job.run()

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

    __name__ = 'RuNNer'

    # These properties are needed by RuNNer as input data (depending on the
    # chosen RuNNer mode).
    _input_properties = ['scaling', 'weights', 'sfvalues', 'splittraintest']

    # Define a default executable.
    _executable = Executable(
        codename='runner',
        module='runner',
        path_binary_codes=state.settings.resource_paths
    )

    def __init__(self, project, job_name, **kwargs):
        """Initialize the class.

        Args:
            project (Project): The project container where the job is created.
            job_name (str):  The label of the job (used for all directories).
        """
        # Initialize the base class.
        GenericJob.__init__(self, project=project, job_name=job_name)
        HasStorage.__init__(self)

        self.storage.create_group('input')
        self.storage.create_group('output')
        self.storage.structures = RunnerTrainingContainer(project, job_name)

        self.storage.input.update(DEFAULT_PARAMETERS)

        for prop in self._input_properties:
            self.storage.create_group(prop)

            val = kwargs.pop(prop, None)
            if val is not None:
                self.storage[prop] = DataContainer(val)

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
    def scaling(self):
        """Show the symmetry function scaling data in storage."""
        return self.storage.scaling

    @property
    def weights(self):
        """Show the atomic neural network weights data in storage."""
        return self.storage.weights

    @property
    def sfvalues(self):
        """Show the symmetry function value data in storage."""
        return self.storage.sfvalues

    @property
    def splittraintest(self):
        """Show the split between training and testing data in storage."""
        return self.storage.splittraintest

    @property
    def input(self):
        """Show input options in storage."""
        return self.storage.input

    @property
    def output(self):
        """Show all calculation output in storage."""
        return self.storage.output

    @property
    def structures(self):
        """Store a dataset consisting of many chemical structures."""
        return self.storage.structures

    @structures.setter
    def structures(self, structures):
        """Store a dataset consisting of many chemical structures.

        Args:
            structures (list): A list of ASE Atoms objects or Pyiron Atoms
                               objects which are to be stored.
        """
        for structure in structures:
            energy = structure.get_potential_energy()

            if isinstance(structure.calc, RunnerSinglePointCalculator):
                totalcharge = structure.calc.get_property('totalcharge')
            else:
                totalcharge = np.sum(structure.get_initial_charges())

            self.storage.structures.include_structure(structure, energy,
                                                      totalcharge=totalcharge)

    def write_input(self):
        """Write the relevant job input files.

        This routine writes the input files for the job using the ASE Runner
        calculator.
        """
        # Create an ASE Runner calculator object.
        # Pay attention to the different name: `structures` --> `dataset`.
        calc = Runner(
            label='pyiron',
            dataset=self.structures.to_ase(),
            scaling=self.scaling.to_builtin(),
            weights=self.weights.to_builtin(),
            sfvalues=self.sfvalues.to_builtin(),
            splittraintest=self.splittraintest.to_builtin(),
            **self.input.to_builtin()
        )

        # If no dataset were attached to the calculator, the single structure
        # stored as the atoms property would be written instead.
        atoms = calc.dataset or calc.atoms
        calc.directory = self.working_directory

        # Set the 'flags' which ASE uses to see which input files are written.
        targets = {1: 'sfvalues', 2: 'fit', 3: 'energy'}
        system_changes = None  # Always perform a new calculation.

        calc.write_input(
            atoms,
            targets[self.input['runner_mode']],
            system_changes
        )

    def _executable_activate(self, enforce=False):
        """Link to the Runner executable."""
        if self._executable is None or enforce:
            self._executable = Executable(
                codename='runner',
                module='runner',
                path_binary_codes=state.settings.resource_paths
            )

    def collect_output(self):
        """Read and store the job results."""
        # Compose job label (needed by the ASE I/O routines) and store dir.
        label = f'{self.working_directory}/mode{self.input.runner_mode}'
        directory = self.working_directory

        # If successful, RuNNer Mode 1 returns symmetry function values for
        # each structure and the information, which structure belongs to the
        # training and which to the testing set.
        if self.input.runner_mode == 1:
            sfvalues, splittraintest = read_results_mode1(label, directory)
            self.output.sfvalues = sfvalues
            self.output.splittraintest = splittraintest

        # If successful, RuNNer Mode 2 returns the weights of the atomic neural
        # networks, the symmetry function scaling data, and the results of the
        # fitting process.
        if self.input.runner_mode == 2:
            fitresults, weights, scaling = read_results_mode2(
                label,
                directory
            )
            self.output.fitresults = fitresults
            self.output.weights = weights
            self.output.scaling = scaling

        # If successful, RuNNer Mode 3 returns the energy and forces of the
        # structure for which it was executed.
        if self.input.runner_mode == 3:
            energy, forces = read_results_mode3(directory)
            self.output.energy = energy
            self.output.forces = forces

        # Store all calculation rsults in the project's HDF5 file.
        self.to_hdf()

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store all job information in HDF5 format.

        Args:
            hdf (ProjectHDFio, optional): HDF5-object which contains the
                                          project data.
            group_name (str, optional):   Subcontainer name.
        """
        GenericJob.to_hdf(self, hdf=hdf, group_name=group_name)
        HasStorage.to_hdf(self, hdf=self.project_hdf5, group_name="")

    def from_hdf(self, hdf=None, group_name=None):
        """
        Reload all job information from HDF5 format.

        Args:
            hdf (ProjectHDFio, optional): HDF5-object which contains the
                                          project data.
            group_name (str, optional):   Subcontainer name.
        """
        GenericJob.from_hdf(self, hdf=hdf, group_name=group_name)
        HasStorage.from_hdf(self, hdf=self.project_hdf5, group_name="")

    def restart(self, *args, **kwargs):
        """
        Reload all calculation details.

        This procedure extends the base class `restart()` by setting results
        from previous calculations as input parameters to the new calculation.
        The recognized properties depend on the class variable
        self._input_properties (see class docstring for further details).

        Returns:
            new_ham (RunnerFit): the newly created RunnerFit object.
        """
        # Call the base class routine to generate the new Hamiltonian, which is
        # a RunnerFit class instance.
        # At this point, the Hamiltonian holds the input parameters,
        # structures, and outputs of the previous calculation. However, it
        # cannot access the relevant properties as input values which is
        # necessary for starting a new calculation.
        new_ham = super().restart(*args, **kwargs)

        for prop in self._input_properties:
            if prop in self.output.keys():
                new_ham.storage[prop] = DataContainer(self.output[prop])

        return new_ham
