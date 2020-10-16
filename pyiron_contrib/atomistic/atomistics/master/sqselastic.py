# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base import InputList, ParallelMaster, JobGenerator
from pyiron_base.master.flexible import FlexibleMaster
import numpy as np
from pyiron import Atoms
from pyiron.atomistics.job.atomistic import AtomisticGenericJob
from pyiron.atomistics.job.sqs import SQSJob
from pyiron_mpie.interactive.elastic import ElasticMatrixJob
import matplotlib.pyplot as plt
import seaborn as sns

"""
Calculate the elastic matrices for special quasi-random structures.
"""
# TODO: Add citation information to the publication attribute
# TODO: Add warnings to the warnings attribute (once it exists)

__author__ = "Liam Huber"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Oct 2, 2020"


class StatsArray:
    """
    A convenience class for wrapping arrays of data where one of the dimensions spans across some statistical variation.

    Attributes:
        axis (int): The axis of the data over which to take statistical measures.
    """

    def __init__(self, data, axis=0):
        self._data = np.array(data)
        self.axis = axis

    @property
    def array(self):
        return self._data

    @property
    def mean(self):
        return self._data.mean(axis=self.axis)

    @property
    def std(self):
        return self._data.std(axis=self.axis)

    @property
    def sem(self):
        return self._data.std(axis=self.axis) / len(self._data)

    def __repr__(self):
        return str(self._data)

    def to_hdf(self, hdf, group_name):
        # TODO: This is not actually ever used, but is useful if InputList started leaning on childrens' hdf methods
        with hdf.open(group_name) as hdf5:
            hdf5['data'] = self._data

    def from_hdf(self, hdf, group_name):
        # TODO: This is not actually ever used, but is useful if InputList started leaning on childrens' hdf methods
        with hdf.open(group_name) as hdf5:
            self._data = hdf5['data']


# TODO: Right now this decorator is only used in *one* place, so we could just get rid of it.
def _as_stats_array(axis=0):
    """
    A decorator to wrap function output as a `StatsArray`. Ideal for use together with an `@property`.

    Args:
        axis (int): Which axis the `StatsArray` should calculate statistics over. (Default is 0.)
    """
    def stats_array_wrapper(fnc):
        def decorated(*args, **kwargs):
            return StatsArray(fnc(*args, **kwargs), axis=axis)
        return decorated
    return stats_array_wrapper


# TODO: FlexibleMaster throws an IndentationError when you try to load if this is a method of SQSElasticConstants
def _sqs2minimization(sqs_job, min_job):
    min_job.structure_lst = sqs_job.list_structures()


# TODO: FlexibleMaster throws an IndentationError when you try to load if this is a method of SQSElasticConstants
def _minimization2elastic(min_job, elastic_job):
    elastic_job.structure_lst = [
        min_job['struct_{}'.format(int(i))].get_structure()
        for i in np.arange(len(min_job))
    ]


class SQSElasticConstants(FlexibleMaster):
    """
    Calculates the elastic constants for structures generated according to the special quasi-random structure scheme.

    This job itself takes no input, but rather the input is passed in via the reference jobs: `ref_ham` handles all
    input for the force/energy evaluations (e.g. emperical potential if a classical interpreter, kpoints, energy cutoff,
    etc. for quantum interpreters...), `ref_sqs` handles the SQS input like the molar fractions and how many
    structures to produce, and `ref_elastic` holds the input for evaluating the elastic constants, like what strain
    magnitude to apply. See the docstrings of these individual jobs for more details.

    Output is constructed from the output of the underlying child jobs and will become invalid if their data is tampered
    with.

    Attributes:
        ref_ham (pyiron.atomistics.job.atomistic.AtomisticGenericJob): The interpreter for calculating forces and
            energies from the atomic structure.
        ref_sqs (pyiron.atomistics.job.sqs.SQSJob): Uses the special quasi-random structure approach to randomize the
            species in `ref_ham.structure` according to the molar fractions provided in its input.
        ref_elastic (pyiron_mpie.interactive.elastic.ElasticMatrixJob): Calculates elastic constants for each of the SQS
            structures.
        output (SQSElasticOutput): Output collected from the various child jobs.
        sqs_job_name (str): The name of the sqs child job.
        min_job_name (str): The name of the minimiztion list child job.
        elastic_job_name (str): The name of the elastic constants list child job.

    Output:
        elastic_matrices (StatsArray): The six-component elastic matrices.
        residual_pressures (StatsArray): The remaining pressures after minimization.
        structures (list): The minimized structures.
        cells (StatsArray): The cells of the minimized structures.
        symbols (list): The chemical symbols (possibly including '0' for vacancy) in the SQS structures.
        compositions (dict): The actual chemical fraction of each species (including '0' for vacancy) in the structures.

    Methods:
        get_elastic_output: A function taking a string key for accessing other output from the `ElasticMatrixJob` as a
            `StatsArray`. (If you ask for an invalid key, all the valid possibilities will be printed for you.)

    Note: The `StatsArray` class is a wrapper for numpy arrays that allows you to look at the `.array`, `.mean`,
        `.std` and `.sem` (standard error) with respect to the different SQS structures.

    Warning: Initial relaxation is only *isotropic* with respect to cell shape!
    """
    def __init__(self, project, job_name):
        super().__init__(project, job_name=job_name)
        self.__name__ = "SQSElasticConstants"
        self.__version__ = "0.1"
        self.__hdf_version__ = "0.2.0"

        self.ref_ham = None
        self.ref_sqs = None
        self.ref_elastic = None

        self.output = SQSElasticOutput(table_name='output', axis=0)

        self._ham_ref_name_tail = 'ham_ref'
        self._sqs_ref_name_tail = 'sqs_ref'
        self._sqs_job_name_tail = 'sqs'
        self._min_ref_name_tail = 'min_ref'
        self._min_job_name_tail = 'min'
        self._elastic_ref_ham_name_tail = 'el_ref'
        self._elastic_ref_single_name_tail = 'el_job'
        self._elastic_job_name_tail = 'elastic'
        # TODO: Once creating/copying jobs as HDF children is sorted out, this attribute salad can probably be
        #  refactored away (or at least simplified)

    @property
    def sqs_job_name(self):
        return self._relative_name(self._sqs_job_name_tail)

    @property
    def min_job_name(self):
        return self._relative_name(self._min_job_name_tail )

    @property
    def elastic_job_name(self):
        return self._relative_name(self._elastic_job_name_tail)

    def validate_ready_to_run(self):
        self._create_pipeline()
        super().validate_ready_to_run()

    def _create_pipeline(self):
        self.append(self._instantiate_sqs())
        self.function_lst.append(_sqs2minimization)
        self.append(self._instantiate_minimization())
        self.function_lst.append(_minimization2elastic)
        self.append(self._instantiate_elastic())

    def _instantiate_sqs(self):
        # sqs_job = self.create_job(self._job_type.SQSJob, self._relative_name('sqs_job'))
        # sqs_job.input = self.ref_sqs.input
        # TODO: Get the sqs job created as a proper HDF child
        sqs_job = self._copy_job(self.ref_sqs, self._sqs_job_name_tail)
        # sqs_job = self.ref_sqs.copy_to(
        #     project=self._hdf5,
        #     new_job_name=self._relative_name(self._sqs_job_name_tail),
        #     new_database_entry=False
        # )
        sqs_job.input.mole_fractions = dict(sqs_job.input.mole_fractions)  # Input expects dict but gets InputList(dict)
        sqs_job.structure = self.ref_ham.structure.copy()
        return sqs_job

    def _create_job(self, job_type, name):
        return self.project.create_job(job_type, self._relative_name(name))

    def _relative_name(self, name):
        # TODO: Once things are being created
        return '_'.join([self.job_name, name])

    @property
    def _job_type(self):
        return self.project.job_type

    def _instantiate_minimization(self):
        min_ref = self._copy_job(self.ref_ham, self._min_ref_name_tail)
        # TODO: Get the min ref saved as a proper HDF child
        min_ref.calc_minimize(pressure=0)
        # min_job = self.create_job(self._job_type.StructureListMaster, self._relative_name('min'))
        # TODO: Get the min job created as a proper HDF child
        min_job = self._create_job(self._job_type.StructureListMaster, self._min_job_name_tail)
        min_job.ref_job = min_ref
        return min_job

    def _copy_job(self, job, name):
        return job.copy_to(
            new_job_name='_'.join([self.job_name, name]),  # , job.job_name
            new_database_entry=False
        )

    def _instantiate_elastic(self):
        elastic_ref = self._copy_job(self.ref_ham, self._elastic_ref_ham_name_tail)
        # TODO: Get the elastic ref saved as a proper HDF child

        # This should work, but doesn't maintain the elastic ref input and goes back to class defaults:
        # elastic_job = elastic_ref.create_job(self._job_type.ElasticMatrixJob, self._relative_name('el_job'))
        # return elastic_job.create_job(self._job_type.StructureListMaster, self._relative_name('el_list_job'))

        # elastic_job = self._copy_job(self.ref_elastic, 'el_job')  # ValueError: Unknown item: se_el_ham_ref

        # elastic_job = self.create_job(self._job_type.ElasticMatrixJob, self._relative_name('el_job'))
        # TODO: Get the elastic job created as a proper HDF child
        elastic_job = self._create_job(self._job_type.ElasticMatrixJob, self._elastic_ref_single_name_tail)
        elastic_job.input = self.ref_elastic.input
        elastic_job.ref_job = elastic_ref
        elastic_job_list = self._create_job(self._job_type.StructureListMaster, self._elastic_job_name_tail)
        elastic_job_list.ref_job = elastic_job
        return elastic_job_list

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf, group_name)
        self._ensure_has_references()
        self._ensure_references_saved()
        self._hdf5['refjobs/refhamname'] = self.ref_ham.job_name
        self._hdf5['refjobs/refsqsname'] = self.ref_sqs.job_name
        self._hdf5['refjobs/refelasticname'] = self.ref_elastic.job_name
        self.output.to_hdf(hdf=self._hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf, group_name)
        self.ref_ham = self.project.load(self._hdf5['refjobs/refhamname'])
        self.ref_sqs = self.project.load(self._hdf5['refjobs/refsqsname'])
        self.ref_elastic = self.project.load(self._hdf5['refjobs/refelasticname'])
        self.output.from_hdf(hdf=self._hdf5)

    def _ensure_has_references(self):
        for attribute_name, class_ in zip(
                ['ref_ham', 'ref_sqs', 'ref_elastic'],
                [AtomisticGenericJob, SQSJob, ElasticMatrixJob],
        ):
            job = getattr(self, attribute_name)
            if not isinstance(job, class_):
                raise TypeError('{} expected a {} with type {} but got {}'.format(
                    self.job_name,
                    attribute_name,
                    class_.__name__,
                    type(job)
                ))

    def _ensure_references_saved(self):
        # TODO: Once creating subjobs as proper HDF children is sorted, perhaps this can all be removed
        if self.ref_ham.job_name != self._relative_name(self._ham_ref_name_tail):
            self.ref_ham = self._copy_job(self.ref_ham, self._ham_ref_name_tail)

        if self.ref_sqs.job_name != self._relative_name(self._sqs_ref_name_tail):
            self.ref_sqs = self._copy_job(self.ref_sqs, self._sqs_ref_name_tail)

        if self.ref_elastic.ref_job is None \
                or self.ref_elastic.ref_job.job_name != self._relative_name(self._elastic_ref_ham_name_tail):
            self.ref_elastic.ref_job = self._copy_job(self.ref_ham, self._elastic_ref_ham_name_tail)
        if self.ref_elastic.job_name != self._relative_name(self._elastic_ref_single_name_tail):
            self.ref_elastic = self._copy_job(self.ref_elastic, self._elastic_ref_single_name_tail)

        # TODO: Once `copy_to` doesn't ignore new names when a new project is given, try this again:
        # self.ref_ham = self._copy_to_self(self.ref_ham, self._ham_ref_name_tail)
        # self.ref_sqs = self._copy_to_self(self.ref_sqs, self._sqs_ref_name_tail)
        # self.ref_elastic = self._copy_to_self(self.ref_elastic, self._elastic_ref_single_name_tail)
        # self.ref_elastic.ref_job = self._copy_to_parent(self.ref_ham, self.ref_elastic, self._elastic_ref_ham_name_tail)

        for job in [self.ref_ham, self.ref_sqs, self.ref_elastic]:
            # TODO: It should be possible to wrap this whole routine in some sort of `if self.status.created` or
            #  something so that we can just job.save() instead of being careful
            #  Or maybe once the references are inside the self._hdf5 envelope we won't need this at all?
            self._save_reference_if_new(job)

    @staticmethod
    def _copy_to_parent(job, parent, name):
        # TODO: Waiting for `copy_to` to get fixed to be useful
        return job.copy_template(project=parent._hdf5, new_job_name=name)

    def _copy_to_self(self, job, name):
        # TODO: Waiting for `copy_to` to get fixed to be useful
        return self._copy_to_parent(job, self, name)

    def _save_reference_if_new(self, job):
        # TODO: Rework your saving routine so this method is superfluous, then kill it
        if job.job_name not in self.project.list_nodes():
            job.save()
        else:
            job.to_hdf()

    def run_static(self):
        super().run_static()
        # Flexible master does not use output collection, but we want that so let's force the issue:
        self.status.collect = True
        self.run()

    def collect_output(self):
        self.output.n_sites = len(self[self.min_job_name]['input/structures/s_0/positions'])

        self.output.n_structures = len(self[self.min_job_name]['input/structures'].list_groups())

        self.output.elastic_matrices = self.get_elastic_output('C').array

        self.output.residual_pressures = [
            self[self.min_job_name]['struct_{}/output/generic/pressures'.format(n)][-1]
            for n in np.arange(self.output.n_structures)
        ]

        # self.output.structures = [
        #     Atoms().from_hdf(self[self.min_job_name]['struct_{}/output'.format(n)])
        #     for n in np.arange(self.output.n_structures)
        # ]
        # TODO: InputList does not yet handle saving a list of structures

        # self.output.cells = [structure.cell.array for structure in self.output.structures]
        # TODO: Awaiting structures output
        self.output.cells = [
            self[self.min_job_name]['struct_{}/output/structure/cell/cell'.format(n)]
            for n in np.arange(self.output.n_structures)
        ]

        self.output.symbols = list(self[self.sqs_job_name]['input/custom_dict/data']['mole_fractions'].keys())

        # struct = self.output.structures[0]  # TODO: Awaiting structures output, then inline this below
        struct = Atoms().from_hdf(self[self.min_job_name]['struct_0/output'])
        self.output.compositions = {
            symbol: self._species_fraction(struct, symbol)
            for symbol in self.output.symbols
        }

        self.to_hdf()

    def _species_fraction(self, structure, symbol):
        if symbol == '0':
            return len(structure) / self.output.n_sites
        else:
            return np.sum(structure.get_chemical_symbols() == symbol) / self.output.n_sites

    @_as_stats_array()
    def get_elastic_output(self, key):
        """
        Collects output from the underlying elastic job.

        Args:
            key (str):

        Returns:
            (StatsArray): A sub-element from the elastic job output.
        """
        try:
            return [
                self[self.elastic_job_name]['struct_{}/output/elasticmatrix'.format(n)][key]
                for n in np.arange(self.output.n_structures)
            ]
        except KeyError:
            raise KeyError("Tried to find {} in elastic output keys but it wasn't there. Please try one of "
                           "{}.".format(key, list(self[self.elastic_job_name]['struct_0/output/elasticmatrix'].keys())))


class SQSElasticOutput(InputList):
    """
    Powers up certain attributes as `StatsArray` taking the average over a specified axis.

    Attributes:
        axis (int): The axis over which to take statistical measures, i.e. the axis that runs over SQS structures.
            (Default is 0.)
    """
    # TODO: If InputList starts using the `to_hdf` and `from_hdf` methods of its children, we don't need to power up
    #  certain fields by wrapping them as StatsArrays, because the StatsArrays will be directly saved and loaded.
    #  At that point we could just directly use InputList (or IOList or whatever it's called by then) instead of
    #  subclassing like this
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        object.__setattr__(instance, "axis", 0)
        return instance

    def __init__(self, *args, axis=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis
        self._elastic_matrices = None
        self._residual_pressures = None
        self._cells = None

    @property
    def elastic_matrices(self):
        """Six-component elastic matrices"""
        return StatsArray(self._elastic_matrices, axis=self.axis)

    @elastic_matrices.setter
    def elastic_matrices(self, em):
        self._elastic_matrices = em

    @property
    def residual_pressures(self):
        """Pressures after minimization."""
        return StatsArray(self._residual_pressures, axis=self.axis)

    @residual_pressures.setter
    def residual_pressures(self, rp):
        self._residual_pressures = rp

    @property
    def cells(self):
        """Minimized SQS cells"""
        return StatsArray(self._cells, axis=self.axis)

    @cells.setter
    def cells(self, c):
        self._cells = c


class _SQSElasticConstantsGenerator(JobGenerator):
    """Generates jobs with different compositions."""

    @property
    def parameter_list(self):
        return self._job.input.compositions

    @staticmethod
    def job_name(parameter):
        return '_'.join(['{}{:.4}'.format(k, v).replace('.', '') for k, v in parameter.items()])

    def modify_job(self, job, parameter):
        job.ref_sqs.input.mole_fractions = parameter
        return job


class SQSElasticConstantsList(ParallelMaster):
    """
    Calculates the elastic constants for structures generated according to the special quasi-random structure scheme for
    a series of compositions.

    Attributes:
        ref_job (SQSElasticConstants): The job over which to iterate different compositions. (Must in turn have all its
            own reference jobs set.)
        input (InputList): Paremeters for controlling the run.
        output (SQSElasticOutput): Output collected from the various child jobs.

    Input:
        compositions (list): A list of dictionary items, each providing (key, value) pairs that are the chemical symbol
            and (approximate) fraction of the cell that should have that species. Following the standard of
            pyiron.atomistics.job.sqs.SQSJob, each dictionary should have fractions summing to 1 and '0' may be used as
            a key to indicate vacancies.

    Output:
        elastic_matrices (StatsArray): The six-component elastic matrices of the children.
        residual_pressures (StatsArray): The remaining pressures after minimization of the children.
        # structures (list): The minimized structures of the children.
        cells (StatsArray): The cells of the minimized structures of the children.
        symbols (list): The chemical symbols (possibly including '0' for vacancy) in the SQS structures of the children.
        compositions (dict): The actual chemical fraction of each species (including '0' for vacancy) in the structures
            of the children.

    Methods:
        get_elastic_constant_data: Get an elastic constant and chemical fractions for a particular species.
        get_elastic_constant_poly: Get a fit polynomial of an elastic constant for a particular species.
        get_C11/12/44_data: Get the C11, 12, or 44 elastic constant and species fraction for a particular species.
        get_C11/12/44_poly: Get the C11, 12, or 44 polynomial for a particular species.
        plot: Visualize the concentration-dependent elastic constants vs a particular species' concentration.
    """
    def __init__(self, project, job_name):
        super().__init__(project, job_name=job_name)
        self.__name__ = "SQSElasticConstantsList"
        self.__version__ = "0.1"
        self.__hdf_version__ = "0.2.0"

        self.input = InputList(table_name='input')
        self.input.compositions = []
        self.output = SQSElasticOutput(table_name='output', axis=1)

        self._job_generator = _SQSElasticConstantsGenerator(self)
        self._python_only_job = True

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf, group_name)
        self.input.to_hdf(self._hdf5)
        self.output.to_hdf(self._hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf, group_name)
        self.input.from_hdf(self._hdf5)
        self.output.from_hdf(self._hdf5)

    def collect_output(self):
        self.output._cells = self._collect_output_from_children('_cells')
        self.output._elastic_matrices = self._collect_output_from_children('_elastic_matrices')
        self.output._residual_pressures = self._collect_output_from_children('_residual_pressures')
        self.output.compositions = self._collect_output_from_children('compositions')
        self.output.n_sites = self._collect_output_from_children('n_sites')
        self.output.n_structures = self._collect_output_from_children('n_structures')
        self.output.symbols = self._collect_output_from_children('symbols')
        self.to_hdf()

    def _collect_output_from_children(self, key):
        return [job['output/data'][key] for job in self._ordered_children]

    @property
    def _ordered_children(self):
        return [self[self.child_names[n]] for n in np.sort(self.child_ids)]

    def _get_species_fraction(self, symbol):
        # return [composition[symbol] for composition in self.output.compositions]
        # TODO: InputList does not currently load in a friendly way, so look directly at the HDF
        return [composition[symbol] for composition in self['output/data']['compositions']]

    def _get_species_fraction_range(self, symbol):
        return (
            np.amin(self._get_species_fraction(symbol)),
            np.amax(self._get_species_fraction(symbol))
        )

    def _get_species_elastic_constant(self, symbol, indices):
        # return self.output.elastic_matrices.mean[:, indices[0], indices[1]]
        # TODO: InputList does not currently load in a friendly way, so look directly at the HDF
        return np.mean(self['output/data']['_elastic_matrices'], axis=1)[:, indices[0], indices[1]]

    def get_elastic_constant_data(self, symbol, indices):
        return (
            self._get_species_fraction(symbol),
            self._get_species_elastic_constant(symbol, indices)
        )

    def get_elastic_constant_poly(self, symbol, indices, deg=2):
        return np.poly1d(
            np.polyfit(
                *self.get_elastic_constant_data(symbol, indices),
                deg=deg
            )
        )

    def get_C11_poly(self, symbol, deg=2):
        return self.get_elastic_constant_poly(symbol, (0, 0), deg=deg)

    def get_C12_poly(self, symbol, deg=2):
        return self.get_elastic_constant_poly(symbol, (0, 1), deg=deg)

    def get_C44_poly(self, symbol, deg=2):
        return self.get_elastic_constant_poly(symbol, (3, 3), deg=deg)

    def get_C11_data(self, symbol):
        return self.get_elastic_constant_data(symbol, (0, 0))

    def get_C12_data(self, symbol):
        return self.get_elastic_constant_data(symbol, (0, 1))

    def get_C44_data(self, symbol):
        return self.get_elastic_constant_data(symbol, (3, 3))

    def plot(self, symbol, ax=None, deg=2):
        if ax is None:
            _, ax = plt.subplots()
        concentration = np.linspace(*self._get_species_fraction_range(symbol), num=1000)
        for label, color, data, poly in zip(
                ['C11', 'C12', 'C44'],
                sns.color_palette(n_colors=3),
                [
                    self.get_C11_data(symbol),
                    self.get_C12_data(symbol),
                    self.get_C44_data(symbol)
                ],
                [
                    self.get_C11_poly(symbol, deg=deg),
                    self.get_C12_poly(symbol, deg=deg),
                    self.get_C44_poly(symbol, deg=deg)
                ]
        ):
            ax.scatter(*data, label=label, color=color)
            ax.plot(concentration, poly(concentration), color=color)
        ax.legend()
        ax.set_xlabel(symbol + ' Atomic fraction')
        ax.set_ylabel('Elastic constants (GPa)')
        return ax
