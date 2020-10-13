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
# import matplotlib.pyplot as plt
# import seaborn as sns
from functools import lru_cache

"""
Calculate the elastic matrix for special quasi-random structures.
"""

__author__ = "Liam Huber"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Oct 2, 2020"


# This can't be a method, or FlexibleMaster throws an IndentationError when you try to load
def _sqs2minimization(sqs_job, min_job):
    min_job.structure_lst = sqs_job.list_structures()


# This can't be a method, or FlexibleMaster throws an IndentationError when you try to load
def _minimization2elastic(min_job, elastic_job):
    elastic_job.structure_lst = [
        min_job['struct_{}'.format(int(i))].get_structure()
        for i in np.arange(len(min_job))
    ]


class SQSElasticConstants(FlexibleMaster):
    """
    Calculates the elastic constants for structures generated according to the special quasi-random structure scheme.

    This job itself takes no input, but rather the input is passed in via the reference jobs, e.g. `ref_ham` handles
    all input for the force/energy evaluations (e.g. emperical potential if a classical interpreter, kpoints, energy
    cutoff, etc. for quantum interpreters...), `ref_sqs` handles the SQS input like the molar fractions and how many
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

    Output:
        elastic_matrix (ChemicalArray): The six-component elastic matrix.
        residual_pressures (ChemicalArray): The remaining pressure after minimization.
        structures (list): The minimized structures.
        cells (ChemicalArray): The cells of the minimized structures.
        symbols (list): The chemical symbols (possibly including '0' for vacancy) in the SQS structures.
        chemistry (dict): The actual chemical fraction of each species (including '0' for vacancy) in the structures.
        get_elastic_output (fnc): A function taking a string key for accessing other output from the ElasticMatrixJob
            as a ChemicalArray. (If you ask for an invalid key, all the valid possibilities will be printed for you.)

    Note: The `ChemicalArray` class is a wrapper for numpy arrays that allows you to look at the `.array`, `.mean`,
        `.std` and `.sem` (standard error) with respect to the different SQS structures.

    Warning: Initial relaxation is only isotropic with respect to cell shape.
    """
    def __init__(self, project, job_name):
        super().__init__(project, job_name=job_name)
        self.__name__ = "SQSElasticConstants"
        self.__version__ = "0.1"
        self.__hdf_version__ = "0.2.0"

        self.ref_ham = None
        self.ref_sqs = None
        self.ref_elastic = None

        self._sqs_job_name_tail = 'sqs'
        self._min_job_name_tail = 'min'
        self._min_ref_name_tail = 'min_ref'
        self._elastic_ref_ham_name_tail = 'el_ref'
        self._elastic_ref_single_name_tail = 'el_job'
        self._elastic_job_name_tail = 'elastic'

        self.sqs_job_name = self._relative_name('sqs')
        self.min_job_name = self._relative_name('min')
        self.elastic_job_name = self._relative_name('elastic')

        self.output = _SQSElasticConstantsOutput(self)

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
        sqs_job = self._copy_job(self.ref_sqs, self._sqs_job_name_tail)
        sqs_job.input.mole_fractions = dict(sqs_job.input.mole_fractions)  # Input expects dict but gets InputList(dict)
        sqs_job.structure = self.ref_ham.structure.copy()
        return sqs_job

    def _create_job(self, job_type, name):
        return self.project.create_job(job_type, self._relative_name(name))

    def _relative_name(self, name):
        return '_'.join([self.job_name, name])

    @property
    def _job_type(self):
        return self.project.job_type

    def _instantiate_minimization(self):
        min_ref = self._copy_job(self.ref_ham, self._min_ref_name_tail)
        min_ref.calc_minimize(pressure=0)
        # min_job = self.create_job(self._job_type.StructureListMaster, self._relative_name('min'))
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

        # This should work, but doesn't maintain the elastic ref input and goes back to class defaults:
        # elastic_job = elastic_ref.create_job(self._job_type.ElasticMatrixJob, self._relative_name('el_job'))
        # return elastic_job.create_job(self._job_type.StructureListMaster, self._relative_name('el_list_job'))

        # elastic_job = self._copy_job(self.ref_elastic, 'el_job')  # ValueError: Unknown item: se_el_ham_ref

        # elastic_job = self.create_job(self._job_type.ElasticMatrixJob, self._relative_name('el_job'))
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

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf, group_name)
        self.ref_ham = self.project.load(self._hdf5['refjobs/refhamname'])
        self.ref_sqs = self.project.load(self._hdf5['refjobs/refsqsname'])
        self.ref_elastic = self.project.load(self._hdf5['refjobs/refelasticname'])

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
        if self.ref_ham.job_name != self._relative_name('ham_ref'):
            self.ref_ham = self._copy_job(self.ref_ham, 'ham_ref')

        if self.ref_sqs.job_name != self._relative_name('sqs_ref'):
            self.ref_sqs = self._copy_job(self.ref_sqs, 'sqs_ref')

        if self.ref_elastic.ref_job is None \
                or self.ref_elastic.ref_job.job_name != self._relative_name(self._elastic_ref_single_name_tail):
            self.ref_elastic.ref_job = self._copy_job(self.ref_ham, self._elastic_ref_single_name_tail)
        if self.ref_elastic.job_name != self._relative_name('el_ref'):
            self.ref_elastic = self._copy_job(self.ref_elastic, 'el_ref')

        for job in [self.ref_ham, self.ref_sqs, self.ref_elastic]:
            self._save_reference_if_new(job)

    def _save_reference_if_new(self, job):
        if job.name not in self.project.list_nodes():
            job.save()
        else:
            job.to_hdf()


class ChemicalArray:
    """
    A convenience class for wrapping arrays of SQS elastic data where the zeroth dimension spans across chemistry.
    """

    def __init__(self, data):
        self._data = np.array(data)

    @property
    def array(self):
        return self._data

    @property
    def mean(self):
        return self._data.mean(axis=0)

    @property
    def std(self):
        return self._data.std(axis=0)

    @property
    def sem(self):
        return self._data.std(axis=0) / len(self._data)

    def __repr__(self):
        return str(self._data)


def _as_chemical_array(fnc):
    def decorated(*args, **kwargs):
        return ChemicalArray(fnc(*args, **kwargs))

    return decorated


class _SQSElasticConstantsOutput:
    def __init__(self, parent):
        self.parent = parent

    def get_elastic_output(self, key):
        elastic_hdf = self.parent[self.parent.elastic_job_name]
        n_structures = len(elastic_hdf['input/structures'].list_groups())
        try:
            return [
                elastic_hdf['struct_{}/output/elasticmatrix'.format(n)][key]
                for n in np.arange(n_structures)
            ]
        except KeyError:
            raise KeyError("Tried to find {} in elastic output keys but it wasn't there. Please try one of "
                           "{}.".format(key, list(elastic_hdf['struct_0/output/elasticmatrix'].keys())))

    @property
    @_as_chemical_array
    @lru_cache()
    def elastic_matrix(self):
        """Six-component elastic matrix."""
        return self.get_elastic_output('C')

    @property
    @_as_chemical_array
    @lru_cache()
    def residual_pressures(self):
        """Pressures after minimization."""
        min_hdf = self.parent[self.parent.min_job_name]
        n_structures = len(min_hdf['input/structures'].list_groups())
        return [
            min_hdf['struct_{}/output/generic/pressures'.format(n)][-1]
            for n in np.arange(n_structures)
        ]

    @property
    @lru_cache()
    def structures(self):
        min_hdf = self.parent[self.parent.min_job_name]
        n_structures = len(min_hdf['input/structures'].list_groups())
        return [
            Atoms().from_hdf(min_hdf['struct_{}/output'.format(n)])
            for n in np.arange(n_structures)
        ]

    @property
    @_as_chemical_array
    def cells(self):
        return [structure.cell.array for structure in self.structures]

    @property
    @lru_cache()
    def symbols(self):
        return list(self.parent[self.parent.sqs_job_name]['input/custom_dict/data']['mole_fractions'].keys())

    @property
    def chemistry(self):
        n_atoms = len(self.parent[self.parent.min_job_name]['input/structures/s_0/positions'])
        return {
            symbol: self._species_fraction(structure, symbol, n_atoms)
            for symbol, structure in zip(self.symbols, self.structures)
        }

    @staticmethod
    def _species_fraction(structure, symbol, reference_count):
        if symbol == '0':
            return len(structure) / reference_count
        else:
            return np.sum(structure.get_chemical_symbols() == symbol) / reference_count


class _SQSElasticConstantsGenerator(JobGenerator):
    @property
    def parameter_list(self):
        return self._job.input.chemistry

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

    Input:
        chemistry (list): A list of dictionary items, each providing (key, value) pairs that are the chemical symbol and
            (approximate) fraction of the cell that should have that species. Following the standard of
            pyiron.atomistics.job.sqs.SQSJob, each dictionary should have fractions summing to 1 and '0' may be used as
            a key to indicate vacancies.

    Output:

    """
    def __init__(self, project, job_name):
        super().__init__(project, job_name=job_name)
        self.__name__ = "SQSElasticConstantsList"
        self.__version__ = "0.1"
        self.__hdf_version__ = "0.2.0"

        self.input = InputList(table_name='input')
        self.input.chemistry = []

        self._job_generator = _SQSElasticConstantsGenerator(self)
        self._python_only_job = True

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf, group_name)
        self.input.to_hdf(self._hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf, group_name)
        self.input.from_hdf(self._hdf5)

    def collect_output(self):
        pass

# Snippets for later:

    # def _collect_actual_chemistry(self, job):
    #     structure = job.output.structures[0]
    #     n0 = len(self.ref_ham.structure)
    #     actual_chemistry = {}
    #     for symbol in structure.get_species_symbols():
    #         actual_chemistry[symbol] = np.sum(structure.get_chemical_symbols() == symbol) / n0
    #     if len(structure) < n0:
    #         actual_chemistry['0'] = (n0 - len(structure)) / n0
    #     self.output.chemistry.append(actual_chemistry)
    #
    # def get_elastic_constant(self, indices, chemical_symbol):
    #     concentration = [chemistry[chemical_symbol] for chemistry in self.output.chemistry]
    #     elastic_constant = [matrix[indices] for matrix in self.output.elastic_matrices]
    #     return np.array(concentration), np.array(elastic_constant)
    #
    # def plot_elastic_constant(self, indices, chemical_symbol):
    #     fig, ax = plt.subplots()
    #     ax.scatter(*self.get_elastic_constant(indices, chemical_symbol))
    #     return fig, ax


# class _SQSElasticConstantsOutput:
#     def __init__(self, parent):
#         self.parent = parent
#
#         self.sqs_structures = []
#         self.structures = []
#
#         self.cell = InputList(table_name='output/cell')
#         self.cell.array = None
#         self.cell.mean = None
#         self.cell.std = None
#         self.cell.sem = None
#
#         self.elastic_matrix = InputList(table_name='output/elastic_matrix')
#         self.elastic_matrix.array = None
#         self.elastic_matrix.mean = None
#         self.elastic_matrix.std = None
#         self.elastic_matrix.sem = None
#
#         self._hdf5 = self.parent._hdf5
#
#     def to_hdf(self):
#         with self._hdf5.open('output/sqs_structures') as hdf5_server:
#             for n, structure in enumerate(self.sqs_structures):
#                 structure.to_hdf(hdf5_server, group_name='structure{}'.format(n))
#         with self._hdf5.open('output/structures') as hdf5_server:
#             for n, structure in enumerate(self.structures):
#                 structure.to_hdf(hdf5_server, group_name='structure{}'.format(n))
#         self.cell.to_hdf(self._hdf5)
#         self.elastic_matrix.to_hdf(self._hdf5)
#
#     def from_hdf(self):
#         with self._hdf5.open('output/sqs_structures') as hdf5_server:
#             for group in hdf5_server.list_groups():
#                 self.sqs_structures.append(Atoms().from_hdf(hdf5_server, group_name=group))
#         with self._hdf5.open('output/structures') as hdf5_server:
#             for group in hdf5_server.list_groups():
#                 self.structures.append(Atoms().from_hdf(hdf5_server, group_name=group))
#         self.cell.from_hdf(self._hdf5)
#         self.elastic_matrix.from_hdf(self._hdf5)
