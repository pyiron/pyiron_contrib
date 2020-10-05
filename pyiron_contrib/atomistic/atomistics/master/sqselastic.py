# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base import InputList, ParallelMaster, JobGenerator
from pyiron_base.master.flexible import FlexibleMaster
import numpy as np
# from pyiron import Atoms
from pyiron.atomistics.job.atomistic import AtomisticGenericJob
from pyiron.atomistics.job.sqs import SQSJob
from pyiron_mpie.interactive.elastic import ElasticMatrixJob
# import matplotlib.pyplot as plt
# import seaborn as sns

"""
Calculate the elastic matrix for SQS structure(s).
"""

__author__ = "Liam Huber"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Oct 2, 2020"


# These can't be methods, or FlexibleMaster throws an IndentationError when you try to load
def _sqs2minimization(sqs_job, min_job):
    min_job.structure_lst = sqs_job.list_structures()


def _minimization2elastic(min_job, elastic_job):
    elastic_job.structure_lst = [
        min_job['struct_{}'.format(int(i))].get_structure()
        for i in np.arange(len(min_job))
    ]


class SQSElasticConstants(FlexibleMaster):
    def __init__(self, project, job_name):
        super().__init__(project, job_name=job_name)
        self.__name__ = "SQSElasticConstants"
        self.__version__ = "0.1"
        self.__hdf_version__ = "0.2.0"

        self.ref_ham = None
        self.ref_sqs = None
        self.ref_elastic = None

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
        sqs_job = self._copy_job(self.ref_sqs, 'sqs_job')
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
        min_ref = self._copy_job(self.ref_ham, 'min_ref')
        min_ref.calc_minimize(pressure=0)
        # min_job = self.create_job(self._job_type.StructureListMaster, self._relative_name('min'))
        min_job = self._create_job(self._job_type.StructureListMaster, 'min')
        min_job.ref_job = min_ref
        return min_job

    def _copy_job(self, job, name):
        return job.copy_to(
            new_job_name='_'.join([self.job_name, name]),  # , job.job_name
            new_database_entry=False
        )

    def _instantiate_elastic(self):
        elastic_ref = self._copy_job(self.ref_ham, 'el_ref')

        # This should work, but doesn't maintain the elastic ref input and goes back to class defaults:
        # elastic_job = elastic_ref.create_job(self._job_type.ElasticMatrixJob, self._relative_name('el_job'))
        # return elastic_job.create_job(self._job_type.StructureListMaster, self._relative_name('el_list_job'))

        # elastic_job = self._copy_job(self.ref_elastic, 'el_job')  # ValueError: Unknown item: se_el_ham_ref

        # elastic_job = self.create_job(self._job_type.ElasticMatrixJob, self._relative_name('el_job'))
        elastic_job = self._create_job(self._job_type.ElasticMatrixJob, 'el_job')
        elastic_job.input = self.ref_elastic.input
        elastic_job.ref_job = elastic_ref
        elastic_job_list = self._create_job(self._job_type.StructureListMaster, 'el_list_job')
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
                or self.ref_elastic.ref_job.job_name != self._relative_name('el_ham_ref'):
            self.ref_elastic.ref_job = self._copy_job(self.ref_ham, 'el_ham_ref')
        if self.ref_elastic.job_name != self._relative_name('el_ref'):
            self.ref_elastic = self._copy_job(self.ref_elastic, 'el_ref')

        for job in [self.ref_ham, self.ref_sqs, self.ref_elastic]:
            self._save_reference_if_new(job)

    def _save_reference_if_new(self, job):
        if job.name not in self.project.list_nodes():
            job.save()
        else:
            job.to_hdf()


class _SQSElasticConstantsOutput:
    def __init__(self, parent):
        self.parent = parent


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
