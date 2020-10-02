# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base.master.generic import GenericMaster
from pyiron_base import InputList
import numpy as np

"""
Calculate the elastic matrix for SQS structure(s).

TODO:
    - Add an extra layer to loop over SQS concentration ranges
    - Test and implement server stuff so it can exploit parallelism
"""

__author__ = "Liam Huber"
__copyright__ = "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Oct 2, 2020"


class SQSElasticConstants(GenericMaster):

    def __init__(self, project, job_name):
        super().__init__(project, job_name=job_name)
        self.__name__ = "SQSElasticConstants"
        self.__version__ = "0.1"

        self.ref_job = None

        self.sqs_input = InputList(table_name='sqs_input')
        self.sqs_input.mole_fractions = None
        self.sqs_input.iterations = 1e6
        self.sqs_input.weights = None
        self.sqs_input.n_output_structures = 1

        self.elastic_input = InputList(table_name='elastic_input')
        self.elastic_input.num_of_points = 5
        self.elastic_input.fit_order = 2
        self.elastic_input.eps_range = 0.005
        self.elastic_input.relax_atoms = 1
        self.elastic_input.sqrt_eta = True

        self.output_list = InputList(table_name='output_list')
        self.output_list.elastic_matrix = None

    def _relative_name(self, name):
        return '_'.join([self.job_name, name])

    def _create_job(self, job_type, job_name):
        return self.project.create_job(job_type, self._relative_name(job_name))

    @property
    def _job_type(self):
        return self.project.job_type

    @staticmethod
    def _apply_inputlist(target, source):
        for k, v in source.items():
            target[k] = v

    def _wait(self, job_or_jobs):
        try:
            self.project.wait_for_job(job_or_jobs)
        except AttributeError:
            for job in job_or_jobs:
                self._wait(job)

    def _copy_ref_job(self, name):
        return self.ref_job.copy_to(new_job_name=self._relative_name(name), new_database_entry=False)

    def _std_to_sqs_sem(self, array):
        return array / np.sqrt(self.sqs_input.n_output_structures)

    def run_static(self):
        self._run_sqs()
        self._run_minimization()
        self._run_elastic_list()

    def _run_sqs(self):
        sqs_job = self._create_job(self._job_type.SQSJob, 'sqs')
        sqs_job.structure = self.ref_job.structure.copy()
        self._apply_inputlist(sqs_job.input, self.sqs_input)
        sqs_job.run()
        self._wait(sqs_job)
        self.output_list.sqs_structures = sqs_job.list_structures()
        # TODO: Grab the corrected Mole fractions

    def _run_minimization(self):
        ref_min = self._copy_ref_job('minref')
        ref_min.calc_minimize(pressure=0)

        min_job = self._create_job(self._job_type.StructureListMaster, 'minlist')
        min_job.ref_job = ref_min
        min_job.structure_lst = list(self.output_list.sqs_structures)
        min_job.run()
        self._wait(min_job)
        self.output_list.structures = [
            self.project.load(child_id).get_structure()
            for child_id in min_job.child_ids
        ]
        cells = np.array([structure.cell.array for structure in self.output_list.structures])
        self.output_list.cell_mean = cells.mean(axis=0)
        self.output_list.cell_std = cells.std(axis=0)
        self.output_list.cell_sem = self._std_to_sqs_sem(self.output_list.cell_std)

    def _run_elastic_list(self):
        job_list = [self._run_elastic(str(n), structure) for n, structure in enumerate(self.output_list.structures)]
        self._wait(job_list)
        self.output_list.elastic_data = [job['output/elasticmatrix'] for job in job_list]
        elastic_matrices = np.array([job.output_list.elastic_data['C'] for job in job_list])
        self.output_list.elastic_matrix_mean = elastic_matrices.mean(axis=0)
        self.output_list.elastic_matrix_std = elastic_matrices.std(axis=0)
        self.output_list.elastic_matrix_sem = self._std_to_sqs_sem(self.output_list.elastic_matrix_std)

    def _run_elastic(self, id_, structure):
        engine_job = self._copy_ref_job('engine_n{}'.format(id_))
        engine_job.structure = structure
        elastic_job = engine_job.create_job(
            self._job_type.ElasticMatrixJob,
            self._relative_name('elastic_n{}'.format(id_))
        )
        self._apply_inputlist(elastic_job.input, self.elastic_input)
        elastic_job.run()
        return elastic_job

    def collect_output(self):
        print("Collecting output")

    def to_hdf(self, hdf=None, group_name=None):
        self.ref_job.save()
        self._hdf5['refjobname'] = self.ref_job.job_name
        self.sqs_input.to_hdf(self._hdf5, group_name)
        self.elastic_input.to_hdf(self._hdf5, group_name)
        self.output_list.to_hdf(self._hdf5, group_name)
        super().to_hdf(hdf, group_name)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf, group_name)
        self.ref_job = self.project.load(self._hdf5['refjobname'])
        self.sqs_input.from_hdf(self._hdf5)
        self.elastic_input.from_hdf(self._hdf5)
        self.output_list.from_hdf(self._hdf5)

    def write_input(self):
        pass  # We need to reconsider our inheritance scheme
