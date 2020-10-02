# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base.master.generic import GenericMaster
from pyiron_base import InputList

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

    def run_static(self):
        self._run_sqs()
        self._run_minimization()
        self._run_elastic_list()

    def _run_sqs(self):
        sqs_job = self.project.create_job(self.project.job_type.SQSJob, self.job_name + '_sqs')
        sqs_job.structure = self.ref_job.structure.copy()
        sqs_job.input.mole_fractions = self.sqs_input.mole_fractions
        sqs_job.input.iterations = self.sqs_input.iterations
        sqs_job.input.weights = self.sqs_input.weights
        sqs_job.input.n_output_structures = self.sqs_input.n_output_structures
        sqs_job.run()
        self.project.wait_for_job(sqs_job)
        self.output_list.sqs_structures = sqs_job.list_structures()

    def _run_minimization(self):
        ref_min = self.ref_job.copy_to(
            new_job_name='_'.join([self.job_name, self.ref_job.job_name, 'min_ref']),
            new_database_entry=False
        )
        ref_min.calc_minimize(pressure=0)

        min_job = self.project.create_job(
            self.project.job_type.StructureListMaster,
            '_'.join([self.job_name, self.ref_job.job_name, 'minlist'])
        )
        min_job.ref_job = ref_min
        min_job.structure_lst = list(self.output_list.sqs_structures)
        min_job.run()
        self.project.wait_for_job(min_job)
        self.output_list.minimized_structures = [
            self.project.load(child_id).get_structure()
            for child_id in min_job.child_ids
        ]

    def _run_elastic_list(self):
        print("N minimized structures = {}".format(len(self.output_list.minimized_structures)))
        job_list = [
            self._run_elastic(str(n), structure)
            for n, structure in enumerate(self.output_list.minimized_structures)
        ]
        for job in job_list:
            self.project.wait_for_job(job)
        print(job_list)
        self.output_list.elastic_data = [job['output/elasticmatrix'] for job in job_list]

    def _run_elastic(self, id_, structure):
        # TODO: Parallelize over a list of structures, e.g. with StructureListMaster
        engine_job = self.ref_job.copy_to(
            new_job_name='_'.join([self.job_name, self.ref_job.job_name, 'engine_n' + id_]),
            new_database_entry=False
        )
        engine_job.structure = structure
        elastic_job = engine_job.create_job(
            self.project.job_type.ElasticMatrixJob,
            self.job_name + '_elastic_n' + id_
        )
        elastic_job.input.num_of_points = self.elastic_input.num_of_points
        elastic_job.input.fit_order = self.elastic_input.fit_order
        elastic_job.input.eps_range = self.elastic_input.eps_range
        elastic_job.input.relax_atoms = self.elastic_input.relax_atoms
        elastic_job.input.sqrt_eta = self.elastic_input.sqrt_eta
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
