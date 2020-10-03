# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base import InputList, GenericJob
import numpy as np
from pyiron import Atoms
from pyiron.atomistics.job.atomistic import AtomisticGenericJob
from pyiron.atomistics.job.sqs import SQSJob
from pyiron_mpie.interactive.elastic import ElasticMatrixJob

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


class SQSElasticConstants(GenericJob):

    def __init__(self, project, job_name):
        super().__init__(project, job_name=job_name)
        self.__name__ = "SQSElasticConstants"
        self.__version__ = "0.1"

        self.ref_job = None
        self.ref_sqs = None
        self.ref_elastic = None

        self.output = _SQSElasticConstantsOutput(self)

        self._wait_interval_in_s = 1
        self._wait_max_iterations = 3600

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
            self.project.wait_for_job(
                job_or_jobs,
                interval_in_s=self._wait_interval_in_s,
                max_iterations=self._wait_max_iterations
            )
        except AttributeError:
            for job in job_or_jobs:
                self._wait(job)

    def _copy_ref(self, job, name):
        return job.copy_to(new_job_name=self._relative_name(name), new_database_entry=False)

    def _copy_ref_job(self, name):
        return self._copy_ref(self.ref_job, name)

    def _copy_ref_sqs(self):
        job = self._copy_ref(self.ref_sqs, 'sqs')
        job.input = self.ref_sqs.input
        # This is an ugly hack -- somehow the input mole_fractions dict otherwise gets converted to an InputList of
        # that dict, which the underlying SQSJob is not happy about.
        return job

    def _copy_ref_elastic(self, name):
        # This isn't actually used because of (what I consider to be) an oddity of ElasticMatrixJob and it's ref_job
        return self._copy_ref(self.ref_elastic, name)
        # job.input = self.ref_elastic.input
        # return job

    def _std_to_sqs_sem(self, array):
        return array / np.sqrt(self.ref_sqs.input.n_output_structures)

    def _store_statistics_over_sqs(self, storage_object, data):
        """
        Given some numpy array data running over different SQS structures, take statistics over the structure-axis and
        store it in place.

        Args:
            storage_object (pyiron_base.InputList): Where to store the output.
            data (numpy.ndarray): The data to take statistics of, assuming the 0th axis runs over SQS structures.
        """
        storage_object.array = data
        storage_object.mean = data.mean(axis=0)
        storage_object.std = data.std(axis=0)
        storage_object.sem = self._std_to_sqs_sem(storage_object.std)

    def run_static(self):
        self._run_sqs()
        self._run_minimization()
        self._run_elastic_list()
        self.to_hdf()

    def _run_sqs(self):
        sqs_job = self._copy_ref_sqs()
        sqs_job.structure = self.ref_job.structure.copy()
        self.sqs_job = sqs_job
        sqs_job.run()
        self._wait(sqs_job)
        self.output.sqs_structures = sqs_job.list_structures()
        # TODO: Grab the corrected Mole fractions

    def _run_minimization(self):
        ref_min = self._copy_ref_job('minref')
        ref_min.calc_minimize(pressure=0)

        min_job = self._create_job(self._job_type.StructureListMaster, 'minlist')
        min_job.ref_job = ref_min
        min_job.structure_lst = list(self.output.sqs_structures)
        min_job.run()
        self._wait(min_job)
        self.output.structures = [
            self.project.load(child_id).get_structure()
            for child_id in min_job.child_ids
        ]
        self._store_statistics_over_sqs(
            self.output.cell,
            np.array([structure.cell.array for structure in self.output.structures])
        )

    def _run_elastic_list(self):
        job_list = [self._run_elastic(str(n), structure) for n, structure in enumerate(self.output.structures)]
        self._wait(job_list)
        self._store_statistics_over_sqs(
            self.output.elastic_matrix,
            np.array([job['output/elasticmatrix']['C'] for job in job_list])
        )
        # TODO: Save more of the elsaticmatrix output as it becomes clear from usage that it's something you need
        #       or just save all of it once output is a class that can save structures as easily as other objects!

    def _run_elastic(self, id_, structure):
        engine_job = self._copy_ref_job('engine_n{}'.format(id_))
        engine_job.structure = structure

        elastic_job = engine_job.create_job(self._job_type.ElasticMatrixJob, 'elastic_n{}'.format(id_))
        elastic_job.input = self.ref_elastic.input
        # TODO: Figure out what is up with ElasticMatrixJob that this doesn't work instead:
        #     elastic_job = self._copy_ref_job('elastic_n{}'.format(id_))
        #     elastic_job.ref_job = engine_job
        elastic_job.run()
        return elastic_job

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf, group_name)
        self.output.to_hdf()

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf, group_name)
        self.output.from_hdf()

    def validate_ready_to_run(self):
        for attribute_name, class_ in zip(
            ['ref_job', 'ref_sqs', 'ref_elastic'],
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
        super().validate_ready_to_run()

    def collect_output(self):
        pass

    def write_input(self):
        pass  # We need to reconsider our inheritance scheme

    def run_if_interactive(self):
        raise NotImplementedError("Interactive running is not configured for {}".format(self.__name__))

    def run_if_refresh(self):
        raise NotImplementedError("Refreshed running is not configured for {}".format(self.__name__))



class _SQSElasticConstantsOutput:
    def __init__(self, parent):
        self.parent = parent

        self.sqs_structures = []
        self.structures = []

        self.cell = InputList(table_name='output/cell')
        self.cell.array = None
        self.cell.mean = None
        self.cell.std = None
        self.cell.sem = None

        self.elastic_matrix = InputList(table_name='output/elastic_matrix')
        self.elastic_matrix.array = None
        self.elastic_matrix.mean = None
        self.elastic_matrix.std = None
        self.elastic_matrix.sem = None

        self._hdf5 = self.parent._hdf5

    def to_hdf(self):
        with self._hdf5.open('output/sqs_structures') as hdf5_server:
            for n, structure in enumerate(self.sqs_structures):
                structure.to_hdf(hdf5_server, group_name='structure{}'.format(n))
        with self._hdf5.open('output/structures') as hdf5_server:
            for n, structure in enumerate(self.structures):
                structure.to_hdf(hdf5_server, group_name='structure{}'.format(n))
        self.cell.to_hdf(self._hdf5)
        self.elastic_matrix.to_hdf(self._hdf5)

    def from_hdf(self):
        with self._hdf5.open('output/sqs_structures') as hdf5_server:
            for group in hdf5_server.list_groups():
                self.sqs_structures.append(Atoms().from_hdf(hdf5_server, group_name=group))
        with self._hdf5.open('output/structures') as hdf5_server:
            for group in hdf5_server.list_groups():
                self.structures.append(Atoms().from_hdf(hdf5_server, group_name=group))
        self.cell.from_hdf(self._hdf5)
        self.elastic_matrix.from_hdf(self._hdf5)
