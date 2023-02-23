from pyiron_atomistics.atomistics.job.sqs import SQSJob, get_sqs_structures


class SQSJobWithoutOutput(SQSJob):
    def __init__(self, project, job_name):
        super(SQSJobWithoutOutput, self).__init__(project, job_name)
        self._data_storage_disabled_implemented = True

    def run_static(self):
        if self.data_storage_enabled:
            super(SQSJobWithoutOutput, self).run_static()
        else:
            self._lst_of_struct, decmp, iterations, cycle_time = get_sqs_structures(
                structure=self.structure,
                mole_fractions={k: v for k, v in self.input.mole_fractions.items()},
                weights=self.input.weights,
                objective=self.input.objective,
                iterations=self.input.iterations,
                output_structures=self.input.n_output_structures,
                num_threads=self.server.cores,
            )
