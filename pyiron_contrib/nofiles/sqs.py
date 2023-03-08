from pyiron_atomistics.atomistics.job.sqs import SQSJob, get_sqs_structures
from pyiron_base import state


class SQSJobWithoutOutput(SQSJob):
    def __init__(self, project, job_name):
        if not state.database.database_is_disabled:
            raise RuntimeError(
                "To run a `Without` job, the database must first be disabled. Please "
                "`from pyiron_base import state; "
                "state.update({'disable_database': True})`, and try again."
            )
        super(SQSJobWithoutOutput, self).__init__(project, job_name)
        self._interactive_disable_log_file = True

    def to_hdf(self, hdf=None, group_name=None):
        """

        Args:
            hdf:
            group_name:

        Returns:

        """
        if not self._interactive_disable_log_file:
            super(SQSJobWithoutOutput, self).to_hdf(hdf=hdf, group_name=group_name)

    def run_static(self):
        if not self._interactive_disable_log_file:
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

    def refresh_job_status(self):
        if not self._interactive_disable_log_file:
            super(SQSJobWithoutOutput, self).refresh_job_status()
