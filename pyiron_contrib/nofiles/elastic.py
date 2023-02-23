from pyiron_base import GenericMaster
from pyiron_gpl.elastic.elastic import ElasticMatrixJob


class ElasticMatrixJobWithoutFiles(ElasticMatrixJob):
    def __init__(self, project, job_name):
        super(ElasticMatrixJobWithoutFiles, self).__init__(project, job_name)
        self._data_storage_disabled_implemented = True

    def collect_output(self):
        if not self._data:
            self.from_hdf()
        self.create_calculator()

        energies = {}
        self._data["id"] = []
        if self.server.run_mode.interactive and self.data_storage_enabled:
            child_id = self.child_ids[0]
            self._data["id"].append(child_id)
            child_job = self.project_hdf5.inspect(child_id)
            energies = {job_name: energy for job_name, energy in zip(self.structure_dict.keys(),
                                                                     child_job["output/generic/energy_tot"])}
        elif self.server.run_mode.interactive and not self.data_storage_enabled:
            energies = {job_name: energy for job_name, energy in zip(self.structure_dict.keys(),
                                                                     self.ref_job.interactive_cache["energy_tot"])}
        else:
            for job_id in self.child_ids:
                ham = self.project_hdf5.inspect(job_id)
                en = ham["output/generic/energy_tot"][-1]
                energies[ham.job_name] = en
                self._data["id"].append(ham.job_id)

        self.property_calculator.analyse_structures(energies)
        self._data.update(self.property_calculator._data)
        if self.data_storage_enabled:
            self.to_hdf()

    def append(self, job):
        """
        Append a job to the GenericMaster - just like you would append an element to a list.

        Args:
            job (GenericJob): job to append
        """
        if self.status.initialized and not job.status.initialized:
            raise ValueError(
                "GenericMaster requires reference jobs to have status initialized, rather than ",
                job.status.string,
            )
        if job.server.cores >= self.server.cores:
            self.server.cores = job.server.cores
        if job.job_name not in self._job_name_lst:
            self._job_name_lst.append(job.job_name)
            if self.data_storage_enabled:
                self._child_job_update_hdf(parent_job=self, child_job=job)

    def pop(self, i=-1):
        """
        Pop a job from the GenericMaster - just like you would pop an element from a list

        Args:
            i (int): position of the job. (Default is last element, -1.)

        Returns:
            GenericJob: job
        """
        job_name_to_return = self._job_name_lst[i]
        job_to_return = self._load_all_child_jobs(
            self._load_job_from_cache(job_name_to_return)
        )
        del self._job_name_lst[i]
        if self.data_storage_enabled:
            with self.project_hdf5.open("input") as hdf5_input:
                hdf5_input["job_list"] = self._job_name_lst
            job_to_return.relocate_hdf5()
        if isinstance(job_to_return, GenericMaster):
            for sub_job in job_to_return._job_object_dict.values():
                self._child_job_update_hdf(parent_job=job_to_return, child_job=sub_job)
        job_to_return.status.initialized = True
        return job_to_return

    def _run_if_collect(self):
        """
        Internal helper function the run if collect function is called when the job status is 'collect'. It collects
        the simulation output using the standardized functions collect_output() and collect_logfiles(). Afterwards the
        status is set to 'finished'.
        """

        if self.data_storage_enabled:
            super(ElasticMatrixJobWithoutFiles)._run_if_collect()
        else:
            self._logger.info(
                "{}, status: {}, finished".format(self.job_info_str, self.status)
            )
            self.collect_output()
            self._logger.info(
                "{}, status: {}, parallel master".format(self.job_info_str, self.status)
            )
            self.update_master()
            # self.send_to_database()