import pandas
import numpy as np
from pyiron_base import GenericMaster
from pyiron_atomistics.atomistics.master.murnaghan import Murnaghan


class MurnaghanWithoutFiles(Murnaghan):
    @property
    def child_project(self):
        """
        :class:`.Project`: project which holds the created child jobs
        """
        if self.data_storage_enabled:
            return super(MurnaghanWithoutFiles, self).child_project
        else:
            return self.project

    def _fit_eos_general(self, vol_erg_dic=None, fittype="birchmurnaghan"):
        self._set_fit_module(vol_erg_dic=vol_erg_dic)
        fit_dict = self.fit_module.fit_eos_general(fittype=fittype)
        self.input["fit_type"] = fit_dict["fit_type"]
        self.input["fit_order"] = 0
        if self.data_storage_enabled:
            self._store_fit_in_hdf(fit_dict=fit_dict)
        self.fit_dict = fit_dict
        return fit_dict

    def poly_fit(self, fit_order=3, vol_erg_dic=None):
        self._set_fit_module(vol_erg_dic=vol_erg_dic)
        fit_dict = self.fit_module.fit_polynomial(fit_order=fit_order)
        if fit_dict is None:
            self._logger.warning("Minimum could not be found!")
        elif self.data_storage_enabled:
            self.input["fit_type"] = fit_dict["fit_type"]
            self.input["fit_order"] = fit_dict["fit_order"]
            if not self._interactive_disable_log_file:
                self._store_fit_in_hdf(fit_dict=fit_dict)
        self.fit_dict = fit_dict
        return fit_dict

    def collect_output(self):
        if self.data_storage_enabled:
            super(MurnaghanWithoutFiles).collect_output()
        elif self.ref_job.server.run_mode.interactive:
            erg_lst = self.ref_job.output.energy_pot.copy()
            vol_lst = self.ref_job.output.volume.copy()
            arg_lst = np.argsort(vol_lst)

            self._output["volume"] = vol_lst[arg_lst]
            self._output["energy"] = erg_lst[arg_lst]
            if self.input["fit_type"] == "polynomial":
                self.fit_polynomial(fit_order=self.input["fit_order"])
            else:
                self._fit_eos_general(fittype=self.input["fit_type"])
        else:
            raise ValueError("No files execution requires interactive jobs.")

    def _run_if_collect(self):
        """
        Internal helper function the run if collect function is called when the job status is 'collect'. It collects
        the simulation output using the standardized functions collect_output() and collect_logfiles(). Afterwards the
        status is set to 'finished'.
        """

        if self.data_storage_enabled:
            super(MurnaghanWithoutFiles)._run_if_collect()
        else:
            self._logger.info(
                "{}, status: {}, finished".format(self.job_info_str, self.status)
            )
            self.collect_output()
            self._logger.info(
                "{}, status: {}, parallel master".format(self.job_info_str, self.status)
            )
            self.update_master()
            self.status.finished = True
            # self.send_to_database()

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

    def output_to_pandas(self, sort_by=None, h5_path="output"):
        """
        Convert output of all child jobs to a pandas Dataframe object.

        Args:
            sort_by (str): sort the output using pandas.DataFrame.sort_values(by=sort_by)
            h5_path (str): select child output to include - default='output'

        Returns:
            pandas.Dataframe: output as dataframe
        """
        # TODO: The output to pandas function should no longer be required
        if self.data_storage_enabled:
            super(MurnaghanWithoutFiles, self).output_to_pandas(sort_by=sort_by, h5_path=h5_path)
        else:
            df = pandas.DataFrame(self._output)
            if sort_by is not None:
                df = df.sort_values(by=sort_by)
            return df