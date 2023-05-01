from pyiron_base import GenericMaster, state
from pyiron_atomistics.atomistics.master.phonopy import PhonopyJob


class PhonopyJobWithoutFiles(PhonopyJob):
    def __init__(self, project, job_name):
        if not state.database.database_is_disabled:
            raise RuntimeError(
                "To run a `Without` job, the database must first be disabled. Please "
                "`from pyiron_base import state; "
                "state.update({'disable_database': True})`, and try again."
            )
        super(PhonopyJobWithoutFiles, self).__init__(project, job_name)
        self._interactive_disable_log_file = True

    @property
    def child_project(self):
        """
        :class:`.Project`: project which holds the created child jobs
        """
        if not self._interactive_disable_log_file:
            return super(PhonopyJobWithoutFiles, self).child_project
        else:
            return self.project

    def to_hdf(self, hdf=None, group_name=None):
        """
        Args:
            hdf:
            group_name:
        Returns:
        """
        if not self._interactive_disable_log_file:
            super(PhonopyJobWithoutFiles, self).to_hdf(hdf=hdf, group_name=group_name)

    def refresh_job_status(self):
        if not self._interactive_disable_log_file:
            super(PhonopyJobWithoutFiles).refresh_job_status()

    def _run_if_collect(self):
        """
        Internal helper function the run if collect function is called when the job status is 'collect'. It collects
        the simulation output using the standardized functions collect_output() and collect_logfiles(). Afterwards the
        status is set to 'finished'.
        """

        if not self._interactive_disable_log_file:
            super(PhonopyJobWithoutFiles)._run_if_collect()
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

    def collect_output(self):
        """
        Returns:
        """
        if (
            self.ref_job.server.run_mode.interactive
            and self._interactive_disable_log_file
        ):
            forces_lst = self.ref_job.output.forces
        elif self.ref_job.server.run_mode.interactive:
            forces_lst = self.project_hdf5.inspect(self.child_ids[0])[
                "output/generic/forces"
            ]
        else:
            pr_job = self.project_hdf5.project.open(self.job_name + "_hdf5")
            forces_lst = [
                pr_job.inspect(job_name)["output/generic/forces"][-1]
                for job_name in self._get_jobs_sorted()
            ]
        self.phonopy.set_forces(forces_lst)
        self.phonopy.produce_force_constants(
            fc_calculator=None if self.input["number_of_snapshots"] is None else "alm"
        )
        self.phonopy.run_mesh(mesh=[self.input["dos_mesh"]] * 3)
        mesh_dict = self.phonopy.get_mesh_dict()
        self.phonopy.run_total_dos()
        dos_dict = self.phonopy.get_total_dos_dict()

        if not self._interactive_disable_log_file:
            self.to_hdf()
            with self.project_hdf5.open("output") as hdf5_out:
                hdf5_out["dos_total"] = dos_dict["total_dos"]
                hdf5_out["dos_energies"] = dos_dict["frequency_points"]
                hdf5_out["qpoints"] = mesh_dict["qpoints"]
                hdf5_out["supercell_matrix"] = self._phonopy_supercell_matrix()
                hdf5_out[
                    "displacement_dataset"
                ] = self.phonopy.get_displacement_dataset()
                hdf5_out[
                    "dynamical_matrix"
                ] = self.phonopy.dynamical_matrix.get_dynamical_matrix()
                hdf5_out["force_constants"] = self.phonopy.force_constants

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
            if not self._interactive_disable_log_file:
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
        if not self._interactive_disable_log_file:
            with self.project_hdf5.open("input") as hdf5_input:
                hdf5_input["job_list"] = self._job_name_lst
            job_to_return.relocate_hdf5()
        if isinstance(job_to_return, GenericMaster):
            for sub_job in job_to_return._job_object_dict.values():
                self._child_job_update_hdf(parent_job=job_to_return, child_job=sub_job)
        job_to_return.status.initialized = True
        return job_to_return

    def plot_dos(self, ax=None, *args, axis=None, **kwargs):
        """
        Plot the DOS.
        If "label" is present in `kwargs` a legend is added to the plot automatically.
        Args:
            axis (optional): matplotlib axis to use, if None create a new one
            ax: deprecated alias for axis
            *args: passed to `axis.plot`
            **kwargs: passed to `axis.plot`
        Returns:
            matplotlib.axes._subplots.AxesSubplot: axis with the plot
        """
        try:
            import pylab as plt
        except ImportError:
            import matplotlib.pyplot as plt
        if ax is not None and axis is None:
            axis = ax
        if axis is None:
            _, axis = plt.subplots(1, 1)
        if not self._interactive_disable_log_file:
            axis.plot(
                self["output/dos_energies"], self["output/dos_total"], *args, **kwargs
            )
        else:
            dos_dict = self.phonopy.get_total_dos_dict()
            axis.plot(
                dos_dict["frequency_points"], dos_dict["total_dos"], *args, **kwargs
            )
        axis.set_xlabel("Frequency [THz]")
        axis.set_ylabel("DOS")
        axis.set_title("Phonon DOS vs Energy")
        if "label" in kwargs:
            axis.legend()
        return ax
