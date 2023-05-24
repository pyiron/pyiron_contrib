import os
import importlib
from pyiron_atomistics.lammps.interactive import LammpsInteractive
from pyiron_base import state

try:  # mpi4py is only supported on Linux and Mac Os X
    from pylammpsmpi import LammpsLibrary
except ImportError:
    pass


class LammpsInteractiveWithoutOutput(LammpsInteractive):
    def __init__(self, project, job_name):
        if not state.database.database_is_disabled:
            raise RuntimeError(
                "To run a `Without` job, the database must first be disabled. Please "
                "`from pyiron_base import state; "
                "state.update({'disable_database': True})`, and try again."
            )
        super().__init__(project, job_name)
        self._interactive_disable_log_file = True

    def to_hdf(self, hdf=None, group_name=None):
        """

        Args:
            hdf:
            group_name:

        Returns:

        """
        if not self._interactive_disable_log_file:
            super(LammpsInteractiveWithoutOutput, self).to_hdf(
                hdf=hdf, group_name=group_name
            )

    def interactive_flush(self, path="interactive", include_last_step=False):
        if not self._interactive_disable_log_file:
            super(LammpsInteractiveWithoutOutput, self).interactive_flush(
                path=path, include_last_step=include_last_step
            )

    def interactive_initialize_interface(self):
        if not self._interactive_disable_log_file:
            self._create_working_directory()
        if self.server.run_mode.interactive and self.server.cores == 1:
            lammps = getattr(importlib.import_module("lammps"), "lammps")
            if self._log_file is None:
                self._log_file = os.path.join(self.working_directory, "log.lammps")
            if not self._interactive_disable_log_file:
                self._interactive_library = lammps(
                    cmdargs=["-screen", "none", "-log", self._log_file],
                    comm=self._interactive_mpi_communicator,
                )
            else:
                self._interactive_library = lammps(
                    cmdargs=["-screen", "none", "-log", "none"],
                    comm=self._interactive_mpi_communicator,
                )
        else:
            self._interactive_library = LammpsLibrary(
                cores=self.server.cores, working_directory=self.working_directory
            )
        if not all(self.structure.pbc):
            self.input.control["boundary"] = " ".join(
                ["p" if coord else "f" for coord in self.structure.pbc]
            )
        self._reset_interactive_run_command()
        self.interactive_structure_setter(self.structure)

    def interactive_close(self):
        if not self._interactive_disable_log_file:
            super().interactive_close()

    def refresh_job_status(self):
        if not self._interactive_disable_log_file:
            super().refresh_job_status()
