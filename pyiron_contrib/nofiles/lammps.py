import os
import importlib
from pyiron_atomistics.lammps.interactive import LammpsInteractive

try:  # mpi4py is only supported on Linux and Mac Os X
    from pylammpsmpi import LammpsLibrary
except ImportError:
    pass


class LammpsInteractiveWithoutOutput(LammpsInteractive):
    def __init__(self, project, job_name):
        super(LammpsInteractiveWithoutOutput, self).__init__(project, job_name)
        self._data_storage_disabled_implemented = True

    def interactive_flush(self, path="interactive", include_last_step=False):
        if self.data_storage_enabled:
            super(LammpsInteractiveWithoutOutput, self).interactive_flush(
                path=path,
                include_last_step=include_last_step
            )

    def interactive_initialize_interface(self):
        if self.data_storage_enabled:
            self._create_working_directory()
        if self.server.run_mode.interactive and self.server.cores == 1:
            lammps = getattr(importlib.import_module("lammps"), "lammps")
            if self._log_file is None:
                self._log_file = os.path.join(self.working_directory, "log.lammps")
            if self.data_storage_enabled:
                self._interactive_library = lammps(
                    cmdargs=["-screen", "none", "-log", self._log_file],
                    comm=self._interactive_mpi_communicator
                )
            else:
                self._interactive_library = lammps(
                    cmdargs=["-screen", "none", "-log", "none"],
                    comm=self._interactive_mpi_communicator
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
        if self.data_storage_enabled:
            super(LammpsInteractiveWithoutOutput).interactive_close()
