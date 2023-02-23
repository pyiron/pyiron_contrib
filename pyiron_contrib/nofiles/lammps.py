import os
import importlib
from pyiron_atomistics.lammps.interactive import LammpsInteractive
from pyiron_contrib.nofiles.wrapper import wrap_without_files

try:  # mpi4py is only supported on Linux and Mac Os X
    from pylammpsmpi import LammpsLibrary
except ImportError:
    pass

LammpsInteractiveWithoutOutputBase = wrap_without_files(
        LammpsInteractive, "LammpsInteractiveWithoutOutputBase",
        LammpsInteractive.interactive_flush,
        LammpsInteractive.interactive_close
)

class LammpsInteractiveWithoutOutput(LammpsInteractiveWithoutOutputBase):

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
