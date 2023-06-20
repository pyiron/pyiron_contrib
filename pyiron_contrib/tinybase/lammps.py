import os
from tempfile import TemporaryDirectory

from pymatgen.io.lammps.outputs import (
    parse_lammps_dumps,
    parse_lammps_log,
)

from pyiron_atomistics.lammps.potential import (
    LammpsPotential,
    LammpsPotentialFile,
    list_potentials,
)
from pyiron_atomistics.lammps.control import LammpsControl

from pyiron_contrib.tinybase.container import (
    AbstractInput,
    AbstractOutput,
    StorageAttribute,
    StructureInput,
    EnergyPotOutput,
    EnergyKinOutput,
    ForceOutput,
)
from pyiron_contrib.tinybase.task import (
    AbstractTask,
    ReturnStatus,
)
from pyiron_contrib.tinybase.shell import ShellTask, ExecutablePathResolver


class LammpsInputInput(AbstractInput):
    working_directory = StorageAttribute().type(str)

    calc_type = StorageAttribute()

    def calc_static(self):
        self.calc_type = "static"

    def check_ready(self):
        return (
            self.working_directory is not None
            and self.calc_type == "static"
            and super().check_ready()
        )


class LammpsInputOutput(AbstractOutput):
    working_directory = StorageAttribute().type(str)


class LammpsInputTask(AbstractTask):
    """
    Write a set of input files for lammps calculations.

    The potential input is written to `potential.inp` together with any
    additional files needed by the potential.

    The structure is written to `structure.inp`.

    The lammps input script is to `control.inp`.
    """

    def _get_input(self):
        return LammpsInputInput()

    def _get_output(self):
        return LammpsInputOutput()

    def _execute(self, output):
        with open(
            os.path.join(self.input.working_directory, "structure.inp"), "w"
        ) as f:
            self.input.structure.write(f, format="lammps-data")

        potential = LammpsPotential()
        potential.df = LammpsPotentialFile().find_by_name(self.input.potential)
        potential.write_file(
            file_name="potential.inp", cwd=self.input.working_directory
        )
        potential.copy_pot_files(self.input.working_directory)

        control = LammpsControl()
        assert self.input.calc_type == "static", "Cannot happen"
        control.calc_static()
        control.write_file(file_name="control.inp", cwd=self.input.working_directory)

        output.working_directory = self.input.working_directory


class LammpsStaticParserInput(AbstractInput):
    working_directory = StorageAttribute().type(str)

    def check_ready(self):
        return self.working_directory is not None and super().check_ready()


class LammpsStaticOutput(EnergyPotOutput, EnergyKinOutput, ForceOutput):
    pass


class LammpsStaticParserTask(AbstractTask):
    """
    Parse a static lammps calculation.

    In practice return the last step of lammps dump and log file.
    """

    def _get_input(self):
        return LammpsStaticParserInput()

    def _get_output(self):
        return LammpsStaticOutput()

    def _execute(self, output):
        log = parse_lammps_log(
            os.path.join(self.input.working_directory, "log.lammps")
        )[-1]
        output.energy_pot = energy_pot = log["PotEng"].iloc[-1]
        output.energy_kin = log["TotEng"].iloc[-1] - energy_pot
        dump = list(
            parse_lammps_dumps(os.path.join(self.input.working_directory, "dump.out"))
        )[-1]
        output.forces = dump.data[["fx", "fy", "fz"]].to_numpy()


class LammpsInput(StructureInput):
    potential = StorageAttribute().type(str)

    def list_potentials(self):
        """
        List available potentials compatible with a set structure.

        If no structure is set, return an empty list.
        """
        if self.structure is not None:
            return list_potentials(self.structure)
        else:
            return []

    def check_ready(self):
        return self.potential is not None and super().check_ready()


class LammpsStaticTask(AbstractTask):
    """
    A static calculation with lammps.
    """

    def _get_input(self):
        return LammpsInput()

    def _get_output(self):
        return LammpsStaticOutput()

    def _execute(self, output):
        with TemporaryDirectory() as tmp_dir:
            inp = LammpsInputTask(capture_exceptions=self._capture_exceptions)
            inp.input.working_directory = tmp_dir
            inp.input.structure = self.input.structure
            inp.input.potential = self.input.potential
            inp.input.calc_static()
            ret, out = inp.execute()
            if not ret.is_done():
                return ReturnStatus.aborted(f"Writing input failed: {ret.msg}")

            lmp = ShellTask(capture_exceptions=self._capture_exceptions)
            lmp.input.command = ExecutablePathResolver("lammps", "lammps")
            lmp.input.working_directory = tmp_dir
            ret, out = lmp.execute()
            if not ret.is_done():
                return ReturnStatus.aborted(f"Running lammps failed: {ret.msg}")

            psr = LammpsStaticParserTask(capture_exceptions=self._capture_exceptions)
            psr.input.working_directory = tmp_dir
            ret, out = psr.execute()
            if not ret.is_done():
                return ReturnStatus.aborted(f"Parsing failed: {ret.msg}")
            output.take(out)
