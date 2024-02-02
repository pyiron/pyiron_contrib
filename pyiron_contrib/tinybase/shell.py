import os
import subprocess
from glob import glob

from pyiron_base.state import state

from pyiron_contrib.tinybase.container import (
    AbstractInput,
    AbstractOutput,
    USER_REQUIRED,
    field,
)

from pyiron_contrib.tinybase.task import AbstractTask, ReturnStatus

if os.name == "nt":
    EXE_SUFFIX = "bat"
else:
    EXE_SUFFIX = "sh"


class ExecutablePathResolver:
    """
    Locates executables in pyiron resource folders.

    This expects executables to be located in folder structures like this

    {resource_folder}/{module}/bin/run_{code}_{version}.sh

    and be executable (on UNIX).

    If multiple executables are found for the same combination of `module`,
    `code` and `version`, :meth:`.list()` returns all of them sorted first by
    resource path (as in PYIRON_RESOURCE_PATH) and then alphabetically.

    If an executable exists that has `default` in its version and `version` is
    `None`, the first such executable is picked for :meth:`.path()`.

    :meth:`.__str__` is overloaded to :meth:`.path()`.
    """

    def __init__(self, module, code, version=None):
        self._module = module
        self._code = code
        self._version = version

    def list(self, version=None):
        """
        List all possible executables found.

        Returns: list of str
        """
        if version is None:
            version = self._version or "*"
        alternatives = []
        for p in state.settings.resource_paths:
            exe_path = f"run_{self._code}_{version}.{EXE_SUFFIX}"
            bin_path = os.path.join(p, self._module, "bin", exe_path)
            alternatives.extend(sorted(glob(bin_path)))
        return alternatives

    def list_versions(self):
        """
        List unique version strings found.
        """
        exes = self.list(version="*")

        def extract(p):
            return os.path.splitext(
                os.path.basename(p).split(f"run_{self._code}", maxsplit=1)[1]
            )[0][1:]

        return list(set(map(extract, exes)))

    @property
    def version(self):
        vers = self.list_versions()
        for v in vers:
            if "default" in vers:
                return v
        return vers[0]

    @version.setter
    def version(self, value):
        vers = self.list_versions()
        if value in vers:
            self._version = value
        else:
            raise ValueError(f"Given version '{value}' not in {vers}!")

    def path(self):
        """
        Returns a direct path where a executable has been found.
        """
        exes = self.list()
        if self._version is not None:
            return exes[0]
        for p in exes:
            if "default" in p:
                return p
        return exes[0]

    def __str__(self):
        return self.path()


def _zero_list():
    return [0]


class ShellInput(AbstractInput):
    command: str = USER_REQUIRED
    working_directory: str = USER_REQUIRED
    arguments: list = field(default_factory=list)
    environ: dict = field(default_factory=dict)
    allowed_returncode: list = field(default_factory=_zero_list)


class ShellOutput(AbstractOutput):
    stdout: str
    stderr: str
    returncode: int


class ShellTask(AbstractTask):
    def _get_input(self):
        return ShellInput()

    def _execute(self):
        environ = dict(os.environ)
        environ.update({k: str(v) for k, v in self.input.environ.items()})
        proc = subprocess.run(
            [str(self.input.command), *map(str, self.input.arguments)],
            capture_output=True,
            cwd=self.input.working_directory,
            encoding="utf8",
            env=environ,
        )
        output = ShellOutput(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
        allowed_returncode = self.input.allowed_returncode
        if allowed_returncode is None:
            allowed_returncode = [0]
        if proc.returncode not in allowed_returncode:
            return (
                ReturnStatus("aborted", f"non-zero error code {proc.returncode}"),
                output,
            )
        else:
            return output
