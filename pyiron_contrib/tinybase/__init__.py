import abc
import enum
from typing import Optional

from pyiron_base.interfaces.object import HasStorage

from .executor import (
        ForegroundExecutor,
        BackgroundExecutor
)

def make_storage_mapping(name):
    def fget(self):
        return self.storage[name]

    def fset(self, value):
        self.storage[name] = value

    return property(fget=fget, fset=fset)

class AbstractInput(HasStorage, abc.ABC):
    pass

class StructureInput(AbstractInput):
    def __init__(self):
        super().__init__()
        self.storage.structure = None

    structure = make_storage_mapping('structure')

class AbstractOutput(HasStorage, abc.ABC):
    pass

class ReturnStatus:

    class Code(enum.Enum):
        DONE = "done"
        ABORTED = "aborted"
        WARNING = "warning"
        NOT_CONVERGED = "not_converged"

    def __init__(self, code, msg=None):
        self.code = code if not isinstance(code, str) else ReturnStatus.Code(code)
        self.msg = msg

    def __repr__(self):
        return f"ReturnStatus({self.code}, {self.msg})"
    def __str__(self):
        return f"{self.code}({self.msg})"

    def is_done(self):
        return self.code == self.Code.DONE

class AbstractNode(abc.ABC):

    _executors = {
            'foreground': ForegroundExecutor,
            'background': BackgroundExecutor
    }

    def __init__(self):
        self._input, self._output = None, None

    @abc.abstractmethod
    def _get_input(self) -> AbstractInput:
        pass

    @property
    def input(self) -> AbstractInput:
        if self._input is None:
            self._input = self._get_input()
        return self._input

    @abc.abstractmethod
    def _get_output(self) -> AbstractOutput:
        pass

    @property
    def output(self) -> AbstractOutput:
        if self._output is None:
            self._output = self._get_output()
        return self._output

    @abc.abstractmethod
    def _execute(self) -> Optional[ReturnStatus]:
        pass

    def execute(self) -> ReturnStatus:
        try:
            ret = self._execute()
            if ret is None:
                ret = ReturnStatus("done")
        except Exception as e:
            ret = ReturnStatus("aborted", msg=e)
        return ret

    def run(self, how='foreground'):
        exe = self._executors[how](node=self)
        exe.run()
        return exe
