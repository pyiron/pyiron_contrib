import abc
import enum
from typing import Optional

from pyiron_base.interfaces.object import HasStorage

from .container import AbstractInput, AbstractOutput
from .executor import (
        ForegroundExecutor,
        BackgroundExecutor,
        ListExecutor,
        SerialListExecutor,
        BackgroundListExecutor
)

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

FunctionInput = AbstractInput.from_attributes("FunctionInput", args=list, kwargs=dict)
FunctionOutput = AbstractOutput.from_attributes("FunctionOutput", "result")
class FunctionNode(AbstractNode):

    def __init__(self, function):
        super().__init__()
        self._function = function

    def _get_input(self):
        return FunctionInput()

    def _get_output(self):
        return FunctionOutput()

    def _execute(self):
        self.output.result = self._function(*self.input.args, **self.input.kwargs)

class ListNode(AbstractNode, abc.ABC):

    _executors = {
            'foreground': SerialListExecutor,
            'background': BackgroundListExecutor
    }

    def __init__(self):
        super().__init__()
        self._nodes = None

    @abc.abstractmethod
    def _create_nodes(self):
        pass

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = self._create_nodes()
        return self._nodes

    @abc.abstractmethod
    def _extract_output(self, step, node, ret):
        pass

    # If our own execute is called we act like a normal node, executing child nodes and then process their output
    def _execute(self):
        for i, node in enumerate(self.nodes):
            ret = node.execute()
            self._extract_output(i, node, ret)

    # If called via run by the user directly we can also dispatch to a list executor
    def run(self, how='foreground'):
        Exe = self._executors[how]
        if issubclass(Exe, ListExecutor):
            exe = Exe(self.nodes)
            exe._run_machine.observe("finished",
                    lambda data: [self._extract_output(i, n, r)
                                    for i, (n, r) in enumerate(zip(self.nodes, data['status']))]
            )
            exe.run()
        else:
            exe = Exe(self)
            exe.run()
        return exe
