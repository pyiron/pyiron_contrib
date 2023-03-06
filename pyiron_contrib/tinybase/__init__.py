import abc
import enum
from typing import Optional

from pyiron_base.interfaces.object import HasStorage

from .container import AbstractInput, AbstractOutput
from .executor import (
        Executor,
        BackgroundExecutor,
        ProcessExecutor
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
            'foreground': Executor,
            'background': BackgroundExecutor,
            'process': ProcessExecutor
    }

    def __init__(self):
        self._input = None

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

    @abc.abstractmethod
    def _execute(self, output) -> Optional[ReturnStatus]:
        pass

    def execute(self) -> ReturnStatus:
        output = self._get_output()
        try:
            ret = self._execute(output)
            if ret is None:
                ret = ReturnStatus("done")
        except Exception as e:
            ret = ReturnStatus("aborted", msg=e)
        return ret, output

    def run(self, how='foreground'):
        exe = self._executors[how](nodes=[self])
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

    def _execute(self, output):
        output.result = self._function(*self.input.args, **self.input.kwargs)

MasterInput = AbstractInput.from_attributes(
        "MasterInput",
        child_executor=lambda: Executor
)

class ListInput(MasterInput, abc.ABC):

    @abc.abstractmethod
    def _create_nodes(self):
        pass

class ListNode(AbstractNode, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def _extract_output(self, output, step, node, ret, node_output):
        pass

    def _execute(self, output):
        nodes = self.input._create_nodes()
        exe = self.input.child_executor(nodes)
        exe.run()
        exe.wait()

        for i, (node, ret, node_output) in enumerate(zip(nodes, exe.status, exe.output)):
            self._extract_output(output, i, node, ret, node_output)

SeriesInputBase = AbstractInput.from_attributes(
        "SeriesInputBase",
        nodes=list,
        connections=list
)

class SeriesInput(SeriesInputBase, MasterInput):
    def check_ready(self):
        return len(self.nodes) == len(connections) + 1

    def first(self, node):
        """
        Set initial node.

        Resets whole input

        Args:
            node (AbstractNode): the first node to execute

        Returns:
            self: the input object
        """
        self.nodes = [node]
        self.connections = []
        return self

    def then(self, next_node, connection):
        """
        Add a new node and how to connect it to the previous node.

        Args:
            next_node (:class:`~.AbstractNode`): next node to execute
            connection (function): takes the input of next_node and the output of the previous node

        Returns:
            self: the input object
        """
        self.nodes.append(next_node)
        self.connections.append(connection)
        return self

class SeriesNode(AbstractNode):

    def _get_input(self):
        return SeriesInput()

    def _get_output(self):
        return self.input.nodes[-1]._get_output()

    def _execute(self, output):
        Exe = self.input.child_executor

        exe = Exe(self.input.nodes[:1])
        exe.run()
        exe.wait()
        ret = exe.status[0]
        if not ret.is_done():
            return ReturnStatus("aborted", ret)

        for node, connection in zip(self.input.nodes[1:], self.input.connections):
            connection(node.input, exe.output[0])
            exe = Exe([node])
            exe.run()
            exe.wait()
            ret = exe.status[0]
            if not ret.is_done():
                return ReturnStatus("aborted", ret)

        output.transfer(exe.output[0])


class TinyJob(abc.ABC):

    _executors = {
            'foreground': Executor,
            'background': BackgroundExecutor,
            'process': ProcessExecutor
    }

    def __init__(self, project, job_name):
        self._project = project
        self._name = job_name

    @abc.abstractmethod
    def _get_node(self):
        pass

    @property
    def input(self):
        return self._node.input

    @property
    def output(self):
        return self._node.output

    def run(self, how='foreground'):
        exe = self._executor = self._executors[how](nodes=[self._node])
        exe._run_machine.observe("ready", self.save_input)
        exe._run_machine.observe("collect", self.save_output)
        exe.run()
        return exe

