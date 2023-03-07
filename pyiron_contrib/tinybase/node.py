import abc
from copy import deepcopy
import enum
from typing import Optional, Callable

from pyiron_base.interfaces.object import HasStorage

from .container import AbstractInput, AbstractOutput, StorageAttribute
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

LoopInputBase = AbstractInput.from_attributes(
        "LoopInput",
        "control",
        trace=bool,
)

class LoopControl(HasStorage):
    def __init__(self, condition, restart):
        super().__init__()
        self._condition = condition
        self._restart = restart

    scratch = StorageAttribute().default(dict)

    def condition(self, node: AbstractNode, output: AbstractNode):
        """
        Whether to terminate the loop or not.

        Args:
            node (AbstractNode): the loop body
            output (AbstractOutput): output of the loop body

        Args:
            bool: True to terminate the loop; False to keep it running
        """
        return self._condition(node, output, self.scratch)

    def restart(self, output: AbstractOutput, input: AbstractInput):
        """
        Setup the input of the next loop body.

        Args:
            output (AbstractOutput): output of the previous loop body
            input (AbstractInput): input of the next loop body
        """
        self._restart(output, input, self.scratch)

class RepeatLoopControl(LoopControl):
    def __init__(self, steps, restart=lambda *_: None):
        super().__init__(condition=self._count_steps, restart=restart)
        self._steps = steps

    def _count_steps(self, output, input, scratch={}):
        c = scratch.get('counter', 0)
        c += 1
        scratch['counter'] = c
        return c >= self._steps


class LoopInput(LoopInputBase, MasterInput):
    """
    Input for :class:`~.LoopNode`.

    Attributes:
        node (:class:`~.AbstractNode`): the loop body
        control (:class:`.LoopControl`): encapsulates control flow of the loop
    """

    def repeat(self, steps: int, restart: Optional[Callable[[AbstractOutput, AbstractInput, dict], None]] = None):
        """
        Set up a loop control that loops for steps and calls restart in between.

        If restart is not given don't do anything to input of the loop body.
        """
        if restart is not None:
            self.control = RepeatLoopControl(steps, restart)
        else:
            self.control = RepeatLoopControl(steps)

    def control_with(
            self,
            condition: Callable[[AbstractNode, AbstractOutput, dict], bool],
            restart: Callable[[AbstractOutput, AbstractInput, dict], None]
    ):
        """
        Set up a loop control that uses the callables for control flow.

        Args:
            condition (function): takes the loop body, its output and a persistant dict
            restart (function): takes the output of the last loop body, the input of the next one and a persistant dict
        """
        self.control = LoopControl(condition, restart)

class LoopNode(AbstractNode):
    """
    Generic node to loop over a given input node.
    """

    def _get_input(self):
        return LoopInput()

    def _get_output(self):
        return self.input.node._get_output()

    def _execute(self, output):
        node = deepcopy(self.input.node)
        control = deepcopy(self.input.control)
        scratch = {}
        while True:
            exe = self.input.child_executor([node])
            exe.run()
            ret = exe.status[-1]
            out = exe.output[-1]
            if not ret.is_done():
                return ReturnStatus("aborted", ret)
            if control.condition(node, out):
                break
            control.restart(out, node.input)
        output.transfer(out)


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


