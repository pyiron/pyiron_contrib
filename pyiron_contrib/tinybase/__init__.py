import abc
import enum
from typing import Union

from pyiron_base.interfaces.object import HasStorage

class RunMachine:

    class Code(enum.Enum):
        INIT = 'init'
        RUNNING = 'running'
        FINISHED = 'finished'

    def __init__(self, initial_state):
        self._state = RunMachine.Code(initial_state)
        self._callbacks = {}
        self._data = {} # state variables associated with each state

    def on(self, state: Union[str, Code], callback):
        if isinstance(state, str):
            state = RunMachine.Code(state)
        self._callbacks[state] = callback

    def goto(self, state: Union[str, Code], **kwargs):
        if isinstance(state, str):
            state = RunMachine.Code(state)
        self._state = state
        self._data = {}
        self._data.update(kwargs)

    def step(self, state: Union[str, Code, None] = None, **kwargs):
        if state is not None:
            self.goto(state, **kwargs)
        self._callbacks.get(self._state, lambda: pass)()

class Executor(abc.ABC):

    def __init__(self):
        self._run_machine = RunMachine("init")

    def run(self):
        self._run_machine.step()

class SingleExecutor(Executor, abc.ABC):

    def __init__(self, node):
        super().__init__()
        self._node = node

    @property
    def node(self):
        return self._node

class ForegroundExecutor(SingleExecutor):

    def __init__(self, node):
        super().__init__(node=node)
        self._run_machine.on("init", self.run_init)

    def run_init(self):
        self._run_machine.goto("running")

        try:
            ret = self.node.execute()
        except Exception as e:
            ret = ReturnStatus("aborted", msg=e)

        self._run_machine.goto("finished", status=ret)

from threading import Thread

class BackgroundExecutor(SingleExecutor):

    def __init__(self, node):
        super().__init__(node=node)
        self._run_machine = RunMachine("init")
        self._run_machine.on("init", self.run_init)
        self._run_machine.on("running", self.run_running)
        self._thread = None

    def run_init(self):
        self._run_machine.goto("running")

        node = self.node
        class NodeThread(Thread):
            def run(self):
                try:
                    self.ret = node.execute()
                except Exception as e:
                    self.ret = ReturnStatus("aborted", msg=e)


        self._thread = NodeThread()
        self._thread.start()

    def run_running(self):
        self._thread.join(timeout=0)
        if not self._thread.is_alive():
            self._run_machine.goto("finished", status=self._thread.ret)
        else:
            print("Node is still running!")

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
    def execute(self) -> ReturnStatus:
        pass

    def run(self, how='foreground'):
        exe = self._executors[how](node=self)
        exe.run()
        return exe
