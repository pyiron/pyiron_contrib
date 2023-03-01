import abc
import enum
from typing import Union

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
        self._callbacks.get(self._state, lambda: None)()

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

        ret = self.node.execute()

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
                self.ret = node.execute()

        self._thread = NodeThread()
        self._thread.start()

    def run_running(self):
        self._thread.join(timeout=0)
        if not self._thread.is_alive():
            self._run_machine.goto("finished", status=self._thread.ret)
        else:
            print("Node is still running!")

class ListExecutor(Executor, abc.ABC):

    def __init__(self, nodes):
        super().__init__()
        self._nodes = nodes

    @property
    def nodes(self):
        return self._nodes

class SerialListExecutor(ListExecutor):

    def __init__(self, nodes):
        super().__init__(nodes=nodes)
        self._run_machine.on("init", self.run_init)

    def run_init(self):
        self._run_machine.goto("running")

        returns = []
        for node in self.nodes:
            returns.append(node.execute())

        self._run_machine.goto("finished", status=returns)
