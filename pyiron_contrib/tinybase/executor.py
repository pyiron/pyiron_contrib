import abc
import enum
from collections import defaultdict
from typing import Union

import logging

class RunMachine:

    class Code(enum.Enum):
        INIT = 'init'
        READY = 'ready'
        RUNNING = 'running'
        COLLECT = 'collect'
        FINISHED = 'finished'

    def __init__(self, initial_state):
        self._state = RunMachine.Code(initial_state)
        self._callbacks = {}
        self._observers = defaultdict(list)
        self._data = {} # state variables associated with each state

    @property
    def state(self):
        return self._state

    def on(self, state: Union[str, Code], callback):
        if isinstance(state, str):
            state = RunMachine.Code(state)
        self._callbacks[state] = callback

    def observe(self, state: Union[str, Code], callback):
        if isinstance(state, str):
            state = RunMachine.Code(state)
        self._observers[state].append(callback)

    def goto(self, state: Union[str, Code], **kwargs):
        if isinstance(state, str):
            state = RunMachine.Code(state)
        self._state = state
        self._data = {}
        self._data.update(kwargs)
        for obs in self._observers[state]:
            obs(self._data)

    def step(self, state: Union[str, Code, None] = None, **kwargs):
        if state is not None:
            self.goto(state, **kwargs)
        self._callbacks.get(self._state, lambda: None)()

class Executor(abc.ABC):

    def __init__(self, nodes):
        self._nodes = nodes
        self._run_machine = RunMachine("init")
        self._run_machine.on("init", self._run_init)
        # exists mostly to let downstream code hook into it, e.g. to write input files and such
        self._run_machine.on("ready", self._run_ready)
        self._run_machine.on("running", self._run_running)
        # exists mostly to let downstream code hook into it, e.g. to read output files and such
        self._run_machine.on("collect", self._run_collect)
        self._run_machine.on("finished", self._run_finished)

    @property
    def nodes(self):
        return self._nodes

    def _run_init(self):
        if all(node.check_ready() for node in self.nodes):
            self._run_machine.step("ready")
        else:
            logging.info("Node is not ready yet!")

    def _run_ready(self):
        self._run_machine.step("running")

    def _run_running(self):
        ret = [node.execute() for node in self.nodes]
        self._run_machine.step("collect", status=ret)

    def _run_collect(self):
        self._run_machine.step("finished",
                status=self._run_machine._data["status"],
                output=[node.output for node in self.nodes]
        )

    def _run_finished(self):
        pass

    def run(self):
        self._run_machine.step()


from threading import Thread

class BackgroundExecutor(Executor):

    def __init__(self, nodes):
        super().__init__(nodes=nodes)
        self._threads = {}
        self._returns = {}

    def _run_running(self):

        def exec_node(node):
            self._returns[node] = node.execute()

        still_running = False
        for node in self.nodes:
            if node not in self._threads:
                thread = self._threads[node] = Thread(target=exec_node, args=(node,))
                thread.start()
                still_running = True
            else:
                thread = self._threads[node]
                thread.join(timeout=0)
                still_running |= thread.is_alive()
        if still_running:
            logging.info("Some nodes are still executing!")
        else:
            # how to ensure ordering?  dict remembers insertion order, so maybe ok for now
            self._run_machine.step("collect", status=list(self._returns.values()))
