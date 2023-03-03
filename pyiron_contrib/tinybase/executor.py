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

from concurrent.futures import ProcessPoolExecutor

def run_node(node):
    ret = node.execute()
    return ret, node.output

class ProcessExecutor(Executor):

    def __init__(self, nodes, processes=None):
        super().__init__(nodes=nodes)
        self._processes = processes if processes is not None else 4
        self._pool = None
        self._futures = {}
        self._returns = {}

    def _run_running(self):

        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self._processes)
        pool = self._pool

        still_running = False
        for node in self.nodes:
            if node not in self._futures:
                self._futures[node] = pool.submit(run_node, node)
                still_running = True
            else:
                future = self._futures[node]
                if future.done():
                    # TODO breaks API
                    ret, output = future.result(timeout=0)
                    node._output = output
                    self._returns[node] = ret
                else:
                    still_running = True

        if still_running:
            logging.info("Some nodes are still executing!")
        else:
            pool.shutdown()
            # how to ensure ordering?  dict remembers insertion order, so maybe ok for now
            self._run_machine.step("collect", status=list(self._returns.values()))




class HdfExecutor(Executor):

    def __init__(self):
        self._run_machine.observe("running", self._save_input)
        self._run_machine.observe("finished", self._save_output)

    def _save_input(self, data):
        self.node.input.to_hdf(self._hdf) # where's that coming from?

    def _save_output(self, data):
        self.node.output.to_hdf(self._hdf) # where's that coming from?
