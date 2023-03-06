import abc
import enum
from collections import defaultdict
from typing import Union
import time

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


class Executor:

    def __init__(self, nodes):
        self._nodes = nodes
        self._run_machine = RunMachine("init")

        # Set up basic flow
        self._run_machine.on("init", self._run_init)
        # exists mostly to let downstream code hook into it, e.g. to write input files and such
        self._run_machine.on("ready", self._run_ready)
        self._run_machine.on("running", self._run_running)
        # exists mostly to let downstream code hook into it, e.g. to read output files and such
        self._run_machine.on("collect", self._run_collect)
        self._run_machine.on("finished", self._run_finished)

        # keeping track of run times
        self._running_start = None
        self._collect_start = None
        self._finished_start = None
        self._run_machine.observe("running", self._time_running)
        self._run_machine.observe("collect", self._time_collect)
        self._run_machine.observe("finished", self._time_finished)

    def _time_running(self, _data):
        self._running_start = time.monotonic()

    def _time_collect(self, _data):
        self._collect_start = time.monotonic()
        self._run_time = self._collect_start - self._running_start

    def _time_finished(self, _data):
        self._finished_start = time.monotonic()
        self._collect_time = self._finished_start - self._collect_start

    @property
    def nodes(self):
        return self._nodes

    def _run_init(self):
        if all(node.input.check_ready() for node in self.nodes):
            self._run_machine.step("ready")
        else:
            logging.info("Node is not ready yet!")

    def _run_ready(self):
        self._run_machine.step("running")

    def _run_running(self):
        status, output = zip(*[node.execute() for node in self.nodes])
        self._run_machine.step("collect", status=status, output=output)

    def _run_collect(self):
        self._run_machine.step("finished",
                status=self.status,
                output=self.output,
        )

    def _run_finished(self):
        pass

    def run(self):
        self._run_machine.step()

    def wait(self, until="finished", sleep=0.1):
        """
        Sleep until specified state of the run state machine is reached.
        """
        until = RunMachine.Code(until)
        while until != self._run_machine.state:
            time.sleep(sleep)

    @property
    def status(self):
        return self._run_machine._data["status"]

    @property
    def output(self):
        return self._run_machine._data["output"]


from concurrent.futures import (
        ThreadPoolExecutor,
        ProcessPoolExecutor,
        Executor as FExecutor
)
from threading import Lock

def run_node(node):
    return node.execute()

class FuturesExecutor(Executor, abc.ABC):

    _FuturePoolExecutor: FExecutor = None

    # poor programmer's abstract attribute check
    def __init_subclass__(cls):
        if cls._FuturePoolExecutor is None:
            raise TypeError(f"Subclass {cls} of FuturesExecutor does not define 'FuturePoolExecutor'!")

    def __init__(self, nodes, max_tasks=None):
        super().__init__(nodes=nodes)
        self._max_tasks = max_tasks if max_tasks is not None else 4
        self._done = 0
        self._futures = {}
        self._status = {}
        self._output = {}
        self._index = {}
        self._lock = Lock()

    def _process_future(self, future):
        node = self._futures[future]

        status, output = future.result(timeout=0)
        self._status[node] = status
        self._output[node] = output
        with self._lock:
            self._done += 1
        self._check_finish()

    def _check_finish(self, log=False):
        with self._lock:
            if self._done == len(self.nodes):
                status = [self._status[n] for n in sorted(self.nodes, key=lambda n: self._index[n])]
                output = [self._output[n] for n in sorted(self.nodes, key=lambda n: self._index[n])]
                self._run_machine.step("collect",
                        status=status,
                        output=output,
                )

    def _run_running(self):
        if len(self._futures) < len(self.nodes):
            pool = self._FuturePoolExecutor(max_workers=self._max_tasks)
            try:
                for i, node in enumerate(self.nodes):
                    future = pool.submit(run_node, node)
                    self._futures[future] = node
                    self._index[node] = i
                    future.add_done_callback(self._process_future)
            finally:
                # with statement doesn't allow me to put wait=False, so I gotta do it here with try/finally.
                pool.shutdown(wait=False)
        else:
            logging.info("Some nodes are still executing!")

class BackgroundExecutor(FuturesExecutor):
    _FuturePoolExecutor = ThreadPoolExecutor

class ProcessExecutor(FuturesExecutor):
    _FuturePoolExecutor = ProcessPoolExecutor
