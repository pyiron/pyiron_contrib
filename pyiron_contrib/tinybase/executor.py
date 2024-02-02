import enum
from collections import defaultdict
from functools import partial
from typing import Union, List
import time
import logging
from math import inf

from pyiron_contrib.tinybase.task import AbstractTask, TaskGenerator


class RunMachine:
    class Code(enum.Enum):
        INIT = "init"
        READY = "ready"
        RUNNING = "running"
        COLLECT = "collect"
        FINISHED = "finished"

    def __init__(self, initial_state):
        self._state = RunMachine.Code(initial_state)
        self._callbacks = {}
        self._observers = defaultdict(list)
        self._data = {}  # state variables associated with each state

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


class ExecutionContext:
    def __init__(self, tasks):
        self._tasks = tasks
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
    def tasks(self):
        return self._tasks

    def _run_init(self):
        if all(task.input.check_ready() for task in self.tasks):
            self._run_machine.step("ready")
        else:
            logging.info("Task is not ready yet!")

    def _run_ready(self):
        self._run_machine.step("running")

    def _run_running(self):
        status, output = zip(*[task.execute() for task in self.tasks])
        self._run_machine.step("collect", status=status, output=output)

    def _run_collect(self):
        self._run_machine.step(
            "finished",
            status=self.status,
            output=self.output,
        )

    def _run_finished(self):
        pass

    def run(self):
        self._run_machine.step()

    def wait(self, until="finished", timeout=None, sleep=0.1):
        """
        Sleep until specified state of the run state machine is reached.

        Before calling this method, the state of this context must be past
        `init`, i.e. you have to call :meth:`.run` at least once.

        Args:
            until (str): wait until the executor has reached this state; must
                be a valid state name of :class:`.RunMachine.Code`.
            timeout (float): maximum amount of seconds to wait; wait
                indefinitely by default
            sleep (float): amount of seconds to sleep in between status checks

        Raises:
            ValueError: if the current state is `init`
        """
        if timeout is None:
            timeout = inf
        if self._run_machine.state == RunMachine.Code("init"):
            raise ValueError("Still in state 'init'! Call run() first!")
        until = RunMachine.Code(until)
        start = time.monotonic()
        while until != self._run_machine.state and time.monotonic() - start < timeout:
            time.sleep(sleep)

    @property
    def status(self):
        return self._run_machine._data["status"]

    @property
    def output(self):
        return self._run_machine._data["output"]


class Submitter:
    """
    Simple wrapper around ExecutionContext and its sub classes.

    Exists only to have a single object from which multiple contexts can be
    spawned.
    """

    def submit(self, tasks: List[AbstractTask]) -> ExecutionContext:
        return ExecutionContext(tasks)


from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
)
from threading import Lock


def run_task(task):
    return task.execute()


class FuturesExecutionContext(ExecutionContext):
    def __init__(self, pool, tasks):
        super().__init__(tasks=tasks)
        self._pool = pool
        # self._max_tasks = max_tasks if max_tasks is not None else 4
        self._done = 0
        self._futures = {}
        self._subcontexts = {}
        self._generators = {}
        self._status = {}
        self._output = {}
        self._index = {}
        self._lock = Lock()

    def _process_future(self, future):
        task = self._futures[future]

        # do not specify a timeout even though we already know that the future
        # has finished.  For executors with high latency, low number of workers
        # and large number of tasks transfering the results back to this thread
        # can take longer than one would naively assume (>1ms).  If this is the
        # case we might trip the timeout while waiting for the result, botching
        # the calculation
        status, output = future.result()
        with self._lock:
            self._status[task] = status
            self._output[task] = output
            self._done += 1
        self._check_finish()

    def _prepare_subcontext(self, task, sub_tasks):
        sub = self._subcontexts[task] = type(self)(self._pool, sub_tasks)
        sub._run_machine.observe("finished", partial(self._process_generator, task))
        sub.run()
        return sub

    def _process_generator(self, task, _data):
        gen = self._generators[task]
        sub = self._subcontexts[task]
        try:
            tasks = gen.send(list(zip(sub.status, sub.output)))
            self._prepare_subcontext(task, tasks)
        except StopIteration as stop:
            with self._lock:
                self._status[task], self._output[task] = stop.args[0]
                self._done += 1
                del self._subcontexts[task]
                del self._generators[task]
            self._check_finish()

    def _check_finish(self, log=False):
        with self._lock:
            if self._done == len(self.tasks):
                status = [
                    self._status[n]
                    for n in sorted(self.tasks, key=lambda n: self._index[n])
                ]
                output = [
                    self._output[n]
                    for n in sorted(self.tasks, key=lambda n: self._index[n])
                ]
                self._run_machine.step(
                    "collect",
                    status=status,
                    output=output,
                )

    def _run_running(self):
        if len(self._futures) == 0:
            for i, task in enumerate(self.tasks):
                self._index[task] = i
                if isinstance(task, TaskGenerator):
                    gen = self._generators[task] = iter(task)
                    self._prepare_subcontext(task, next(gen))
                else:
                    future = self._pool.submit(run_task, task)
                    self._futures[future] = task
                    future.add_done_callback(self._process_future)
        else:
            logging.info("Some tasks are still executing!")


class FuturesSubmitter(Submitter):
    def __init__(self, executor):
        self._executor = executor

    def submit(self, tasks):
        return FuturesExecutionContext(self._executor, tasks)
