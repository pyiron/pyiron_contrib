import abc
from copy import deepcopy
import contextlib
from dataclasses import asdict
import enum
from tempfile import TemporaryDirectory
from typing import Optional, Callable, List, Generator, Tuple, Any, Union

from pyiron_base.interfaces.object import HasStorage

from pyiron_contrib.tinybase.storage import Storable
from pyiron_contrib.tinybase.container import (
    AbstractInput,
    AbstractOutput,
    StorageAttribute,
    USER_REQUIRED,
    field,
)


class ReturnStatus:
    """
    Status of the calculation.
    """

    class Code(enum.Enum):
        DONE = "done"
        ABORTED = "aborted"
        WARNING = "warning"
        NOT_CONVERGED = "not_converged"

    def __init__(self, code, msg=None):
        self.code = code if not isinstance(code, str) else ReturnStatus.Code(code)
        self.msg = msg

    @classmethod
    def done(cls, msg=None):
        return cls(code=cls.Code.DONE, msg=msg)

    @classmethod
    def aborted(cls, msg=None):
        return cls(code=cls.Code.ABORTED, msg=msg)

    @classmethod
    def warning(cls, msg=None):
        return cls(code=cls.Code.WARNING, msg=msg)

    @classmethod
    def not_converged(cls, msg=None):
        return cls(code=cls.Code.NOT_CONVERGED, msg=msg)

    def __repr__(self):
        return f"ReturnStatus({self.code}, {self.msg})"

    def __str__(self):
        return f"{self.code}({self.msg})"

    def is_done(self) -> True:
        """
        Returns True if calculation was successful.
        """
        return self.code == self.Code.DONE


class ComputeContext(AbstractInput):
    cores: int = None
    gpus: int = None
    runtime: float = None
    memory: float = None
    working_directory: str = None


class AbstractTask(Storable, abc.ABC):
    """
    Basic unit of calculations.

    Subclasses must implement :meth:`._get_input()` and :meth:`._execute()` and generally supply
    their own :class:`.AbstractInput` and :class:`.AbstractOutput` (as returned from `_execute()`).
    """

    def __init__(self, capture_exceptions=True):
        self._input = None
        self._context = ComputeContext()
        self._capture_exceptions = capture_exceptions

    @property
    def context(self):
        return self._context

    @abc.abstractmethod
    def _get_input(self) -> AbstractInput:
        """
        Return an instance of the input class.

        This is called once per life time of the task object on first access to :attr:`.input` and then saved.
        """
        pass

    @property
    def input(self) -> AbstractInput:
        if self._input is None:
            self._input = self._get_input()
        return self._input

    @abc.abstractmethod
    def _execute(self) -> Union[Tuple[ReturnStatus, AbstractOutput], AbstractOutput]:
        """
        Run the calculation.

        Should return either the output object or a :class:`.ReturnStatus`

        Returns:
            :class:`.ReturnStatus`: optional
        """
        pass

    def execute(self) -> Tuple[ReturnStatus, Optional[AbstractOutput]]:
        if not self.input.check_ready():
            return ReturnStatus.aborted("Input not ready!"), None
        if self.context.working_directory is not None:
            nwd = contextlib.nullcontext(self.context.working_directory)
        else:
            nwd = TemporaryDirectory()
        try:
            with nwd as path, contextlib.chdir(path):
                ret = self._execute()
            if isinstance(ret, tuple):
                ret, output = ret
            elif isinstance(ret, AbstractOutput):
                output = ret
                ret = ReturnStatus("done")
            else:
                raise ValueError("Must return tuple or output!")
            return ret, output
        except Exception as e:
            if not self._capture_exceptions:
                raise
            return ReturnStatus("aborted", msg=e), None

    # TaskIterator Impl'
    def __iter__(
        self,
    ) -> Generator[
        List["Task"],
        List[Tuple[ReturnStatus, AbstractOutput]],
        Tuple[ReturnStatus, AbstractOutput],
    ]:
        ret, *_ = yield [self]
        return ret

    # Storable Impl'
    # We might even avoid this by deriving from HasStorage and put _input in there
    def _store(self, storage):
        storage["input"] = self.input
        storage["context"] = self.context

    @classmethod
    def _restore(cls, storage, version):
        task = cls()
        task._input = storage["input"].to_object()
        task._context = storage["context"].to_object()
        return task

    def then(self, body, task=None):
        series = SeriesTask()
        series.input.first(self)
        series.input.then(FunctionTask(body), lambda inp, out: inp.args.append(out))
        return series


class TaskGenerator(AbstractTask, abc.ABC):
    """
    A generator that yields collections of tasks that can be executed in
    parallel.

    In each iteration it will yield a list of tasks and accept a list of their
    return status and outputs for the next iteration.  When it is done, it will
    return a final return status and output.
    """

    @abc.abstractmethod
    def __iter__(
        self,
    ) -> Generator[
        List["Task"],
        List[Tuple[ReturnStatus, AbstractOutput]],
        Tuple[ReturnStatus, AbstractOutput],
    ]:
        pass

    def _execute(self):
        gen = iter(self)
        tasks = next(gen)
        while True:
            ret = [t.execute() for t in tasks]
            try:
                tasks = gen.send(ret)
            except StopIteration as stop:
                # forward return value of the generator
                return stop.args[0]


# TaskGenerator.register(AbstractTask)
# assert


class FunctionInput(AbstractInput):
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


class FunctionOutput(AbstractOutput):
    result: Any


class FunctionTask(AbstractTask):
    """
    A task that wraps a generic function.

    The given function is called with :attr:`.FunctionInput.args` and :attr:`.FunctionInput.kwargs` as `*args` and
    `**kwargs` respectively.  The return value is set to :attr:`.FunctionOutput.result`.
    """

    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self._function = function

    def _get_input(self):
        return FunctionInput()

    def _execute(self):
        res = self._function(*self.input.args, **self.input.kwargs)
        if isinstance(res, AbstractOutput):
            return res
        else:
            return FunctionOutput(result=res)


class ListInput(abc.ABC):
    """
    The input of :class:`.ListTaskGenerator`.

    To use it overload :meth:`._create_tasks()` here and subclass :class:`.ListTaskGenerator` as well.
    """

    @abc.abstractmethod
    def _create_tasks(self):
        """
        Return a list of tasks to execute.

        This is called once by :class:`.ListNode.execute`.
        """
        pass


class ListTaskGenerator(TaskGenerator, abc.ABC):
    """
    A task that executes other tasks in parallel.

    To use it overload :meth:`._extract_output` here and subclass :class:`.ListInput` as well.
    """

    def _get_input(self):
        return ListInput()

    @abc.abstractmethod
    def _extract_output(self, step, task, ret, output) -> dict:
        """
        Extract the output of each task.

        Args:
            step (int): index of the task to extract the output from,
            corresponds to the index of the task in the list returned by :meth:`.ListInput._create_tasks()`.
            task (:class:`.AbstractTask`): task to extract the output from, you can use this to extract parts of the input as well
            ret (:class:`.ReturnStatus`): the return status of the execution of the task
            output (:class:`.AbstractOutput`): the output of the task to extract
        """
        pass

    @abc.abstractmethod
    def _join_output(self, outputs) -> AbstractOutput:
        pass

    def __iter__(self):
        tasks = self.input._create_tasks()
        returns, outputs = zip(*(yield tasks))

        if any(not r.is_done() for r in returns):
            return ReturnStatus("aborted"), None

        extracted_outputs = []
        for i, (task, ret, output) in enumerate(zip(tasks, returns, outputs)):
            extracted_outputs.append(self._extract_output(i, task, ret, output))

        return ReturnStatus("done"), self._join_output(extracted_outputs)


class SeriesInput(AbstractInput):
    """
    Keeps a list of tasks and their connection functions to run sequentially.

    The number of added tasks must be equal to the number of connections plus one.  It's recommended to set up this
    input with :meth:`.first()` and :meth:`.then()` which can be composed in a builder pattern.  The connection
    functions take as arguments the output of the last task and the input of the next task.  You may call
    :meth:`.then()` any number of times.

    >>> task = SeriesNode()
    >>> def transfer(input, output):
    ...     input.my_param = output.my_result
    >>> task.input.first(MyNode()).then(MyNode(), transfer)
    """

    tasks: list = USER_REQUIRED
    connections: list = USER_REQUIRED

    def check_ready(self):
        if not super().check_ready():
            return False
        if not (0 < len(self.tasks) == len(self.connections) + 1):
            return False
        if not self.tasks[0].input.check_ready():
            return False
        return True

    def first(self, task):
        """
        Set initial task.

        Resets whole input.

        Args:
            task (AbstractTask): the first task to execute

        Returns:
            self: the input object
        """
        self.tasks = [task]
        self.connections = []
        return self

    def then(self, next_task, connection):
        """
        Add a new task and how to connect it to the previous task.

        Args:
            next_task (:class:`~.AbstractTask`): next task to execute
            connection (function): takes the input of next_task and the output
            of the previous task

        Returns:
            self: the input object
        """
        self.tasks.append(next_task)
        self.connections.append(connection)
        return self

    # FIXME: error handling
    def __setattr__(self, name, value):
        if self.tasks is not USER_REQUIRED and len(self.tasks) > 0:
            field_names = [f.name for f in self.tasks[0].input.fields()]
            if name in field_names:
                setattr(self.tasks[0].input, name, value)
                return
        super().__setattr__(name, value)

    def __getattr__(self, name):
        if self.tasks is USER_REQUIRED or len(self.tasks) == 0:
            raise AttributeError(name)
        return getattr(self.tasks[0].input, name)

    def fields(self):
        base = super().fields()
        if self.tasks is not USER_REQUIRED and len(self.tasks) > 0:
            base += self.tasks[0].input.fields()
        return base


class SeriesTask(TaskGenerator):
    """
    Executes a series of tasks sequentially.

    Its input specifies the tasks to execute and functions (:attr:`.SeriesInput.connections`) to move input from one
    output to the next input in the series.
    """

    def _get_input(self):
        return SeriesInput()

    def _get_output(self):
        return self.input.tasks[-1]._get_output()

    def __iter__(self):
        (ret, out), *_ = yield [self.input.tasks[0]]
        if not ret.is_done():
            return ReturnStatus("aborted", ret), None

        for task, connection in zip(self.input.tasks[1:], self.input.connections):
            connection(task.input, out)
            (ret, out), *_ = yield [task]
            if not ret.is_done():
                return ReturnStatus("aborted", ret), None

        return ret, out


class LoopControl(HasStorage):
    def __init__(self, condition, restart):
        super().__init__()
        self._condition = condition
        self._restart = restart

    scratch = StorageAttribute().constructor(dict)

    def condition(self, task: AbstractTask, output: AbstractTask):
        """
        Whether to terminate the loop or not.

        Args:
            task (AbstractTask): the loop body
            output (AbstractOutput): output of the loop body

        Args:
            bool: True to terminate the loop; False to keep it running
        """
        return self._condition(task, output, self.scratch)

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
        self.storage.steps = steps
        self.storage.counter = 0

    def _count_steps(self, output, input, scratch):
        c = self.storage.counter
        self.storage.counter += 1
        return c >= self.storage.steps


class LoopInput(AbstractInput):
    """
    Input for :class:`~.LoopTask`.

    Attributes:
        task (:class:`~.AbstractTask`): the loop body
        control (:class:`.LoopControl`): encapsulates control flow of the loop
    """

    trace: bool = False
    control: LoopControl = USER_REQUIRED

    def repeat(
        self,
        steps: int,
        restart: Optional[Callable[[AbstractOutput, AbstractInput, dict], None]] = None,
    ):
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
        condition: Callable[[AbstractTask, AbstractOutput, dict], bool],
        restart: Callable[[AbstractOutput, AbstractInput, dict], None],
    ):
        """
        Set up a loop control that uses the callables for control flow.

        Args:
            condition (function): takes the loop body, its output and a persistant dict
            restart (function): takes the output of the last loop body, the input of the next one and a persistant dict
        """
        self.control = LoopControl(condition, restart)


class LoopTask(TaskGenerator):
    """
    Generic task to loop over a given input task.
    """

    def _get_input(self):
        return LoopInput()

    def __iter__(self):
        task = deepcopy(self.input.task)
        control = deepcopy(self.input.control)

        while True:
            (ret, out), *_ = yield [task]
            if not ret.is_done():
                return ReturnStatus("aborted", ret), None
            if control.condition(task, out):
                break
            control.restart(out, task.input)
        return ret, out
