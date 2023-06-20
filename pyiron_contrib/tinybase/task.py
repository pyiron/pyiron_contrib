import abc
from copy import deepcopy
import enum
from typing import Optional, Callable, List, Generator, Tuple

from pyiron_base.interfaces.object import HasStorage

from pyiron_contrib.tinybase.storage import Storable, pickle_dump, pickle_load
from pyiron_contrib.tinybase.container import (
    AbstractInput,
    AbstractOutput,
    StorageAttribute,
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


class AbstractTask(Storable, abc.ABC):
    """
    Basic unit of calculations.

    Subclasses must implement :meth:`._get_input()`, :meth:`._get_output()` and :meth:`._execute()` and generally supply
    their own :class:`.AbstractInput` and :class:`.AbstractOutput`.
    """

    def __init__(self, capture_exceptions=True):
        self._input = None
        self._capture_exceptions = capture_exceptions

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
    def _get_output(self) -> AbstractOutput:
        """
        Return an instance of the output class.

        This is called every time :meth:`.execute()` is called.
        """
        pass

    @abc.abstractmethod
    def _execute(self, output) -> Optional[ReturnStatus]:
        """
        Run the calculation.

        Every time this method is called a new instance of the output is created and passed as the argument.  This
        method should populate it.

        If no value is returned from the method, a return status of DONE is assumed implicitly.

        Args:
            output (:class:`.AbstractOutput`): instance returned by :meth:`._get_output()`.

        Returns:
            :class:`.ReturnStatus`: optional
        """
        pass

    def execute(self) -> Tuple[ReturnStatus, AbstractOutput]:
        if not self.input.check_ready():
            return ReturnStatus.aborted("Input not ready!")
        output = self._get_output()
        try:
            ret = self._execute(output)
            if ret is None:
                ret = ReturnStatus("done")
        except Exception as e:
            ret = ReturnStatus("aborted", msg=e)
            if not self._capture_exceptions:
                raise
        return ret, output

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
        # right now not all ASE objects can be stored in HDF, so let's just pickle for now
        storage["input"] = pickle_dump(self.input)
        # self.input.store(storage, "input")

    @classmethod
    def _restore(cls, storage, version):
        task = cls()
        task._input = pickle_load(storage["input"])
        return task


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

    def _execute(self, output):
        gen = iter(self)
        tasks = next(gen)
        while True:
            ret = [t.execute() for t in tasks]
            try:
                tasks = gen.send(ret)
            except StopIteration as stop:
                ret, out = stop.args[0]
                output.take(out)
                return ret


# TaskGenerator.register(AbstractTask)
# assert


class FunctionInput(AbstractInput):
    args = StorageAttribute().type(list).constructor(list)
    kwargs = StorageAttribute().type(dict).constructor(dict)


class FunctionOutput(AbstractOutput):
    result = StorageAttribute()


class FunctionTask(AbstractTask):
    """
    A task that wraps a generic function.

    The given function is called with :attr:`.FunctionInput.args` and :attr:`.FunctionInput.kwargs` as `*args` and
    `**kwargs` respectively.  The return value is set to :attr:`.FunctionOutput.result`.
    """

    def __init__(self, function):
        super().__init__()
        self._function = function

    def _get_input(self):
        return FunctionInput()

    def _get_output(self):
        return FunctionOutput()

    def _execute(self, output):
        output.result = self._function(*self.input.args, **self.input.kwargs)


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
    def _extract_output(self, output, step, task, ret, task_output):
        """
        Extract the output of each task.

        Args:
            output (:class:`.AbstractOutput`): output of this task to populate
            step (int): index of the task to extract the output from,
            corresponds to the index of the task in the list returned by :meth:`.ListInput._create_tasks()`.
            task (:class:`.AbstractTask`): task to extract the output from, you can use this to extract parts of the input as well
            ret (:class:`.ReturnStatus`): the return status of the execution of the task
            task_output (:class:`.AbstractOutput`): the output of the task to extract
        """
        pass

    def __iter__(self):
        tasks = self.input._create_tasks()
        returns, outputs = zip(*(yield tasks))

        output = self._get_output()
        for i, (task, ret, task_output) in enumerate(zip(tasks, returns, outputs)):
            self._extract_output(output, i, task, ret, task_output)

        return ReturnStatus("done"), output


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

    tasks = StorageAttribute().type(list)
    connections = StorageAttribute().type(list)

    def check_ready(self):
        return len(self.tasks) == len(self.connections) + 1

    def first(self, task):
        """
        Set initial task.

        Resets whole input

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

    trace = StorageAttribute().type(bool).default(False)
    control = StorageAttribute().type(LoopControl)

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

    def _get_output(self):
        return self.input.task._get_output()

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
