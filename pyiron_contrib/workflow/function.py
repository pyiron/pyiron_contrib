from __future__ import annotations

import inspect
import warnings
from functools import partialmethod
from typing import get_args, get_type_hints, Optional, TYPE_CHECKING

from pyiron_contrib.workflow.channels import InputData, OutputData, NotData
from pyiron_contrib.workflow.has_channel import HasChannel
from pyiron_contrib.workflow.io import Inputs, Outputs, Signals
from pyiron_contrib.workflow.node import Node

if TYPE_CHECKING:
    from pyiron_contrib.workflow.composite import Composite
    from pyiron_contrib.workflow.workflow import Workflow


class Function(Node):
    """
    Function nodes wrap an arbitrary python function.
    Node IO, including type hints, is generated automatically from the provided function
    and (in the case of labeling output channels) the provided output labels.
    On running, the function node executes this wrapped function with its current input
    and uses the results to populate the node output.

    Function nodes must be instantiated with a callable to deterimine their function,
    and a string to name each returned value of that callable. (If you really want to
    return a tuple, just have multiple return values but only one output label -- there
    is currently no way to mix-and-match, i.e. to have multiple return values at least
    one of which is a tuple.)

    The node label (unless otherwise provided), IO types, and input defaults for the
    node are produced _automatically_ from introspection of the node function.
    Additional properties like storage priority (present but doesn't do anything yet)
    and ontological type (not yet present) can be set using kwarg dictionaries with
    keys corresponding to the channel labels (i.e. the node arguments of the node
    function, or the output labels provided).

    Actual function node instances can either be instances of the base node class, in
    which case the callable node function and output labels *must* be provided, in
    addition to other data, OR they can be instances of children of this class.
    Those children may define some or all of the node behaviour at the class level, and
    modify their signature accordingly so this is not available for alteration by the
    user, e.g. the node function and output labels may be hard-wired.

    Although not strictly enforced, it is a best-practice that where possible, function
    nodes should be both functional (always returning the same output given the same
    input) and idempotent (not modifying input data in-place, but creating copies where
    necessary and returning new objects as output).

    Args:
        node_function (callable): The function determining the behaviour of the node.
        *output_labels (str): A name for each return value of the node function.
        label (str): The node's label. (Defaults to the node function's name.)
        run_on_updates (bool): Whether to run when you are updated and all your
            input is ready. (Default is False).
        update_on_instantiation (bool): Whether to force an update at the end of
            instantiation. (Default is False.)
        channels_requiring_update_after_run (list[str]): All the input channels named
            here will be set to `wait_for_update()` at the end of each node run, such
            that they are not `ready` again until they have had their `.update` method
            called. This can be used to create sets of input data _all_ of which must
            be updated before the node is ready to produce output again. (Default is
            None, which makes the list empty.)
        **kwargs: Any additional keyword arguments whose keyword matches the label of an
            input channel will have their value assigned to that channel.

    Attributes:
        inputs (Inputs): A collection of input data channels.
        outputs (Outputs): A collection of output data channels.
        signals (Signals): A holder for input and output collections of signal channels.
        ready (bool): All input reports ready, not running or failed.
        running (bool): Currently running.
        failed (bool): An exception was thrown when executing the node function.
        connected (bool): Any IO channel has at least one connection.
        fully_connected (bool): Every IO channel has at least one connection.

    Methods:
        update: If `run_on_updates` is true and all your input is ready, will
            run the engine.
        run: Parse and process the input, execute the engine, process the results and
            update the output.
        disconnect: Disconnect all data and signal IO connections.

    Examples:
        At the most basic level, to use nodes all we need to do is provide the
        `Function` class with a function and labels for its output, like so:
        >>> from pyiron_contrib.workflow.function import Function
        >>>
        >>> def mwe(x, y):
        ...     return x+1, y-1
        >>>
        >>> plus_minus_1 = Function(mwe, "p1", "m1")
        >>>
        >>> print(plus_minus_1.outputs.p1)
        <class 'pyiron_contrib.workflow.channels.NotData'>

        There is no output because we haven't given our function any input, it has
        no defaults, and we never ran it! So it has the channel default value of
        `NotData` -- a special non-data class (since `None` is sometimes a meaningful
        value in python).

        We'll run into a hiccup if we try to set only one of the inputs and update
        >>> plus_minus_1.inputs.x = 1
        >>> plus_minus_1.run()
        TypeError

        This is because the second input (y) still has no input value, so we can't do
        the sum.
        Let's set the node to run automatically when its inputs are updated, then update
        x and y.
        >>> plus_minus_1.run_on_updates = True
        >>> plus_minus_1.inputs.x = 2
        >>> print(plus_minus_1.outputs.p1.value)
        <class 'pyiron_contrib.workflow.channels.NotData'>

        The gentler `update()` call sees that the `y` input is still `NotData`, so it
        does not proceed to the `run()` and the output is not yet updated.

        Let's provide a y-value as well:
        >>> plus_minus_1.inputs.y = 3
        >>> plus_minus_1.outputs.to_value_dict()
        {'p1': 3, 'm1': 2}

        Now that both inputs have been provided, the node update triggers a run and we
        get the expected output.

        We can also, optionally, provide initial values for some or all of the input
        >>> plus_minus_1 = Function(
        ...     mwe, "p1", "m1",
        ...     x=1,
        ...     run_on_updates=True
        )
        >>> plus_minus_1.inputs.y = 2  # Automatically triggers an update call now
        >>> plus_minus_1.outputs.to_value_dict()
        {'p1': 2, 'm1': 1}

        Finally, we might want the node to be ready-to-go right after instantiation.
        To do this, we need to provide initial values for everything and set two flags:
        >>> plus_minus_1 = Function(
        ...     mwe, "p1", "m1",
        ...     x=0, y=0,
        ...     run_on_updates=True, update_on_instantiation=True
        ... )
        >>> plus_minus_1.outputs.to_value_dict()
        {'p1': 1, 'm1': -1}

        Another way to stop the node from running with bad input is to provide type
        hints (and, optionally, default values) when defining the function the node
        wraps. All of these get determined by inspection.

        We can provide initial values for our node function at instantiation using our
        kwargs.
        The node update is deferred until _all_ of these initial values are processed.
        Thus, if  _all_ the arguments of our function are receiving good enough initial
        values to facilitate an execution of the node function at the end of
        instantiation, the output gets updated right away:
        >>> plus_minus_1 = Function(
        ...     mwe, "p1", "m1",
        ...     x=1, y=2,
        ...     run_on_updates=True, update_on_instantiation=True
        ... )
        >>>
        >>> print(plus_minus_1.outputs.to_value_dict())
        {'p1': 2, 'm1': 1}

        Second, we could add type hints/defaults to our function so that it knows better
        than to try to evaluate itself with bad data.
        You can always force the node to run with its current input using `run()`, but
        `update()` will always check if the node is `ready` -- i.e. if none of its
        inputs are `NotData` and all of them obey any type hints that have been
        provided.
        Let's make a new node following the second path.

        In this example, note the mixture of old-school (`typing.Union`) and new (`|`)
        type hints as well as nested hinting with a union-type inside the tuple for the
        return hint.
        Our treatment of type hints is **not infinitely robust**, but covers a wide
        variety of common use cases.
        >>> from typing import Union
        >>>
        >>> def hinted_example(
        ...     x: Union[int, float],
        ...     y: int | float = 1
        ... ) -> tuple[int, int | float]:
        ...     return x+1, y-1
        >>>
        >>> plus_minus_1 = Function(
        ...     hinted_example, "p1", "m1",
        ...     run_on_updates=True, update_on_instantiation=True
        ... )
        >>> plus_minus_1.outputs.to_value_dict()
        {'p1': <class 'pyiron_contrib.workflow.channels.NotData'>, 'm1': <class
        'pyiron_contrib.workflow.channels.NotData'>}

        Here we got an update automatically at the end of instantiation, but because
        both values are type hinted this didn't result in any errors!
        Still, we need to provide the rest of the input data in order to get results:

        >>> plus_minus_1.inputs.x = 1
        >>> plus_minus_1.outputs.to_value_dict()
        {'p1': 2, 'm1': 0}

        Note: the `Fast(Node)` child class will enforce all function arguments to
        be type-hinted and have defaults, and will automatically set the updating and
        instantiation flags to `True` for nodes that execute quickly and are meant to
        _always_ have good output data.

        In these examples, we've instantiated nodes directly from the base `Function`
        class, and populated their input directly with data.
        In practice, these nodes are meant to be part of complex workflows; that means
        both that you are likely to have particular nodes that get heavily re-used, and
        that you need the nodes to pass data to each other.

        For reusable nodes, we want to create a sub-class of `Function` that fixes some
        of the node behaviour -- usually the `node_function` and `output_labels`.

        This can be done most easily with the `node` decorator, which takes a function
        and returns a node class:
        >>> from pyiron_contrib.workflow.function import function_node
        >>>
        >>> @function_node(
        ...     "p1", "m1",
        ...     run_on_updates=True, update_on_instantiation=True
        ... )
        ... def my_mwe_node(
        ...     x: int | float, y: int | float = 1
        ... ) -> tuple[int | float, int | float]:
        ...     return x+1, y-1
        >>>
        >>> node_instance = my_mwe_node(x=0)
        >>> node_instance.outputs.to_value_dict()
        {'p1': 1, 'm1': 0}

        Where we've passed the output labels and class arguments to the decorator,
        and inital values to the newly-created node class (`my_mwe_node`) at
        instantiation.
        Because we told it to run on updates and to update on instantation _and_ we
        provided a good initial value for `x`, we get our result right away.

        Using the decorator is the recommended way to create new node classes, but this
        magic is just equivalent to these two more verbose ways of defining a new class.
        The first is to override the `__init__` method directly:
        >>> from typing import Literal, Optional
        >>>
        >>> class AlphabetModThree(Function):
        ...     def __init__(
        ...         self,
        ...         label: Optional[str] = None,
        ...         run_on_updates: bool = True,
        ...         update_on_instantiation: bool = False,
        ...         **kwargs
        ...     ):
        ...         super().__init__(
        ...             self.alphabet_mod_three,
        ...             "letter",
        ...             labe=label,
        ...             run_on_updates=run_on_updates,
        ...             update_on_instantiation=update_on_instantiation,
        ...             **kwargs
        ...         )
        ...
        ...     @staticmethod
        ...     def alphabet_mod_three(i: int) -> Literal["a", "b", "c"]:
        ...         return ["a", "b", "c"][i % 3]

        The second effectively does the same thing, but leverages python's
        `functools.partialmethod` to do so much more succinctly.
        In this example, note that the function is declared _before_ `__init__` is set,
        so that it is available in the correct scope (above, we could place it
        afterwards because we were accessing it through self).
        >>> from functools import partialmethod
        >>>
        >>> class Adder(Function):
        ...     @staticmethod
        ...     def adder(x: int = 0, y: int = 0) -> int:
        ...         return x + y
        ...
        ...     __init__ = partialmethod(
        ...         Function.__init__,
        ...         adder,
        ...         "sum",
        ...         run_on_updates=True,
        ...         update_on_instantiation=True
        ...     )

        Finally, let's put it all together by using both of these nodes at once.
        Instead of setting input to a particular data value, we'll set it to
        be another node's output channel, thus forming a connection.
        When we update the upstream node, we'll see the result passed downstream:
        >>> adder = Adder()
        >>> alpha = AlphabetModThree(i=adder.outputs.sum)
        >>>
        >>> adder.inputs.x = 1
        >>> print(alpha.outputs.letter)
        "b"
        >>> adder.inputs.y = 1
        >>> print(alpha.outputs.letter)
        "c"
        >>> adder.inputs.x = 0
        >>> adder.inputs.y = 0
        >>> print(alpha.outputs.letter)
        "a"

        To see more details on how to use many nodes together, look at the
        `Workflow` class.

    Comments:

        If you use the function argument `self` in the first position, the
        whole node object is inserted there:

        >>> def with_self(self, x):
        >>>     ...
        >>>     return x

        For this function, you don't have a freedom to choose `self`, because
        pyiron automatically sets the node object there (which is also the
        reason why you do not see `self` in the list of inputs).
    """

    def __init__(
        self,
        node_function: callable,
        *output_labels: str,
        label: Optional[str] = None,
        run_on_updates: bool = False,
        update_on_instantiation: bool = False,
        channels_requiring_update_after_run: Optional[list[str]] = None,
        parent: Optional[Composite] = None,
        **kwargs,
    ):
        super().__init__(
            label=label if label is not None else node_function.__name__,
            parent=parent,
            # **kwargs,
        )
        if len(output_labels) == 0:
            raise ValueError("Nodes must have at least one output label.")

        self.node_function = node_function

        self._inputs = None
        self._outputs = None
        self._output_labels = output_labels
        # TODO: Parse output labels from the node function in case output_labels is None

        self.signals = self._build_signal_channels()

        self.channels_requiring_update_after_run = (
            []
            if channels_requiring_update_after_run is None
            else channels_requiring_update_after_run
        )
        self._verify_that_channels_requiring_update_all_exist()

        self.run_on_updates = False
        # Temporarily disable running on updates to set all initial values at once
        for k, v in kwargs.items():
            if k in self.inputs.labels:
                self.inputs[k] = v
            elif k not in self._init_keywords:
                warnings.warn(f"The keyword '{k}' was received but not used.")
        self.run_on_updates = run_on_updates  # Restore provided value

        if update_on_instantiation:
            self.update()

    @property
    def _input_args(self):
        return inspect.signature(self.node_function).parameters

    @property
    def inputs(self) -> Inputs:
        if self._inputs is None:
            self._inputs = Inputs(*self._build_input_channels())
        return self._inputs

    @property
    def outputs(self) -> Outputs:
        if self._outputs is None:
            self._outputs = Outputs(*self._build_output_channels(*self._output_labels))
        return self._outputs

    def _build_input_channels(self):
        channels = []
        type_hints = get_type_hints(self.node_function)

        for ii, (label, value) in enumerate(self._input_args.items()):
            is_self = False
            if label == "self":  # `self` is reserved for the node object
                if ii == 0:
                    is_self = True
                else:
                    warnings.warn(
                        "`self` is used as an argument but not in the first"
                        " position, so it is treated as a normal function"
                        " argument. If it is to be treated as the node object,"
                        " use it as a first argument"
                    )
            if label in self._init_keywords:
                # We allow users to parse arbitrary kwargs as channel initialization
                # So don't let them choose bad channel names
                raise ValueError(
                    f"The Input channel name {label} is not valid. Please choose a "
                    f"name _not_ among {self._init_keywords}"
                )

            try:
                type_hint = type_hints[label]
                if is_self:
                    warnings.warn("type hint for self ignored")
            except KeyError:
                type_hint = None

            default = NotData  # The standard default in DataChannel
            if value.default is not inspect.Parameter.empty:
                if is_self:
                    warnings.warn("default value for self ignored")
                else:
                    default = value.default

            if not is_self:
                channels.append(
                    InputData(
                        label=label,
                        node=self,
                        default=default,
                        type_hint=type_hint,
                    )
                )
        return channels

    @property
    def _init_keywords(self):
        return list(inspect.signature(self.__init__).parameters.keys())

    def _build_output_channels(self, *return_labels: str):
        try:
            type_hints = get_type_hints(self.node_function)["return"]
            if len(return_labels) > 1:
                type_hints = get_args(type_hints)
                if not isinstance(type_hints, tuple):
                    raise TypeError(
                        f"With multiple return labels expected to get a tuple of type "
                        f"hints, but got type {type(type_hints)}"
                    )
                if len(type_hints) != len(return_labels):
                    raise ValueError(
                        f"Expected type hints and return labels to have matching "
                        f"lengths, but got {len(type_hints)} hints and "
                        f"{len(return_labels)} labels: {type_hints}, {return_labels}"
                    )
            else:
                # If there's only one hint, wrap it in a tuple so we can zip it with
                # *return_labels and iterate over both at once
                type_hints = (type_hints,)
        except KeyError:
            type_hints = [None] * len(return_labels)

        channels = []
        for label, hint in zip(return_labels, type_hints):
            channels.append(
                OutputData(
                    label=label,
                    node=self,
                    type_hint=hint,
                )
            )

        return channels

    def _verify_that_channels_requiring_update_all_exist(self):
        if not all(
            channel_name in self.inputs.labels
            for channel_name in self.channels_requiring_update_after_run
        ):
            raise ValueError(
                f"On or more channel name among those listed as requiring updates "
                f"after the node runs ({self.channels_requiring_update_after_run}) was "
                f"not found among the input channels ({self.inputs.labels})"
            )

    @property
    def on_run(self):
        return self.node_function

    @property
    def run_args(self) -> dict:
        kwargs = self.inputs.to_value_dict()
        if "self" in self._input_args:
            kwargs["self"] = self
        return kwargs

    def process_run_result(self, function_output):
        """
        Take the results of the node function, and use them to update the node output.

        By extracting this as a separate method, we allow the node to pass the actual
        execution off to another entity and release the python process to do other
        things. In such a case, this function should be registered as a callback
        so that the node can finishing "running" and push its data forward when that
        execution is finished.
        """
        for channel_name in self.channels_requiring_update_after_run:
            self.inputs[channel_name].wait_for_update()

        if len(self.outputs) == 1:
            function_output = (function_output,)

        for out, value in zip(self.outputs, function_output):
            out.update(value)

    def __call__(self) -> None:
        self.run()

    def to_dict(self):
        return {
            "label": self.label,
            "ready": self.ready,
            "connected": self.connected,
            "fully_connected": self.fully_connected,
            "inputs": self.inputs.to_dict(),
            "outputs": self.outputs.to_dict(),
            "signals": self.signals.to_dict(),
        }


class Fast(Function):
    """
    Like a regular node, but _all_ input channels _must_ have default values provided,
    and the initialization signature forces `run_on_updates` and
    `update_on_instantiation` to be `True`.
    """

    def __init__(
        self,
        node_function: callable,
        *output_labels: str,
        label: Optional[str] = None,
        run_on_updates=True,
        update_on_instantiation=True,
        parent: Optional[Workflow] = None,
        **kwargs,
    ):
        self.ensure_params_have_defaults(node_function)
        super().__init__(
            node_function,
            *output_labels,
            label=label,
            run_on_updates=run_on_updates,
            update_on_instantiation=update_on_instantiation,
            parent=parent,
            **kwargs,
        )

    @classmethod
    def ensure_params_have_defaults(cls, fnc: callable) -> None:
        """Raise a `ValueError` if any parameters of the callable lack defaults."""
        if any(
            param.default == inspect._empty
            for param in inspect.signature(fnc).parameters.values()
        ):
            raise ValueError(
                f"{cls.__name__} requires all function parameters to have defaults, "
                f"but {fnc.__name__} has the parameters "
                f"{inspect.signature(fnc).parameters.values()}"
            )


class SingleValue(Fast, HasChannel):
    """
    A fast node that _must_ return only a single value.

    Attribute and item access is modified to finally attempt access on the output value.
    """

    def __init__(
        self,
        node_function: callable,
        *output_labels: str,
        label: Optional[str] = None,
        run_on_updates=True,
        update_on_instantiation=True,
        parent: Optional[Workflow] = None,
        **kwargs,
    ):
        self.ensure_there_is_only_one_return_value(output_labels)
        super().__init__(
            node_function,
            *output_labels,
            label=label,
            run_on_updates=run_on_updates,
            update_on_instantiation=update_on_instantiation,
            parent=parent,
            **kwargs,
        )

    @classmethod
    def ensure_there_is_only_one_return_value(cls, output_labels):
        if len(output_labels) > 1:
            raise ValueError(
                f"{cls.__name__} must only have a single return value, but got "
                f"multiple output labels: {output_labels}"
            )

    @property
    def single_value(self):
        return self.outputs[self.outputs.labels[0]].value

    @property
    def channel(self) -> OutputData:
        """The channel for the single output"""
        return list(self.outputs.channel_dict.values())[0]

    def __getitem__(self, item):
        return self.single_value.__getitem__(item)

    def __getattr__(self, item):
        return getattr(self.single_value, item)

    def __repr__(self):
        return self.single_value.__repr__()

    def __str__(self):
        return f"{self.label} ({self.__class__.__name__}) output single-value: " + str(
            self.single_value
        )


def function_node(*output_labels: str, **node_class_kwargs):
    """
    A decorator for dynamically creating node classes from functions.

    Decorates a function.
    Takes an output label for each returned value of the function.
    Returns a `Function` subclass whose name is the camel-case version of the function node,
    and whose signature is modified to exclude the node function and output labels
    (which are explicitly defined in the process of using the decorator).
    """

    def as_node(node_function: callable):
        return type(
            node_function.__name__.title().replace("_", ""),  # fnc_name to CamelCase
            (Function,),  # Define parentage
            {
                "__init__": partialmethod(
                    Function.__init__,
                    node_function,
                    *output_labels,
                    **node_class_kwargs,
                )
            },
        )

    return as_node


def fast_node(*output_labels: str, **node_class_kwargs):
    """
    A decorator for dynamically creating fast node classes from functions.

    Unlike normal nodes, fast nodes _must_ have default values set for all their inputs.
    """

    def as_fast_node(node_function: callable):
        Fast.ensure_params_have_defaults(node_function)
        return type(
            node_function.__name__.title().replace("_", ""),  # fnc_name to CamelCase
            (Fast,),  # Define parentage
            {
                "__init__": partialmethod(
                    Fast.__init__,
                    node_function,
                    *output_labels,
                    **node_class_kwargs,
                )
            },
        )

    return as_fast_node


def single_value_node(*output_labels: str, **node_class_kwargs):
    """
    A decorator for dynamically creating fast node classes from functions.

    Unlike normal nodes, fast nodes _must_ have default values set for all their inputs.
    """

    def as_single_value_node(node_function: callable):
        SingleValue.ensure_there_is_only_one_return_value(output_labels)
        SingleValue.ensure_params_have_defaults(node_function)
        return type(
            node_function.__name__.title().replace("_", ""),  # fnc_name to CamelCase
            (SingleValue,),  # Define parentage
            {
                "__init__": partialmethod(
                    SingleValue.__init__,
                    node_function,
                    *output_labels,
                    **node_class_kwargs,
                )
            },
        )

    return as_single_value_node
