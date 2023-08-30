"""
Meta nodes are callables that create a node class instead of a node instance.
"""

from __future__ import annotations

from functools import partialmethod
import os
from typing import Any, Optional

from pyiron_contrib.workflow.function import (
    Function,
    SingleValue,
    function_node,
    single_value_node,
)
from pyiron_contrib.workflow.macro import Macro, macro_node
from pyiron_contrib.workflow.node import Node
from pyiron_contrib.workflow.util import DotDict


def list_to_output(length: int, **node_class_kwargs) -> type[Function]:
    """
    A meta-node that returns a node class with `length` input channels and
    maps these to a single output channel with type `list`.
    """

    def _list_to_many(length: int):
        template = f"""
def __list_to_many(l: list):
    {"; ".join([f"out{i} = l[{i}]" for i in range(length)])}
    return [{", ".join([f"out{i}" for i in range(length)])}]
        """
        exec(template)
        return locals()["__list_to_many"]

    return function_node(**node_class_kwargs)(_list_to_many(length=length))


def input_to_list(length: int, **node_class_kwargs) -> type[SingleValue]:
    """
    A meta-node that returns a node class with `length` output channels and
    maps an input list to these.
    """

    def _many_to_list(length: int):
        template = f"""
def __many_to_list({", ".join([f"inp{i}=None" for i in range(length)])}):
    return [{", ".join([f"inp{i}" for i in range(length)])}]
        """
        exec(template)
        return locals()["__many_to_list"]

    return single_value_node(**node_class_kwargs)(_many_to_list(length=length))


def for_loop(
    loop_body_class: type[Node],
    length: int,
    iterate_on: str | tuple[str] | list[str],
    # TODO:
) -> type[Macro]:
    """
    An _extremely rough_ first draft of a for-loop meta-node.

    Takes a node class, how long the loop should be, and which input(s) of the provided
    node class should be looped over (given as strings of the channel labels) and
    builds a macro that
    - Makes copies of the provided node class, i.e. the "body node"
    - For each input channel specified to "loop over", creates a list-to-many node and
      connects each of its outputs to their respective body node inputs
    - For all other inputs, makes a 1:1 node and connects its output to _all_ of the
      body nodes
    - Relables the macro IO to match the passed node class IO so that list-ified IO
      (i.e. the specified input and all output) is all caps

    Examples:
        >>> bulk_loop = for_loop(
        ...     Workflow.create.atomistics.Bulk,
        ...     5,
        ...     iterate_on = ("a",),
        ... )()
        >>>
        >>> [
        ...     struct.cell.volume for struct in bulk_loop(
        ...         name="Al",  # Sent equally to each body node
        ...         A=np.linspace(3.9, 4.1, 5).tolist(),  # Distributed across body nodes
        ...     ).STRUCTURE
        ... ]
        [14.829749999999995,
         15.407468749999998,
         15.999999999999998,
         16.60753125,
         17.230249999999995]

    TODO:
        - Refactor like crazy, it's super hard to read and some stuff is too hard-coded
        - Give some sort of access to flow control??
        - How to handle passing executors to the children? Maybe this is more
          generically a Macro question?
        - Is it possible to somehow dynamically adapt the held graph depending on the
          length of the input values being iterated over? Tricky to keep IO well defined
        - Allow a different mode, or make a different meta node, that makes all possible
          pairs of body nodes given the input being looped over instead of just `length`
        - Provide enter and exit magic methods so we can `for` or `with` this fancy-like
    """
    iterate_on = [iterate_on] if isinstance(iterate_on, str) else iterate_on

    def make_loop(macro):
        macro.inputs_map = {}
        macro.outputs_map = {}
        body_nodes = []

        # Parallelize over body nodes
        for n in range(length):
            body_nodes.append(
                macro.add(loop_body_class(label=f"{loop_body_class.__name__}_{n}"))
            )

        # Make input interface
        for label, inp in body_nodes[0].inputs.items():
            # Don't rely on inp.label directly, since inputs may be a Composite IO
            # panel that has a different key for this input channel than its label

            # Scatter a list of inputs to each node separately
            if label in iterate_on:
                interface = list_to_output(length)(
                    parent=macro,
                    label=label.upper(),
                    output_labels=[
                        f"{loop_body_class.__name__}__{inp.label}_{i}"
                        for i in range(length)
                    ],
                    l=[inp.default] * length,
                )
                # Connect each body node input to the input interface's respective output
                for body_node, out in zip(body_nodes, interface.outputs):
                    body_node.inputs[label] = out
                macro.inputs_map[f"{interface.label}__l"] = interface.label
                # TODO: Don't hardcode __l
            # Or distribute the same input to each node equally
            else:
                interface = macro.create.standard.UserInput(
                    label=label, output_labels=label, user_input=inp.default
                )
                for body_node in body_nodes:
                    body_node.inputs[label] = interface
                macro.inputs_map[f"{interface.label}__user_input"] = interface.label
                # TODO: Don't hardcode __user_input

        # Make output interface: outputs to lists
        for label, out in body_nodes[0].outputs.items():
            interface = input_to_list(length)(
                parent=macro,
                label=label.upper(),
                output_labels=f"{loop_body_class.__name__}__{label}",
            )
            # Connect each body node output to the output interface's respective input
            for body_node, inp in zip(body_nodes, interface.inputs):
                inp.connect(body_node.outputs[label])
            macro.outputs_map[
                f"{interface.label}__{loop_body_class.__name__}__{label}"
            ] = interface.label
            # TODO: Don't manually copy the output label construction

    return macro_node()(make_loop)


def while_loop(
    loop_body_class: type[Node],
) -> type[Macro]:
    """
    An _extremely rough_ first draft of a for-loop meta-node.

    Takes a node class and builds a macro that makes a cyclic signal connection between
    that body node and an "if" node, i.e. when the body node finishes it runs the
    if-node, and when the if-node finishes and evaluates `True` then it runs the body
    node.
    The if-node condition is exposed as input on the resulting macro with the label
    "condition", but it is left to the user to connect...
    - The condition to some output of another node, either an internal node of the body
        node (if it's a macro) or any other node in the workflow
    - The (sub)input of the body node to the (sub)output of the body node, so it
      actually does something different at each iteration

    Args:
        loop_body_class (type[pyiron_contrib.workflow.node.Node]): The class for the
            body of the while-loop.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(0)  # Just for docstring tests, so the output is predictable
        >>>
        >>> from pyiron_contrib.workflow import Workflow
        >>>
        >>> # Build tools
        >>>
        >>> @Workflow.wrap_as.single_value_node()
        >>> def random(length: int | None = None):
        ...     random = np.random.random(length)
        ...     return random
        >>>
        >>> @Workflow.wrap_as.single_value_node()
        >>> def greater_than(x: float, threshold: float):
        ...     gt = x > threshold
        ...     symbol = ">" if gt else "<="
        ...     print(f"{x:.3f} {symbol} {threshold}")
        ...     return gt
        >>>
        >>> RandomWhile = Workflow.create.meta.while_loop(random)
        >>>
        >>> # Define workflow
        >>>
        >>> wf = Workflow("random_until_small_enough")
        >>>
        >>> ## Wire together the while loop and its condition
        >>>
        >>> wf.gt = greater_than()
        >>> wf.random_while = RandomWhile(condition=wf.gt)
        >>> wf.gt.inputs.x = wf.random_while.Random
        >>>
        >>> wf.starting_nodes = [wf.random_while]
        >>>
        >>> ## Give convenient labels
        >>> wf.inputs_map = {"gt__threshold": "threshold"}
        >>> wf.outputs_map = {"random_while__Random__random": "capped_value"}
        >>> # Set a threshold and run
        >>>
        >>> print(f"Finally {wf(threshold=0.1).capped_value:.3f}")
        0.549 > 0.1
        0.715 > 0.1
        0.603 > 0.1
        0.545 > 0.1
        0.424 > 0.1
        0.646 > 0.1
        0.438 > 0.1
        0.892 > 0.1
        0.964 > 0.1
        0.383 > 0.1
        0.792 > 0.1
        0.529 > 0.1
        0.568 > 0.1
        0.926 > 0.1
        0.071 <= 0.1
        Finally 0.071

        We can also loop data _internally_ in the body node.
        In such cases, we can still _initialize_ the data to some other value, because
        this has no impact on the connections -- so for the first run of the body node
        we wind up using to initial value, but then the body node pushes elements of its
        own output back to its own input for future runs.
        E.g.)
        >>> @Workflow.wrap_as.single_value_node(run_on_updates=False)
        >>> def add(a, b):
        ...     print(f"Adding {a} + {b}")
        ...     return a + b
        >>>
        >>> @Workflow.wrap_as.single_value_node()
        >>> def less_than_ten(value):
        ...     return value < 10
        >>>
        >>> AddWhile = Workflow.create.meta.while_loop(add)
        >>>
        >>> wf = Workflow("do_while")
        >>> wf.lt10 = less_than_ten()
        >>> wf.add_while = AddWhile(condition=wf.lt10)
        >>>
        >>> wf.lt10.inputs.value = wf.add_while.Add
        >>> wf.add_while.Add.inputs.a = wf.add_while.Add
        >>>
        >>> wf.starting_nodes = [wf.add_while]
        >>> wf.inputs_map = {
        ...     "add_while__Add__a": "a",
        ...     "add_while__Add__b": "b"
        ... }
        >>> wf.outputs_map = {"add_while__Add__a + b": "total"}
        >>> response = wf(a=1, b=2)
        >>> print(response.total)
        11

        Note that we needed to specify a starting node because in this case our
        graph is cyclic and _all_ our nodes have connected input! We obviously cannot
        automatically detect the "upstream-most" node in a circle!
    """

    def make_loop(macro):
        body_node = macro.add(loop_body_class(label=loop_body_class.__name__))
        macro.create.standard.If(label="if_", run_on_updates=False)

        # Create a cyclic loop between body and if nodes, so that they will keep
        # triggering themselves until the if evaluates false
        body_node.signals.input.run = macro.if_.signals.output.true
        macro.if_.signals.input.run = body_node.signals.output.ran
        macro.starting_nodes = [body_node]

        # Just for convenience:
        macro.inputs_map = {"if___condition": "condition"}

    return macro_node()(make_loop)


def python_script(
        script: str,
        args: Optional[dict[str, type[Any]]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        output_files: Optional[dict[str, str]] = None
):
    """
    Creates a node class that runs a particular python script.
    Output (assumed to be files), command line args, and command line kwargs
    can all be set up as channels from/to the node.
    """
    args = {} if args is None else args
    kwargs = {} if kwargs is None else kwargs
    output_files = {} if output_files is None else output_files

    def make_graph(macro):
        macro.inputs_map = {}
        macro.outputs_map = {}

        InputCollator = macro.create.meta.input_to_list(len(args) + len(kwargs))
        macro.input_collator = InputCollator(
            output_labels="output",
            update_on_instantiation=False,
        )
        i = 0

        for positional_label, type_hint in args.items():
            node = macro.create.standard.UserInput(
                label=positional_label,
                update_on_instantiation=False
            )
            node.inputs.user_input.type_hint = type_hint
            macro.inputs_map[f"{positional_label}__user_input"] = positional_label
            macro.input_collator.inputs[f"inp{i}"] = node
            i += 1

        keyword_input_nodes = []
        for keyword_label, default in kwargs.items():
            clean_label = keyword_label.lstrip("-")
            user_node = macro.create.standard.UserInput(user_input=default,
                                                        label=clean_label)
            macro.inputs_map[f"{clean_label}__user_input"] = clean_label

            @macro.wrap_as.single_value_node()
            def kwarg_creator(value):
                as_string = keyword_label + "=" + str(value)
                return as_string

            internal_node = kwarg_creator(user_node, label=keyword_label, parent=macro)
            macro.input_collator.inputs[f"inp{i}"] = internal_node
            i += 1

        @macro.wrap_as.single_value_node()
        def inputs_to_command(inputs_list: list):
            inputs_list = [str(i) for i in inputs_list]
            return "python " + script + " " + " ".join(inputs_list)

        macro.command = inputs_to_command(macro.input_collator)
        macro.script = macro.create.standard.shell_command(cmd=macro.command)
        node_directory = str(macro.script.working_directory.path.absolute())
        macro.script.inputs.cwd = node_directory

        for channel_label, filename in output_files.items():
            fout = macro.create.standard.UserInput(
                os.path.join(filename, node_directory),
                label=channel_label,
                run_on_updates=False,
            )
            fout.signals.input.run = macro.script.signals.output.ran
            macro.outputs_map[f"{channel_label}__user_input"] = channel_label

        for process_output in macro.script.outputs.labels:
            macro.outputs_map[
                f"{macro.script.label}__{process_output}"] = process_output

    return type(
        script.title().replace("_", "").replace(" ", ""),  # fnc_name to CamelCase
        (Macro,),  # Define parentage
        {
            "__init__": partialmethod(
                Macro.__init__,
                make_graph,
            )
        },
    )


meta_nodes = DotDict(
    {
        for_loop.__name__: for_loop,
        input_to_list.__name__: input_to_list,
        list_to_output.__name__: list_to_output,
        python_script.__name__: python_script,
        while_loop.__name__: while_loop,
    }
)
