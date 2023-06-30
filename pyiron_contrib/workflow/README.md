# Workflows

`pyiron_contrib.workflow` is an incubator module for a new pyiron paradigm using graph-based computational diagrams.
The objective of this paradigm is to hold on to pyiron's existing power, while improving (a) maintainability and (b) extensibility by creating a better separation between functionality (`GenericJob` in v0 of pyiron, `Node` here) from the underlying workflow management.
Unlike existing workflow management tools we are aware of, we plan to provide support for _cyclic_ graphs and HPC compute resources _foundationally_.

This code is under active development and subject to sudden changes.

## Concepts

In the parlance of graph-based computational workflows, we use the term "node" to be a point on the graph at which something is computed.
Each node has "channels" which carry information into and out of the node.
This information can be "data" of any type, i.e. the necessary input for a node to perform its computation, or the output result, or a "signal", which is just a pulse of information connected to the nodes to trigger functionality -- e.g. at the most basic level a node has a `run` input signal which instructs the node to start its computation, and a `ran` output signal which fires when. 
In the current implementation, execution flow is push-based, using both the signals described above and by output data channels updating the values of input data channels. 
By forming connections between different channels, nodes form a computational graph.

Data channels support type hints.
These do not prevent value assignment, but do prohibit connections between channels whose type hints are incompatible (this can be turned off to allow arbitrary connections).
In this case, "compatible" means that the type hint of the sending channel (`OutputData`) should be as-or-more specific than the type hint of the receiving channel (`InputData`).
These type hints _do_ get used to check the "readiness" of a channel, such that a channel whose value does not match its hint is not `ready`.
Further, the initial value of all data channels is set to a special `NotData` class, which forces the channel to be not ready regardless of the presence or absense of type hinting.
This readiness then gets used further down the pipeline, where nodes are in turn `ready` if and only if _all_ of their input channels are `ready`, and readiness filters whether a node that receives an `update()` call will actually proceed to a `run()` call.

(Not implemented) Nodes can be grouped together in macros, which appear to the outside as a single node, but internally have their own graph structure.
Such macro construction can be nested arbitrarily deeply, allowing for powerful abstraction of workflows.

(Not implemented) The actual computation each node performs is packaged as a "task". This "task", along with its necessary input, can either run on the same python process that is aware of the workflow, or it can be shipped off to an "executor" for (potentially parallel) computation.
A callback is registered with the executor building a connection between the node and executor (which may use its resources in service of multiple nodes).
In case the process controlling the workflow is shut down, the executor can serialize the result of the calculation.
When a workflow is restarted, each node will attempt to reconnect with its executor (if any), and/or read the serialized executor output if the execution of the task has finished in the meantime.
In this way, workflows with long durations can be shut down and re-instantiated in a state of partial completion.

All of this allows for a separation of the workflow infrastructure, and the functionality of specific workflows that is defined by specific node functionality and workflow/macro topologies.


## Structure

This is a rough sketch of the code structure.
It is not exhaustive (there are other attributes and methods), and some of it is not yet implemented/still has different names.
It is just to provide a discussion point for devs currently implementing this project, so we can plan things and be using the same language.
Ultimately, one will be able to automatically extract this information from the code itself and the bulk of this section will just be a UML diagram.

- `Channel(ABC)`
  - Has a `label: str` name
  - Can form `connections: list[Channel]` with other channels
  - Belongs to a `parent: Node`
  - `Data(Channel, ABC)`
    - Stores a `value: Any`, which defaults to `NotData`
    - Has a concept of `ready`, which can only be true if the `value is not NotData`
    - Can store type hints to further refine readiness and control connection permissibility
    - (Not implemented:) Can store ontological type information
    - `DataInput(Data)`
      - Can connect to `DataOutput` channels
      - Is `ready: Bool` when `value` matches typing specifications (if any) and is not `waiting_for_update: Bool`
      - On `update()`, takes the new `value` and attempts to `.update()` its `parent` and sets `waiting_for_update` to `False`
      - The update waiting flag allows nodes to synchronize the update of multiple input values before the input collection is all ready
    - `DataOutput(Data)`
      - Can connect to `DataInput` channels
      - On `update()`, uses its `value` to `update` all `DataInput` instances in its `connections`
  - `Signal(Channel, ABC)`
    - Implements a `__call__` method
    - `InputSignal(Signal)`
      - Can connect to `OutputSignal`
      - Takes a `callback: callable` with no arguments (intended to be a method of the `parent`)
      - On `__call__`, invokes its `callback()`
    - `OutputSignal(Signal)`
      - Can connect to `InputSignal`
      - On `__call__` calls each `InputSignal` in its `connections`
- `IO(ABC)`
  - A container class to hold multiple `Channel` objects
  - Gives quality-of-life features: 
    - dot-access to owned objects
    - `connected: bool`/`fully_connected: bool` attributes that take `any`/`all` of corresponding owned channel values for these properties
    - `disconnect()` to disconnect _all_ owned channels at once
    - Provides `__len__`
    - etc.
  - Overrides `__setattr__`:
    - Assigning another `Channel` to an existing `Channel` attempts to make a connections between them
    - Assigning any other value must be specified in children
  - `Data(IO, ABC)`, `InputData(Data)`, `OutputData(Data)`
    - Overrides `__setattr__`: uses the channels `update` method for all non-channel assignments to existing channels
  - `Signal(IO, ABC)`, `InputSignals(Signal)`, `OutputSignals(Signal)`
    - Overrides `__setattr__`: raises an error when a non-channel value is assigned to
- `Signals`
  - A very simply container class holding an `InputSignals` and `OutputSignals` instance
`Node(ABC)`
  - An abstract base for all nodes that will live on the computation graph
  - An attempt is made to interpret unused `kwargs` as initial values for the `input` (including allowing passing other `Channel` objects to directly form a connection)
  - May optionally `run()` at the end of instantiation
  - Provides:
    - `label: str`
    - `signals: Signals`
      - may be extended by adding to the input or output signals later in `__init__` of children
    - `status: ???` to define the current operational state (not implemented)
    - `fully_/connected: bool`
    - `disconnect()`
    - `run()`
      - Controls status
      - Handles execution (currently on the main process or an error)
      - Invokes `on_run()` with `run_args` (not yet implemented: or sends them to an executor)
      - Invokes `finish_run()` with the result of `on_run()` (not yet implemented: or registers as a callback with the executor) 
    - `update()` calls `run()` if `input.ready: bool` and `run_on_updates: bool`
      - I don't think we need a `status` check as that can probably be fully handled in `run()`
    - `finish_run(run_output: tuple)`
      - Controls status and fires the `ran` signal
      - Invokes `process_run_output()` with the `run_output`
  - Demands from children:
    - `input: InputData`
    - `output: OutputData`
    - `on_run: callable[..., tuple]`
      - May take `self: Node` as the 0th argument to reference this node
    - `run_args: dict`
    - `process_run_result(run_output: tuple)`
  - Not yet implemented:
    - A connection to an `Executor`
      - The ability to deserialize executor output to file in case it wasn't available to be called back when the executor finished
    - Serialization
      - Including executor instance data in case a reconnection needs to be made
    - White-listing(?) what data should be stored on serialization 
  - `Function(Node)`
    - Wraps an arbitrary python function call assigned to `on_run` 
    - `input` and (not implemented yet) `output` channels automatically created, named, and typed from the function
    - `run_args` automatically generated from the `input`
    - Best practices: node function should be functional and idempotent
    - `Slow(Function)`
      - Just like the parent except `run_on_updates` and `update_on_instantiation` default to `False`.
        - Useful for expensive functions where you don't want them running accidentally as input gets updated
    - `SingleValue(Function)`
      - A function node that _must_ return only a single value.
      - Gives special access to its output
        - Attribute and item access is modified to finally attempt access on the output value
        - Has a `channel` attribute pointing to its single output channel, allowing the node to be used _directly_ when making new channel connections
  - `Composite(Node)` (not implemented, replaces the current `HasNodes` mix-in)
    - A class of nodes that store and run a sub-graph of `nodes: list[Node]`
    - `add` and `add(node: Node)`
      - A callable class attribute that lets you add to `nodes` by either directly providing a node or accessing constructors for any registered node packages
    - Holds a class method `wrap_as` that gives access to other node classes, e.g. `HasNodes.wrap_as.FastNode(some_callable)` 
      - This is critical for the `Workflow` child class so it can function as a single point of entry for notebook users who want to make new node classes in their jupyter notebook, and helpful for `Macro` devs who want to define new nodes for their macro in some .py file.
    - `on_run()` by default, invokes `run()` on all held `nodes` without input connections, but can be overridden by specifying some other list of nodes to run (maybe??? I'm not sure what this should do)
    - `Macro(Composite)` (not implemented)
      - Holds a statically defined graph with a static IO interface
      - Intended to be sub-classed, where a single computational graph is defined at the child class level and all instances have the same graph (with different interface connections and data values throughout, obviously)
      - `input` and `output` are defined statically to provide particular access to the IO of held nodes
      - Default is to give access to unconnected IO, but this can be explicitly specified instead.
      - Required
        - `???() -> DotDict[Nodes]` builds the nodes to be held, assigns any non-standard 
          - Don't forget to also make any necessary internal `connections` between these nodes
          - Gets invoked during initialization to populate `nodes: list[Node]`
      - Provided
        - `???() -> ???` constructs the macro's IO panel by creating links to node IO (all unconnected IO by default, but can be overridden in child classes)
    - `Workflow(Composite)`
      - This is our single-point of entry for imports!
      - Holds a _dynamically_ defined collections of nodes
      - Not intended to be sub-classed, rather should be instantiated and the graph should be modified on the instance directly
      - `input` and `output` are dynamically generated on request from the _unconnected_ IO of held nodes (`signals` still belongs to the workflow object itself)
      - `to_macro(class_name: Optional[str] = None) -> type[Macro]` takes the current `nodes` with all their connections, the current IO, and current node IO values as defaults, and dynamically creates a `Macro` from it; this locks-in the current `Workflow` behaviour and packages it for use as a node in other workflows
        - The new node class takes its name from the workflow `label` by default, but can be explicitly specified
