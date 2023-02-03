from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from pyiron_contrib.workflow.node import Node


class ChannelTemplate:
    def __init__(
            self,
            name: str,
            default: Optional[Any] = None,
            types: Optional[tuple | type[Any]] = None
    ):
        self.name = name
        self.default = default
        self.types = types

    def _to_IOChannel(self, node: Node, class_: type[Channel]) -> Channel:
        return class_(
            name=self.name,
            node=node,
            default=deepcopy(self.default),
            types=self.types,
        )

    def to_input(self, node: Node) -> InputChannel:
        return self._to_IOChannel(node=node, class_=InputChannel)

    def to_output(self, node: Node) -> OutputChannel:
        return self._to_IOChannel(node=node, class_=OutputChannel)


class Channel(ABC):
    def __init__(
            self,
            name: str,
            node: Node,
            default: Optional[Any] = None,
            types: Optional[tuple] = None
    ):
        self.name = name
        self.node = node
        self.default = default
        self.value = default
        self.types = None if types is None else self._types_to_tuple(types)
        self.connections = []

    @staticmethod
    def _types_to_tuple(types):
        if isinstance(types, tuple):
            return types
        elif isinstance(types, list):
            return tuple(types)
        else:
            return (types,)

    @property
    def ready(self):
        if self.types is not None:
            return isinstance(self.value, self.types)
        else:
            return True

    def connect(self, *others: Channel):
        for other in others:
            if self._valid_connection(other):
                self.connections.append(other)
                other.connections.append(self)
            else:
                if isinstance(other, Channel):
                    warn(
                        f"{self.name} ({self.__class__.__name__}) and {other.name} "
                        f"({other.__class__.__name__}) were not a valid connection"
                    )
                else:
                    raise TypeError(
                        f"Can only connect two channels, but {self.name} "
                        f"({self.__class__.__name__}) got a {other} ({type(other)})"
                    )

    def _valid_connection(self, other):
        if self._is_IO_pair(other) and not self._already_connected(other):
            if self._both_typed(other):
                return self._output_types_are_subset_of_input_types(other)
            else:
                return True
        else:
            return False

    def _is_IO_pair(self, other: Channel):
        return isinstance(other, Channel) and type(self) != type(other)

    def _already_connected(self, other: Channel):
        return other in self.connections

    def _both_typed(self, other: Channel):
        return self.types is not None and other.types is not None

    def _output_types_are_subset_of_input_types(self, other: Channel):
        out, inp = self._figure_out_who_is_who(other)
        return len(set(out.types).difference(inp.types)) == 0

    def _figure_out_who_is_who(self, other: Channel) -> (OutputChannel, InputChannel):
        return (self, other) if isinstance(self, OutputChannel) else (other, self)

    def disconnect(self, *others: Channel):
        for other in others:
            if other in self.connections:
                self.connections.remove(other)
                other.disconnect(self)

    @property
    def connected(self):
        return len(self.connections) > 0


class InputChannel(Channel):
    def update(self, value):
        self.value = value
        self.node.update()


class OutputChannel(Channel):
    def update(self, value):
        self.value = value
        for inp in self.connections:
            inp.update(self.value)