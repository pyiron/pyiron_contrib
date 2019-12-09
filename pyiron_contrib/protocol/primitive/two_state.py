# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from pyiron_mpie.flexible.protocol.generic import PrimitiveVertex
import numpy as np

"""
Primitive protocols which have two outbound execution edges.
"""


class BoolVertex(PrimitiveVertex):
    """
    This is a class of commands designed to branch the graph based on a binary check. They do not have

    Attributes
    """
    def __init__(self, name=None):
        super(BoolVertex, self).__init__(name=name)
        self.possible_vertex_states = ["true", "false"]
        self.vertex_state = "false"

    def run(self):
        self.command(**self.input.resolve())


class IsGEq(BoolVertex):
    """
    Checks if an input value is greater than or equal to a target threshold. Vertex state switches from 'false' to
    'true' when the target exceeds the threshold.

    Input attributes:
        target (float/int): The value being checked. (Default is numpy infinity.)
        threshold (float/int): What it's being checked against. (Default is zero.)
    """
    def __init__(self, name=None):
        super(IsGEq, self).__init__(name=name)
        self.input.default.target = np.inf
        self.input.default.threshold = 0

    def command(self, target=np.inf, threshold=0):
        if target >= threshold:
            self.vertex_state = "true"
        else:
            self.vertex_state = "false"


class IsLEq(BoolVertex):
    """
    Checks if an input value is less than or equal to a target threshold. Vertex state switches from 'false' to
    'true' when the target exceeds the threshold.

    Input attributes:
        target (float/int): The value being checked. (Default is zero.)
        threshold (float/int): What it's being checked against. (Default is numpy infinity.)
    """
    def __init__(self, name=None):
        super(IsLEq, self).__init__(name=name)
        self.input.default.target = np.inf
        self.input.default.threshold = 0

    def command(self, target, threshold):
        if target <= threshold:
            self.vertex_state = "true"
        else:
            self.vertex_state = "false"


class ModIsZero(BoolVertex):
    """
    Checks if the target value mod some number is zero.

    Input attributes:
        target (int): The value being checked.
        mod (int): The modulo to use.
    """

    def command(self, target, mod):
        if target % mod == 0:
            self.vertex_state = "true"
        else:
            self.vertex_state = "false"