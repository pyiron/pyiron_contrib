from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

from pyiron_contrib.workflow.function import single_value_node


@single_value_node(output_labels="select")  # dynamic labeling would be nice
def select(
    data = None, key: str = None
):
    return data.__getattribute__(key)


@single_value_node(output_labels="fig")
def myplot(
    x: Optional[list | np.ndarray] = None,
    y: Optional[list | np.ndarray] = None,
    title: str = None
):
    if title is not None:
        plt.title(title)
    if x is None:
        x = np.arange(len(y))
    return plt.plot(x, y)


@single_value_node(output_labels="fig")
def scatter(
    x: Optional[list | np.ndarray] = None, y: Optional[list | np.ndarray] = None
):
    return plt.scatter(x, y)


@single_value_node()
def user_input(user_input):
    return user_input


nodes = [
    scatter,
    myplot,
    select,
    user_input,
]
