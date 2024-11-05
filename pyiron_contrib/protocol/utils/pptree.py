# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.


import sys

"""
this module provides a function to nicely print the whitelists
"""


__author__ = "Dominik Gehringer, Liam Huber"
__copyright__ = (
    "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Dec 12, 2019"


def count_paths(node):
    """
    Counts the entries in a nested dectionaries
    Args:
        node: (dict) the nested dictionary

    Returns: (int) the number of total leaf nodes

    """
    if not isinstance(node, dict):
        return 1
    else:
        s = 0
        for _, v in node.items():
            s += count_paths(v)
        return s


def print_tree(current_node, indent="", last="downup", name="root", file=sys.stdout):
    """
    Prints a nested dictionary in pretty tree
    This code is adapted version found in "https://github.com/clemtoy/pptree/blob/master/pptree/pptree.py"

    Args:
        current_node: (dict) as nested dictonaries, its keys will be converted to string in order to print the tree
        indent: (str) optional indentation
        last: (str) "downdown", "upup", "downup" or "updown" the direction of the last shape (default: "downup")
        name: (str) the name of the root node
        file: (file) location where to print the output (default: sys.stdout)

    Returns:

    """
    children = lambda node: (
        [
            (k, v) if isinstance(v, dict) else ("%s: %s" % (k, v), None)
            for k, v in node.items()
        ]
        if isinstance(node, dict)
        else []
    )
    nb_children = (
        lambda node: sum(count_paths(child) for childname, child in children(node)) + 1
    )
    size_branch = {
        childname: nb_children(child) for (childname, child) in children(current_node)
    }
    # Creation of balanced lists for "up" branch and "down" branch
    up = sorted(children(current_node), key=lambda node: nb_children(node[0]))
    down = []
    while up and sum(size_branch[downname] for downname, downchild in down) < sum(
        size_branch[upname] for upname, upchild in up
    ):
        down.append(up.pop())

    # print the "up" branch
    for childname, child in up:
        next_last = "up" if up.index((childname, child)) == 0 else ""
        next_indent = "{0}{1}{2}".format(
            indent, " " if "up" in last else "│", " " * len(name)
        )
        print_tree(child, next_indent, next_last, name=childname, file=file)

    # Printing of current node
    if last == "up":
        start_shape = "┌"
    elif last == "down":
        start_shape = "└"
    elif last == "updown":
        start_shape = " "
    else:
        start_shape = "├"

    if up:
        end_shape = "┤"
    elif down:
        end_shape = "┐"
    else:
        end_shape = ""

    print("{0}{1}{2}{3}".format(indent, start_shape, name, end_shape), file=file)

    # Printing of "down" branch
    for childname, child in down:
        next_last = "down" if down.index((childname, child)) is len(down) - 1 else ""
        next_indent = "{0}{1}{2}".format(
            indent, " " if "down" in last else "│", " " * len(name)
        )
        print_tree(child, next_indent, next_last, name=childname, file=file)
