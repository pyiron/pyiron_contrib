# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Alternative structure container that stores them in flattened arrays.
"""

from itertools import chain

import numpy as np
import h5py

from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure

class StructureContainer(HasStructure):
    """
    Class that can write and read lots of structures from and to hdf quickly.

    This is done by storing positions, cells, etc. into large arrays instead of writing every structure into a new
    group.  Structures are stored together with an identifier that should be unique.  The class can be initialized with
    the number of structures and the total number of atoms in all structures, but re-allocates memory as necessary when
    more (or larger) structures are added than initially anticipated.
    """

    def __init__(self, num_structures=1, num_atoms=1):
        """
        Create new structure container.

        Args:
            num_structures (int): pre-allocation for per structure arrays
            num_atoms (int): pre-allocation for per atoms arrays
        """
        # tracks allocated versed as yet used number of structures/atoms
        self._num_structures_alloc = self.num_structures = num_structures
        self._num_atoms_alloc = self.num_atoms = num_atoms
        # store the starting index for properties with unknown length
        self.current_atom_index = 0
        # store the index for properties of known size, stored at the same index as the structure
        self.current_structure_index = 0
        # Also store indices of structure recently added
        self.prev_structure_index = 0
        self.prev_atom_index = 0

        self._init_arrays()

    def __len__(self):
        return self.current_structure_index

    def _init_arrays(self):
        self._per_atom_arrays = {
                # 2 character unicode array for chemical symbols
                "symbols": np.full(self._num_atoms_alloc, "XX", dtype=np.dtype("U2")),
                "positions": np.empty((self._num_atoms_alloc, 3))
        }

        self._per_structure_arrays = {
                "start_indices": np.empty(self._num_structures_alloc, dtype=np.int32),
                "len_current_struct": np.empty(self._num_structures_alloc, dtype=np.int32),
                "identifiers": np.empty(self._num_structures_alloc, dtype=np.dtype("U20")),
                "cells": np.empty((self._num_structures_alloc, 3, 3))
        }

    @property
    def symbols(self):
        return self._per_atom_arrays["symbols"]

    @property
    def positions(self):
        return self._per_atom_arrays["positions"]

    @property
    def start_indices(self):
        return self._per_structure_arrays["start_indices"]

    @property
    def len_current_struct(self):
        return self._per_structure_arrays["len_current_struct"]

    @property
    def identifiers(self):
        return self._per_structure_arrays["identifiers"]

    @property
    def cells(self):
        return self._per_structure_arrays["cells"]

    def get_array(self, name, index):
        if name in self._per_atom_arrays:
            I = self._per_structure_arrays["start_indices"][index]
            E = I + self._per_structure_arrays["len_current_struct"][index]
            return self._per_atom_arrays[name][I:E]
        elif name in self._per_structure_arrays:
            return self._per_structure_arrays[name][index]
        else:
            raise KeyError(f"no array named {name} defined on StructureContainer")

    def _resize_atoms(self, new):
        self._num_atoms_alloc = new
        for k, a in self._per_atom_arrays.items():
            new_shape = (new,) + a.shape[1:]
            try:
                a.resize(new_shape)
            except ValueError:
                self._per_atom_arrays[k] = np.resize(a, new_shape)

    def _resize_structures(self, new):
        self._num_structures_alloc = new
        for k, a in self._per_structure_arrays.items():
            new_shape = (new,) + a.shape[1:]
            try:
                a.resize(new_shape)
            except ValueError:
                self._per_structure_arrays[k] = np.resize(a, new_shape)

    def add_array(self, name, shape=(), dtype=np.float64, fill=None, per="atom"):
        if per == "atom":
            shape = (self._num_atoms_alloc,) + shape
            store = self._per_atom_arrays
        elif per == "structure":
            shape = (self._num_structures_alloc,) + shape
            store = self._per_structure_arrays
        else:
            raise ValueError(f"per must \"atom\" or \"structure\", not {per}")

        if fill is None:
            store[name] = np.empty(shape=shape, dtype=dtype)
        else:
            store[name] = np.full(shape=shape, fill_value=fill, dtype=dtype)

    def add_structure(self, structure, identifier, **arrays):
        n = len(structure)
        new_atoms = self.current_atom_index + n

        if new_atoms > self._num_atoms_alloc:
            self._resize_atoms(max(new_atoms, self._num_atoms_alloc * 2))
        if self.current_structure_index + 1 > self._num_structures_alloc:
            self._resize_structures(self._num_structures_alloc * 2)

        if new_atoms > self.num_atoms:
            self.num_atoms = new_atoms
        if self.current_structure_index + 1 > self.num_structures:
            self.num_structures += 1

        # len of structure to index into the initialized arrays
        i = self.current_atom_index + n

        self._per_atom_arrays["symbols"][self.current_atom_index:i] = np.array(structure.symbols)
        self._per_atom_arrays["positions"][self.current_atom_index:i] = structure.positions

        self._per_structure_arrays["len_current_struct"][self.current_structure_index] = n
        self._per_structure_arrays["cells"][self.current_structure_index] = structure.cell.array
        self._per_structure_arrays["start_indices"][self.current_structure_index] = self.current_atom_index
        self._per_structure_arrays["identifiers"][self.current_structure_index] = identifier

        for k, a in arrays.items():
            a = np.asarray(a)
            if len(a.shape) > 0 and a.shape[0] == n:
                if k not in self._per_atom_arrays:
                    self.add_array(k, shape=a.shape[1:], dtype=a.dtype, per="atom")
                self._per_atom_arrays[k][self.current_atom_index:i] = a
            else:
                if len(a.shape) > 0 and a.shape[0] == 1:
                    a = a[0]
                if k not in self._per_structure_arrays:
                    self.add_array(k, shape=a.shape, dtype=a.dtype, per="structure")
                self._per_structure_arrays[k][self.current_structure_index] = a

        self.prev_structure_index = self.current_structure_index
        self.prev_atom_index = self.current_atom_index

        # Set new current_atom_index and increase current_structure_index
        self.current_structure_index += 1
        self.current_atom_index = i
        #return last_structure_index, last_atom_index


    def to_hdf(self, hdf, group_name="structures"):
        # truncate arrays to necessary size before writing
        self._resize_atoms(self.num_atoms)
        self._resize_structures(self.num_structures)

        with hdf.open(group_name) as hdf_s_lst:
            hdf_s_lst["num_atoms"] =  self._num_atoms_alloc
            hdf_s_lst["num_structures"] = self._num_structures_alloc
            hdf_s_lst["len_current_struct"] = self._per_structure_arrays["len_current_struct"]

            hdf_arrays = hdf_s_lst.open("arrays")
            for k, a in chain(self._per_atom_arrays.items(), self._per_structure_arrays.items()):
                if a.dtype.char == "U":
                    # numpy stores unicode data in UTF-32/UCS-4, but h5py wants UTF-8, so we manually encode them here
                    # TODO: string arrays with shape != () not handled
                    hdf_arrays[k] = np.array([s.encode("utf8") for s in a],
                                             # each character in a utf8 string might be encoded in up to 4 bytes, so to
                                             # make sure we can store any string of length n we tell h5py that the
                                             # string will be 4 * n bytes; numpy's dtype does this calculation already
                                             # in itemsize, so we don't need to repeat it here
                                             # see also https://docs.h5py.org/en/stable/strings.html
                                             dtype=h5py.string_dtype('utf8', a.dtype.itemsize))
                else:
                    hdf_arrays[k] = a


    def from_hdf(self, hdf, group_name="structures"):
        with hdf.open(group_name) as hdf_s_lst:
            self._num_structures_alloc = self.current_structure_index = hdf_s_lst["num_structures"]
            self._num_atoms_alloc = self.current_atom_index = hdf_s_lst["num_atoms"]

            with hdf_s_lst.open("arrays") as hdf_arrays:
                for k in hdf_arrays.list_nodes():
                    a = np.array(hdf_arrays[k])
                    if a.dtype.char == "S":
                        # if saved as bytes, we wrote this as an encoded unicode string, so manually decode here
                        # TODO: string arrays with shape != () not handled
                        a = np.array([s.decode("utf8") for s in a],
                                     # itemsize of original a is four bytes per character, so divide by four to get
                                     # length of the orignal stored unicode string; np.dtype('U1').itemsize is just a
                                     # platform agnostic way of knowing how wide a unicode charater is for numpy
                                     dtype=f"U{a.dtype.itemsize//np.dtype('U1').itemsize}")
                    if a.shape[0] == self._num_atoms_alloc:
                        self._per_atom_arrays[k] = a
                    elif a.shape[0] == self._num_structures_alloc:
                        self._per_structure_arrays[k] = a

    def _translate_frame(self, frame):
        for i, name in enumerate(self._per_structure_arrays["identifiers"]):
            if name == frame:
                return i
        raise KeyError(f"No structure named {frame} in StructureContainer.")

    def _get_structure(self, frame=-1, wrap_atoms=True):
        I = self._per_structure_arrays["start_indices"][frame]
        E = I + self._per_structure_arrays["len_current_struct"][frame]
        return Atoms(symbols=self._per_atom_arrays["symbols"][I:E],
                     cell=self._per_structure_arrays["cells"][frame],
                     positions=self._per_atom_arrays["positions"][I:E])

    def _number_of_structures(self):
        return len(self)
