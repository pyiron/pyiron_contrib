# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Alternative structure container that stores them in flattened arrays.
"""

import numpy as np

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

    def _init_arrays(self):
        self.symbols = np.full(self._num_atoms_alloc, "XX", dtype=np.dtype("U2"))# 2 character unicode array for chemical symbols
        self.positions = np.empty((self._num_atoms_alloc, 3))
        self.cells = np.empty((self._num_structures_alloc, 3, 3))
        self.start_indices = np.empty(self._num_structures_alloc, dtype=np.int32)
        self.len_current_struct = np.empty(self._num_structures_alloc, dtype=np.int32)
        self.identifiers = np.empty(self._num_structures_alloc, dtype=np.dtype("U20"))

    def _resize_atoms(self, new):
        self._num_atoms_alloc = new
        try:
            self.symbols.resize(new)
            self.positions.resize( (new, 3) )
        except ValueError:
            self.symbols = np.resize(self.symbols, new)
            self.positions = np.resize(self.positions, (new, 3) )

    def _resize_structures(self, new):
        self._num_structures_alloc = new
        try:
            self.cells.resize( (new, 3, 3) )
            self.start_indices.resize(new)
            self.len_current_struct.resize(new)
            self.identifiers.resize(new)
        except ValueError:
            self.cells = np.resize(self.cells, (new, 3, 3) )
            self.start_indices = np.resize(self.start_indices, new)
            self.len_current_struct = np.resize(self.len_current_struct, new)
            self.identifiers = np.resize(self.identifiers, new)

    def add_structure(self, structure, identifier):
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

        self.symbols[self.current_atom_index:i] = np.array(structure.symbols)
        self.positions[self.current_atom_index:i] = structure.positions

        self.len_current_struct[self.current_structure_index] = n
        self.cells[self.current_structure_index] = structure.cell.array
        self.start_indices[self.current_structure_index] = self.current_atom_index
        self.identifiers[self.current_structure_index] = identifier

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
            hdf_s_lst["symbols"] = self.symbols.astype(np.dtype("S2"))
            hdf_s_lst["positions"] = self.positions
            hdf_s_lst["cells"] = self.cells
            hdf_s_lst["start_indices"] = self.start_indices
            hdf_s_lst["identifiers"] = self.identifiers.astype(np.dtype("S20"))
            hdf_s_lst["num_atoms"] =  self._num_atoms_alloc
            hdf_s_lst["num_structures"] = self._num_structures_alloc
            hdf_s_lst["len_current_struct"] = self.len_current_struct


    def from_hdf(self, hdf, group_name="structures"):
        with hdf.open(group_name) as hdf_s_lst:
            self._num_structures_alloc = hdf_s_lst["num_structures"]
            self._num_atoms_alloc = hdf_s_lst["num_atoms"]

            self._init_arrays()

            self.symbols = hdf_s_lst["symbols"].astype(np.dtype("U2"))
            self.positions = hdf_s_lst["positions"]
            self.cells = hdf_s_lst["cells"]
            self.start_indices = hdf_s_lst["start_indices"]
            self.identifiers = hdf_s_lst["identifiers"].astype(np.dtype("U20"))
            self.len_current_struct = hdf_s_lst["len_current_struct"]


    def _translate_frame(self, frame):
        for i, name in enumerate(self.identifiers):
            if name == frame:
                return i
        raise KeyError(f"No structure named {frame} in StructureContainer.")

    def _get_structure(self, frame=-1):
        E = I + self.len_current_struct[frame]
        return Atoms(symbols=self.symbols[I:E], cell=self.cells[frame], positions=self.positions[I:E])

    def _number_of_structures(self):
        return self.num_structures
