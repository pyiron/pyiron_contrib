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

class StructureStorage(HasStructure):
    """
    Class that can write and read lots of structures from and to hdf quickly.

    This is done by storing positions, cells, etc. into large arrays instead of writing every structure into a new
    group.  Structures are stored together with an identifier that should be unique.  The class can be initialized with
    the number of structures and the total number of atoms in all structures, but re-allocates memory as necessary when
    more (or larger) structures are added than initially anticipated.

    You can add structures and a human-readable name with :method:`.add_structure()`.

    >>> container = StructureStorage()
    >>> container.add_structure(Atoms(...), "fcc")
    >>> container.add_structure(Atoms(...), "hcp")
    >>> container.add_structure(Atoms(...), "bcc")

    Accessing stored structures works with :method:`.get_strucure()`.  You can either pass the identifier you passed
    when adding the structure or the numeric index

    >>> container.get_structure(frame=0) == container.get_structure(frame="fcc")
    True

    Custom arrays may also be defined on the container

    >>> container.add_array("energy", shape=(), dtype=np.float64, fill=-1, per="structure")

    You can then pass arrays of the corresponding shape to :method:`add_structure()`

    >>> container.add_structure(Atoms(...), "grain_boundary", energy=3.14)

    Saved arrays are accessed with :method:`.get_array()`

    >>> container.get_array("energy", 3)
    3.14
    >>> container.get_array("energy", 0)
    -1

    It is also possible to use the same names in :method:`.get_array()` as in :method:`.get_structure()`.

    >>> container.get_array("energy", 0) == container.get_array("energy", "fcc")
    True

    The length of the container is the number of structures inside it.

    >>> len(container)
    4

    By default the following arrays are defined for each structure:
        - identifier    shape=(),    dtype=str,          per structure; human readable name of the structure
        - cell          shape=(3,3), dtype=np.float64,   per structure; cell shape
        - pbc           shape=(3,),  dtype=bool          per structure; periodic boundary conditions
        - symbols:      shape=(),    dtype=str,          per atom; chemical symbol
        - positions:    shape=(3,),  dtype=np.float64,   per atom: atomic positions
    If a structure has spins/magnetic moments defined on its atoms these will be saved in a per atom array as well.  In
    that case, however all structures in the container must either have all collinear spins or all non-collinear spins.

    When adding new array follow the convention that per-structure arrays should be named in singular and per-atom
    arrays should be named in plural.
    """

    __version__ = "0.1.0"
    __hdf_version__ = "0.1.0"

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
                "start_index": np.empty(self._num_structures_alloc, dtype=np.int32),
                "length": np.empty(self._num_structures_alloc, dtype=np.int32),
                "identifier": np.empty(self._num_structures_alloc, dtype=np.dtype("U20")),
                "cell": np.empty((self._num_structures_alloc, 3, 3)),
                "pbc": np.empty((self._num_atoms_alloc, 3), dtype=bool)
        }

    @property
    def symbols(self):
        """:meta private:"""
        return self._per_atom_arrays["symbols"]

    @property
    def positions(self):
        """:meta private:"""
        return self._per_atom_arrays["positions"]

    @property
    def start_index(self):
        """:meta private:"""
        return self._per_structure_arrays["start_index"]

    @property
    def length(self):
        """:meta private:"""
        return self._per_structure_arrays["length"]

    @property
    def identifier(self):
        """:meta private:"""
        return self._per_structure_arrays["identifier"]

    @property
    def cell(self):
        """:meta private:"""
        return self._per_structure_arrays["cell"]

    @property
    def pbc(self):
        """:meta private:"""
        return self._per_structure_arrays["pbc"]

    def get_elements(self):
        """
        Return a list of chemical elements in the training set.

        Returns:
            :class:`list`: list of unique elements in the training set as strings of their standard abbreviations
        """
        return list(set(self._per_atom_arrays["symbols"]))

    def _get_per_atom_slice(self, frame):
        start = self._per_structure_arrays["start_index"][frame]
        end = start + self._per_structure_arrays["length"][frame]
        return slice(start, end, 1)

    def get_array(self, name, frame):
        """
        Fetch array for given structure.

        Works for per atom and per arrays.

        Args:
            name (str): name of the array to fetch
            frame (int, str): selects structure to fetch, as in :method:`.get_structure()`

        Returns:
            :class:`numpy.ndarray`: requested array

        Raises:
            `KeyError`: if array with name does not exists
        """

        if isinstance(frame, str):
            frame = self._translate_frame(frame)
        if name in self._per_atom_arrays:
            return self._per_atom_arrays[name][self._get_per_atom_slice(frame)]
        elif name in self._per_structure_arrays:
            return self._per_structure_arrays[name][frame]
        else:
            raise KeyError(f"no array named {name} defined on StructureStorage")

    def set_array(self, name, frame, value):
        """
        Add array for given structure.

        Works for per atom and per arrays.

        Args:
            name (str): name of array to set
            frame (int, str): selects structure to set, as in :method:`.get_strucure()`

        Raises:
            `KeyError`: if array with name does not exists
        """

        if isinstance(frame, str):
            frame = self._translate_frame(frame)
        if name in self._per_atom_arrays:
            self._per_atom_arrays[name][self._get_per_atom_slice(frame)] = value
        elif name in self._per_structure_arrays:
            self._per_structure_arrays[name][frame] = value
        else:
            raise KeyError(f"no array named {name} defined on StructureStorage")

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
        """
        Add a custom array to the container.

        When adding an array after some structures have been added, specifying `fill` will be used as a default value
        for the value of the array for those structures.

        Adding an array with the same name twice is ignored, if dtype and shape match, otherwise raises an exception.

        >>> container = StructureStorage()
        >>> container.add_structure(Atoms(...), "foo")
        >>> container.add_array("energy", shape=(), dtype=np.float64, fill=42, per="structure")
        >>> container.get_array("energy", 0)
        42

        Args:
            name (str): name of the new array
            shape (tuple of int): shape of the new array per atom or structure; scalars can pass ()
            dtype (type): data type of the new array, string arrays can pass 'U$n' where $n is the length of the string
            fill (object): populate the new array with this value for existing structure, if given; default `None`
            per (str): either "atom" or "structure"; denotes whether the new array should exist for every atom in a
                       structure or only once for every structure; case-insensitive

        Raises:
            ValueError: if wrong value for `per` is given
            ValueError: if array with same name but different parameters exists already
        """

        if name in self._per_atom_arrays:
            a = self._per_atom_arrays[name]
            if a.shape[1:] != shape or a.dtype != dtype or per != "atom":
                raise ValueError(f"Array with name '{name}' exists with shape {a.shape[1:]} and dtype {a.dtype}.")
            else:
                return

        if name in self._per_structure_arrays:
            a = self._per_structure_arrays[name]
            if a.shape[1:] != shape or a.dtype != dtype or per != "structure":
                raise ValueError(f"Array with name '{name}' exists with shape {a.shape[1:]} and dtype {a.dtype}.")
            else:
                return

        per = per.lower()
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

    def add_structure(self, structure, identifier=None, **arrays):
        """
        Add a new structure to the container.

        Additional keyword arguments given specify additional arrays to store for the structure.  If an array with the
        given keyword name does not exist yet, it will be added to the container.

        >>> container = StructureStorage()
        >>> container.add_structure(Atoms(...), identifier="A", energy=3.14)
        >>> container.get_array("energy", 0)
        3.14

        If the first axis of the extra array matches the length of the given structure, it will be added as an per atom
        array, otherwise as an per structure array.

        >>> structure = Atoms(...)
        >>> container.add_structure(structure, identifier="B", forces=len(structure) * [[0,0,0]])
        >>> len(container.get_array("forces", 1)) == len(structure)
        True

        Reshaping the array to have the first axis be length 1 forces the array to be set as per structure array.  That
        axis will then be stripped.

        >>> container.add_structure(Atoms(...), identifier="C", pressure=np.eye(3)[np.newaxis, :, :])
        >>> container.get_array("pressure", 2).shape
        (3, 3)

        Args:
            structure (:class:`.Atoms`): structure to add
            identifier (str, optional): human-readable name for the structure, if None use current structre index as
                                        string
            **kwargs: additional arrays to store for structure
        """

        if identifier is None:
            identifier = str(self.num_structures)

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

        self._per_structure_arrays["start_index"][self.current_structure_index] = self.current_atom_index
        self._per_structure_arrays["length"][self.current_structure_index] = n
        self._per_structure_arrays["identifier"][self.current_structure_index] = identifier
        self._per_structure_arrays["cell"][self.current_structure_index] = structure.cell.array
        self._per_structure_arrays["pbc"][self.current_structure_index] = structure.pbc

        if structure.spins is not None:
            arrays["spins"] = structure.spins

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


    def _type_to_hdf(self, hdf):
        """
        Internal helper function to save type and version in hdf root

        Args:
            hdf (ProjectHDFio): HDF5 group object
        """
        hdf["NAME"] = self.__class__.__name__
        hdf["TYPE"] = str(type(self))
        hdf["VERSION"] = self.__version__
        hdf["HDF_VERSION"] = self.__hdf_version__
        hdf["OBJECT"] = "StructureStorage"

    def to_hdf(self, hdf, group_name="structures"):
        # truncate arrays to necessary size before writing
        self._resize_atoms(self.num_atoms)
        self._resize_structures(self.num_structures)

        with hdf.open(group_name) as hdf_s_lst:
            self._type_to_hdf(hdf_s_lst)
            hdf_s_lst["num_atoms"] =  self._num_atoms_alloc
            hdf_s_lst["num_structures"] = self._num_structures_alloc

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
            self._num_structures_alloc = self.num_structures = self.current_structure_index = hdf_s_lst["num_structures"]
            self._num_atoms_alloc = self.num_atoms = self.current_atom_index = hdf_s_lst["num_atoms"]

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
        for i, name in enumerate(self._per_structure_arrays["identifier"]):
            if name == frame:
                return i
        raise KeyError(f"No structure named {frame} in StructureStorage.")

    def _get_structure(self, frame=-1, wrap_atoms=True):
        slc = self._get_per_atom_slice(frame)
        if "spins" in self._per_atom_arrays:
            magmoms = self._per_atom_arrays["spins"][slc]
        else:
            # not all structures have spins saved on them
            magmoms = None
        return Atoms(symbols=self._per_atom_arrays["symbols"][slc],
                     positions=self._per_atom_arrays["positions"][slc],
                     cell=self._per_structure_arrays["cell"][frame],
                     pbc=self._per_structure_arrays["pbc"][frame],
                     magmoms=magmoms)

    def _number_of_structures(self):
        return len(self)
