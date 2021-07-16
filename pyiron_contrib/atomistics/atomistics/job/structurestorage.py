# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Alternative structure container that stores them in flattened arrays.
"""

from itertools import chain
import warnings

import numpy as np
import h5py

from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure

class FlattenedStorage:
    """
    Efficient storage of ragged arrays in flattened arrays.

    This class stores multiple arrays at the same time.  Storage is organized in "chunks" that may be of any size, but
    all arrays within chunk are of the same size, e.g.

    >>> a = [ [1], [2, 3], [4,  5,  6] ]
    >>> b = [ [2], [4, 6], [8, 10, 12] ]

    are stored as in three chunks like

    >>> a_flat = [ 1,  2, 3,  4,  5,  6 ]
    >>> b_flat = [ 2,  4, 6,  8, 10, 12 ]

    with additional metadata to indicate where the boundaries of each chunk are. 

    First add arrays and chunks like this

    >>> store = FlattenedStorage()
    >>> store.add_array("even", dtype=np.int64)
    >>> store.add_chunk(1, even=[2])
    >>> store.add_chunk(2, even=[4,  6])
    >>> store.add_chunk(3, even=[8, 10, 12])

    where the first argument indicates the length of each chunk.  You may retrieve stored values like this

    >>> store.get_array("even", 1)
    array([4, 6])
    >>> store.get_array("even", 0)
    array([2])

    where the second arguments are integer indices in the order of insertion.  After intial storage you may modify
    arrays.

    >>> store.set_array("even", 0, [0])
    >>> store.get_array("even", 0)
    array([0])

    You can add arrays to the storage even after you added already other arrays and chunks.

    >>> store.add_array("odd", dtype=np.int64, fill=0)
    >>> store.get_array("odd", 1)
    array([0, 0])
    >>> store.set_array("odd", 0, [1])
    >>> store.set_array("odd", 1, [3, 5])
    >>> store.set_array("odd", 2, [7, 9, 11])
    >>> store.get_array("odd", 2)
    array([ 7,  9, 11])

    Because the second chunk is already known to be of length two and `fill` was specified the 'odd' array has been
    appropriatly allocated.

    Additionally arrays may also only have one value per chunk ("per chunk", previous examples are "per element").

    >>> store.add_array("sum", dtype=np.int64, per="chunk")
    >>> for i in range(len(store)):
    ...    store.set_array("sum", i, sum(store.get_array("even", i) + store.get_array("odd", i)))
    >>> store.get_array("sum", 0)
    1
    >>> store.get_array("sum", 1)
    18
    >>> store.get_array("sum", 2)
    57

    Finally you may add multiple arrays in one call to :method:`.add_chunk` by using keyword arguments

    >>> store.add_chunk(4, even=[14, 16, 18, 20], odd=[13, 15, 17, 19], sum=119)
    >>> store.get_array("sum", 3)
    119
    >>> store.get_array("even", 3)
    array([14, 16, 18, 20])

    Chunks may be given string names, either by passing `identifier` to :method:`.add_chunk` or by setting to the
    special per chunk array "identifier"

    >>> store.set_array("identifier", 1, "second")
    >>> all(store.get_array("even", "second") == store.get_array("even", 1))
    True

    It is usually not necessary to call :method:`.add_array` before :method:`.add_chunk`, the type of the array will be
    inferred in this case.

    Arrays may be of more complicated shape, too, see :method:`.add_array` for details.

    When adding new arrays follow the convention that per-structure arrays should be named in singular and per-atom
    arrays should be named in plural.

    You may initialize flattened storage objects with a ragged lists or numpy arrays of dtype object

    >>> even = [ list(range(0, 2, 2)), list(range(2, 6, 2)), list(range(6, 12, 2)) ]
    >>> even
    [[0], [2, 4], [6, 8, 10]]

    >>> import numpy as np
    >>> odd = np.array([ np.arange(1, 2, 2), np.arange(3, 6, 2), np.arange(7, 12, 2) ], dtype=object)
    >>> odd
    array([array([1]), array([3, 5]), array([ 7,  9, 11])], dtype=object)

    >>> store = FlattenedStorage(even=even, odd=odd)
    >>> store.get_array("even", 1)
    array([2, 4])
    >>> store.get_array("odd", 2)
    array([ 7,  9, 11])
    >>> len(store)
    3
    """

    __version__ = "0.1.0"
    __hdf_version__ = "0.1.0"

    def __init__(self, num_chunks=1, num_elements=1, **kwargs):
        """
        Create new flattened storage.

        Args:
            num_chunks (int): pre-allocation for per chunk arrays
            num_elements (int): pre-allocation for per elements arrays
        """
        # tracks allocated versed as yet used number of chunks/elements
        self._num_chunks_alloc = self.num_chunks = num_chunks
        self._num_elements_alloc = self.num_elements = num_elements
        # store the starting index for properties with unknown length
        self.current_element_index = 0
        # store the index for properties of known size, stored at the same index as the chunk
        self.current_chunk_index = 0
        # Also store indices of chunk recently added
        self.prev_chunk_index = 0
        self.prev_element_index = 0

        self._init_arrays()

        if len(kwargs) == 0: return

        if len(set(len(chunks) for chunks in kwargs.values())) != 1:
            raise ValueError("Not all initializers provide the same number of chunks!")
        keys = kwargs.keys()
        for chunk_list in zip(*kwargs.values()):
            chunk_length = len(chunk_list[0])
            # values in chunk_list may either be a sequence of chunk_length, scalars (see hasattr check) or a sequence of
            # length 1
            if any(hasattr(c, '__len__') and len(c) != chunk_length and len(c) != 1 for c in chunk_list):
                raise ValueError("Inconsistent chunk length in initializer!")
            self.add_chunk(chunk_length, **{k: c for k, c in zip(keys, chunk_list)})

    def _init_arrays(self):
        self._per_element_arrays = {}

        self._per_chunk_arrays = {
                "start_index": np.empty(self._num_chunks_alloc, dtype=np.int32),
                "length": np.empty(self._num_chunks_alloc, dtype=np.int32),
                "identifier": np.empty(self._num_chunks_alloc, dtype=np.dtype("U20"))
        }

    def __len__(self):
        return self.current_chunk_index

    def find_chunk(self, identifier):
        """
        Return integer index for given identifier.

        Args:
            identifier (str): name of chunk previously passed to :method:`.add_chunk`

        Returns:
            int: integer index for chunk

        Raises:
            KeyError: if identifier is not found in storage
        """
        for i, name in enumerate(self._per_chunk_arrays["identifier"]):
            if name == identifier:
                return i
        raise KeyError(f"No chunk named {identifier}")

    def _get_per_element_slice(self, frame):
        start = self._per_chunk_arrays["start_index"][frame]
        end = start + self._per_chunk_arrays["length"][frame]
        return slice(start, end, 1)


    def _resize_elements(self, new):
        self._num_elements_alloc = new
        for k, a in self._per_element_arrays.items():
            new_shape = (new,) + a.shape[1:]
            try:
                a.resize(new_shape)
            except ValueError:
                self._per_element_arrays[k] = np.resize(a, new_shape)

    def _resize_chunks(self, new):
        self._num_chunks_alloc = new
        for k, a in self._per_chunk_arrays.items():
            new_shape = (new,) + a.shape[1:]
            try:
                a.resize(new_shape)
            except ValueError:
                self._per_chunk_arrays[k] = np.resize(a, new_shape)

    def add_array(self, name, shape=(), dtype=np.float64, fill=None, per="element"):
        """
        Add a custom array to the container.

        When adding an array after some chunks have been added, specifying `fill` will be used as a default value
        for the value of the array for those chunks.

        Adding an array with the same name twice is ignored, if dtype and shape match, otherwise raises an exception.

        >>> store = FlattenedStorage()
        >>> store.add_chunk(1, "foo")
        >>> store.add_array("energy", shape=(), dtype=np.float64, fill=42, per="chunk")
        >>> store.get_array("energy", 0)
        42.0

        Args:
            name (str): name of the new array
            shape (tuple of int): shape of the new array per element or chunk; scalars can pass ()
            dtype (type): data type of the new array, string arrays can pass 'U$n' where $n is the length of the string
            fill (object): populate the new array with this value for existing chunk, if given; default `None`
            per (str): either "element" or "chunk"; denotes whether the new array should exist for every element in a
                       chunk or only once for every chunk; case-insensitive

        Raises:
            ValueError: if wrong value for `per` is given
            ValueError: if array with same name but different parameters exists already
        """

        if per == "structure":
            per = "chunk"
            warnings.warn("per=\"structure\" is deprecated, use pr=\"chunk\"",
                          category=DeprecationWarning, stacklevel=2)
        if per == "atom":
            per = "element"
            warnings.warn("per=\"atom\" is deprecated, use pr=\"element\"",
                          category=DeprecationWarning, stacklevel=2)

        if name in self._per_element_arrays:
            a = self._per_element_arrays[name]
            if a.shape[1:] != shape or not np.can_cast(dtype, a.dtype) or per != "element":
                raise ValueError(f"Array with name '{name}' exists with shape {a.shape[1:]} and dtype {a.dtype}.")
            else:
                return

        if name in self._per_chunk_arrays:
            a = self._per_chunk_arrays[name]
            if a.shape[1:] != shape or not np.can_cast(dtype, a.dtype) or per != "chunk":
                raise ValueError(f"Array with name '{name}' exists with shape {a.shape[1:]} and dtype {a.dtype}.")
            else:
                return

        per = per.lower()
        if per == "element":
            shape = (self._num_elements_alloc,) + shape
            store = self._per_element_arrays
        elif per == "chunk":
            shape = (self._num_chunks_alloc,) + shape
            store = self._per_chunk_arrays
        else:
            raise ValueError(f"per must \"element\" or \"chunk\", not {per}")

        if fill is None:
            store[name] = np.empty(shape=shape, dtype=dtype)
        else:
            store[name] = np.full(shape=shape, fill_value=fill, dtype=dtype)

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
            frame = self.find_chunk(frame)
        if name in self._per_element_arrays:
            return self._per_element_arrays[name][self._get_per_element_slice(frame)]
        elif name in self._per_chunk_arrays:
            return self._per_chunk_arrays[name][frame]
        else:
            raise KeyError(f"no array named {name}")

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
            frame = self.find_chunk(frame)
        if name in self._per_element_arrays:
            self._per_element_arrays[name][self._get_per_element_slice(frame)] = value
        elif name in self._per_chunk_arrays:
            self._per_chunk_arrays[name][frame] = value
        else:
            raise KeyError(f"no array named {name}")

    def has_array(self, name):
        """
        Checks whether an array of the given name exists and returns meta data given to :method:`.add_array()`.

        >>> container.has_array("energy")
        {'shape': (), 'dtype': np.float64, 'per': 'chunk'}
        >>> container.has_array("fnorble")
        None

        Args:
            name (str): name of the array to check

        Returns:
            None: if array does not exist
            dict: if array exists, keys corresponds to the shape, dtype and per arguments of :method:`.add_array`
        """
        if name in self._per_element_arrays:
            a = self._per_element_arrays[name]
            per = "element"
        elif name in self._per_chunk_arrays:
            a = self._per_chunk_arrays[name]
            per = "chunk"
        else:
            return None
        return {"shape": a.shape[1:], "dtype": a.dtype, "per": per}


    def add_chunk(self, chunk_length, identifier=None, **arrays):
        """
        Add a new chunk to the storeage.

        Additional keyword arguments given specify arrays to store for the chunk.  If an array with the given keyword
        name does not exist yet, it will be added to the container.

        >>> container = FlattenedStorage()
        >>> container.add_chunk(2, identifier="A", energy=3.14)
        >>> container.get_array("energy", 0)
        3.14

        If the first axis of the extra array matches the length of the chunk, it will be added as an per element array,
        otherwise as an per chunk array.

        >>> container.add_chunk(2, identifier="B", forces=2 * [[0,0,0]])
        >>> len(container.get_array("forces", 1)) == 2
        True

        Reshaping the array to have the first axis be length 1 forces the array to be set as per chunk array.  That axis
        will then be stripped.

        >>> container.add_chunk(2, identifier="C", pressure=np.eye(3)[np.newaxis, :, :])
        >>> container.get_array("pressure", 2).shape
        (3, 3)

        Args:
            chunk_length (int): length of the new chunk
            identifier (str, optional): human-readable name for the chunk, if None use current chunk index as string
            **kwargs: additional arrays to store for the chunk
        """

        if identifier is None:
            identifier = str(self.num_chunks)

        n = chunk_length
        new_elements = self.current_element_index + n

        if new_elements > self._num_elements_alloc:
            self._resize_elements(max(new_elements, self._num_elements_alloc * 2))
        if self.current_chunk_index + 1 > self._num_chunks_alloc:
            self._resize_chunks(self._num_chunks_alloc * 2)

        if new_elements > self.num_elements:
            self.num_elements = new_elements
        if self.current_chunk_index + 1 > self.num_chunks:
            self.num_chunks += 1

        # len of chunk to index into the initialized arrays
        i = self.current_element_index + n

        self._per_chunk_arrays["start_index"][self.current_chunk_index] = self.current_element_index
        self._per_chunk_arrays["length"][self.current_chunk_index] = n
        self._per_chunk_arrays["identifier"][self.current_chunk_index] = identifier

        for k, a in arrays.items():
            a = np.asarray(a)
            if len(a.shape) > 0 and a.shape[0] == n:
                if k not in self._per_element_arrays:
                    self.add_array(k, shape=a.shape[1:], dtype=a.dtype, per="element")
                self._per_element_arrays[k][self.current_element_index:i] = a
            else:
                if len(a.shape) > 0 and a.shape[0] == 1:
                    a = a[0]
                if k not in self._per_chunk_arrays:
                    self.add_array(k, shape=a.shape, dtype=a.dtype, per="chunk")
                self._per_chunk_arrays[k][self.current_chunk_index] = a

        self.prev_chunk_index = self.current_chunk_index
        self.prev_element_index = self.current_element_index

        # Set new current_element_index and increase current_chunk_index
        self.current_chunk_index += 1
        self.current_element_index = i
        #return last_chunk_index, last_element_index


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
        hdf["OBJECT"] = self.__class__.__name__

    def to_hdf(self, hdf, group_name="flat_storage"):
        # truncate arrays to necessary size before writing
        self._resize_elements(self.num_elements)
        self._resize_chunks(self.num_chunks)

        with hdf.open(group_name) as hdf_s_lst:
            self._type_to_hdf(hdf_s_lst)
            hdf_s_lst["num_elements"] =  self._num_elements_alloc
            hdf_s_lst["num_chunks"] = self._num_chunks_alloc

            hdf_arrays = hdf_s_lst.open("arrays")
            for k, a in chain(self._per_element_arrays.items(), self._per_chunk_arrays.items()):
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

    def from_hdf(self, hdf, group_name="flat_storage"):
        with hdf.open(group_name) as hdf_s_lst:
            version = hdf_s_lst.get("HDF_VERSION", "0.0.0")
            num_chunks = hdf_s_lst["num_chunks"] or hdf_s_lst["num_structures"]
            num_elements = hdf_s_lst["num_elements"] or hdf_s_lst["num_atoms"]
            self._num_chunks_alloc = self.num_chunks = self.current_chunk_index = num_chunks
            self._num_elements_alloc = self.num_elements = self.current_element_index = num_elements

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
                    if a.shape[0] == self._num_elements_alloc:
                        self._per_element_arrays[k] = a
                    elif a.shape[0] == self._num_chunks_alloc:
                        self._per_chunk_arrays[k] = a


class StructureStorage(FlattenedStorage, HasStructure):
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

    >>> container.add_array("energy", shape=(), dtype=np.float64, fill=-1, per="chunk")

    (chunk means structure in this case, see below and :class:`.FlattenedStorage`)

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

    Each structure corresponds to a chunk in :class:`.FlattenedStorage` and each atom to an element.  By default the
    following arrays are defined for each structure:
        - identifier    shape=(),    dtype=str,          per chunk; human readable name of the structure
        - cell          shape=(3,3), dtype=np.float64,   per chunk; cell shape
        - pbc           shape=(3,),  dtype=bool          per chunk; periodic boundary conditions
        - symbols:      shape=(),    dtype=str,          per element; chemical symbol
        - positions:    shape=(3,),  dtype=np.float64,   per element: atomic positions
    If a structure has spins/magnetic moments defined on its atoms these will be saved in a per atom array as well.  In
    that case, however all structures in the container must either have all collinear spins or all non-collinear spins.
    """

    def __init__(self, num_atoms=1, num_structures=1):
        """
        Create new structure container.

        Args:
            num_atoms (int): total number of atoms across all structures to pre-allocate
            num_structures (int): number of structures to pre-allocate
        """
        super().__init__(num_elements=num_atoms, num_chunks=num_structures)

    def _init_arrays(self):
        super()._init_arrays()
        # 2 character unicode array for chemical symbols
        self._per_element_arrays["symbols"] = np.full(self._num_elements_alloc, "XX", dtype=np.dtype("U2"))
        self._per_element_arrays["positions"] = np.empty((self._num_elements_alloc, 3))

        self._per_chunk_arrays["cell"] = np.empty((self._num_chunks_alloc, 3, 3))
        self._per_chunk_arrays["pbc"] = np.empty((self._num_elements_alloc, 3), dtype=bool)


    @property
    def symbols(self):
        """:meta private:"""
        return self._per_element_arrays["symbols"]

    @property
    def positions(self):
        """:meta private:"""
        return self._per_element_arrays["positions"]

    @property
    def start_index(self):
        """:meta private:"""
        return self._per_chunk_arrays["start_index"]

    @property
    def length(self):
        """:meta private:"""
        return self._per_chunk_arrays["length"]

    @property
    def identifier(self):
        """:meta private:"""
        return self._per_chunk_arrays["identifier"]

    @property
    def cell(self):
        """:meta private:"""
        return self._per_chunk_arrays["cell"]

    @property
    def pbc(self):
        """:meta private:"""
        return self._per_chunk_arrays["pbc"]


    def get_elements(self):
        """
        Return a list of chemical elements in the training set.

        Returns:
            :class:`list`: list of unique elements in the training set as strings of their standard abbreviations
        """
        return list(set(self._per_element_arrays["symbols"]))

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

        if structure.spins is not None:
            arrays["spins"] = structure.spins

        self.add_chunk(len(structure),
                       identifier=identifier,
                       symbols=np.array(structure.symbols),
                       positions=structure.positions,
                       cell=structure.cell.array,
                       pbc=structure.pbc,
                       **arrays)


    def _translate_frame(self, frame):
        try:
            return self.find_chunk(frame)
        except KeyError:
            raise KeyError(f"No structure named {frame}.") from None

    def _get_structure(self, frame=-1, wrap_atoms=True):
        try:
            magmoms = self.get_array("spins", frame)
        except KeyError:
            # not all structures have spins saved on them
            magmoms = None
        return Atoms(symbols=self.get_array("symbols", frame),
                     positions=self.get_array("positions", frame),
                     cell=self.get_array("cell", frame),
                     pbc=self.get_array("pbc", frame),
                     magmoms=magmoms)

    def _number_of_structures(self):
        return len(self)


    def to_hdf(self, hdf, group_name="structures"):
        # just overwrite group_name default
        super().to_hdf(hdf=hdf, group_name=group_name)

    def from_hdf(self, hdf, group_name="structures"):
        with hdf.open(group_name) as hdf_s_lst:
            version = hdf_s_lst.get("HDF_VERSION", "0.0.0")
            if version == "0.1.0":
                super().from_hdf(hdf=hdf, group_name=group_name)

            elif version == "0.0.0":
                self._per_element_arrays["symbols"] = hdf_s_lst["symbols"].astype(np.dtype("U2"))
                self._per_element_arrays["positions"] = hdf_s_lst["positions"]

                self._per_chunk_arrays["start_index"] = hdf_s_lst["start_indices"]
                self._per_chunk_arrays["length"] = hdf_s_lst["len_current_struct"]
                self._per_chunk_arrays["identifier"] = hdf_s_lst["identifiers"].astype(np.dtype("U20"))
                self._per_chunk_arrays["cell"] = hdf_s_lst["cells"]

                self._per_chunk_arrays["pbc"] = np.full((self.num_chunks, 3), True)
