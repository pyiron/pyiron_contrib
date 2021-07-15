from pyiron_atomistics._tests import TestWithProject
from pyiron_contrib.atomistics.atomistics.job.structurestorage import FlattenedStorage, StructureStorage
import os.path
import numpy as np

class TestFlattenedStorage(TestWithProject):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.store = FlattenedStorage()

        cls.even = [ list(range(0, 2, 2)), list(range(2, 6, 2)), list(range(6, 12, 2)) ]
        cls.odd = np.array([ np.arange(1, 2, 2), np.arange(3, 6, 2), np.arange(7, 12, 2) ], dtype=object)


    def test_add_array(self):
        """Custom arrays added with add_array should be properly allocated with matching shape, dtype and fill"""

        store = FlattenedStorage()
        store.add_array("energy", per="chunk")
        store.add_array("forces", shape=(3,), per="element")
        store.add_array("fnorble", shape=(), dtype=np.int64, fill=0, per="element")

        self.assertTrue("energy" in store._per_chunk_arrays,
                        "no 'energy' array present after adding it with add_array()")
        self.assertEqual(store._per_chunk_arrays["energy"].shape, (store._num_chunks_alloc,),
                        "'energy' array has wrong shape")

        self.assertTrue("forces" in store._per_element_arrays,
                        "no 'forces' array present after adding it with add_array()")
        self.assertEqual(store._per_element_arrays["forces"].shape, (store._num_elements_alloc, 3),
                        "'forces' array has wrong shape")

        self.assertEqual(store._per_element_arrays["fnorble"].dtype, np.int64,
                         "'fnorble' array has wrong dtype after adding it with add_array()")
        self.assertTrue((store._per_element_arrays["fnorble"] == 0).all(),
                         "'fnorble' array not initialized with given fill value")

        try:
            store.add_array("energy", dtype=np.float64, per="chunk")
        except ValueError:
            self.fail("Duplicate calls to add_array should be ignored if types/shapes are compatible!")

        with self.assertRaises(ValueError, msg="Duplicate calls to add_array with invalid shape!"):
            store.add_array("energy", shape=(5,), dtype=np.float64, per="chunk")

        with self.assertRaises(ValueError, msg="Duplicate calls to add_array with invalid type!"):
            store.add_array("energy", dtype=np.complex64, per="chunk")

        try:
            store.add_array("forces", shape=(3,), dtype=np.float64, per="element")
        except ValueError:
            self.fail("Duplicate calls to add_array should be ignored if types/shapes are compatible!")

        with self.assertRaises(ValueError, msg="Duplicate calls to add_array with invalid type!"):
            store.add_array("forces", shape=(3,), dtype=np.complex64, per="element")

        with self.assertRaises(ValueError, msg="Duplicate calls to add_array with invalid shape!"):
            store.add_array("forces", shape=(5,), dtype=np.float64, per="element")

        with self.assertRaises(ValueError, msg="Cannot have per-chunk and per-element array of the same name!"):
            store.add_array("energy", per="element")

        with self.assertRaises(ValueError, msg="Cannot have per-chunk and per-element array of the same name!"):
            store.add_array("forces", per="chunk")

        with self.assertRaises(ValueError, msg="Invalid per value!"):
            store.add_array("foobar", per="xyzzy")

    def test_resize(self):
        """A dynamically resized container should behave exactly as a pre-allocated container."""

        foo = [ [1], [2, 3], [4, 5, 6] ]
        bar = [ 1, 2, 3 ]
        store_static = FlattenedStorage(num_chunks=3, num_elements=6)
        store_dynamic = FlattenedStorage(num_chunks=1, num_elements=1)

        store_static.add_array("foo", per="element")
        store_static.add_array("bar", per="chunk")

        for f, b in zip(foo, bar):
            store_static.add_chunk(len(f), foo=f, bar=b)
            store_dynamic.add_chunk(len(f), foo=f, bar=b)

        self.assertEqual(store_static.current_element_index, store_dynamic.current_element_index,
                         "Dynamic storeainer doesn't have the same current element after adding chunks.")
        self.assertEqual(store_static.current_chunk_index, store_dynamic.current_chunk_index,
                         "Dynamic storeainer doesn't have the same current chunk after adding chunks.")
        self.assertTrue( (store_static._per_element_arrays["foo"][:store_static.current_element_index] \
                            == store_dynamic._per_element_arrays["foo"][:store_dynamic.current_element_index]).all(),
                        "Array of per element quantity not equal after adding chunks.")
        self.assertTrue(np.isclose(store_static._per_chunk_arrays["bar"][:store_static.current_element_index],
                                   store_static._per_chunk_arrays["bar"][:store_dynamic.current_element_index]).all(),
                        "Array of per chunk quantity not equal after adding chunks.")

    def test_init(self):
        """Adding arrays via __init__ should be equivalent to adding them via add_chunks manually."""

        store = FlattenedStorage(even=self.even, odd=self.odd)
        self.assertEqual(len(store), 3, "Length of storage doesn't match length of initializer!")
        self.assertTrue( (store.get_array("even", 1) == np.array([2, 4])).all(),
                        "Values added via init don't match expected values!")
        self.assertTrue( (store.get_array("odd", 2) == np.array([7, 9, 11])).all(),
                        "Values added via init don't match expected values!")

        all_sum = [sum(e + o) for e, o in zip(self.even, self.odd)]
        try:
            FlattenedStorage(even=self.even, odd=self.odd, sum=all_sum)
        except ValueError:
            self.fail("Adding per chunk values to initializers raises error, but shouldn't!")

        with self.assertRaises(ValueError, msg="No error on inconsistent initializers!"):
            odd = self.odd.copy()
            odd[1] = [1,3,4]
            FlattenedStorage(even=self.even, odd=odd)

        with self.assertRaises(ValueError, msg="No error on initializers of different length!"):
            FlattenedStorage(foo=[ [1] ], bar=[ [2], [2, 3] ])

    def test_find_chunk(self):
        """find_chunk() should return the correct indices given an identifier."""

        store = FlattenedStorage()
        store.add_chunk(2, "first", integers=[1, 2])
        store.add_chunk(3, integers=[3, 4, 5])
        store.add_chunk(1, "third", integers=[5])

        self.assertEqual(store.find_chunk("first"), 0, "Incorrect chunk index returned!")
        self.assertEqual(store.find_chunk("1"), 1, "Incorrect chunk index returned for unamed identifier!")
        self.assertEqual(store.find_chunk("third"), 2, "Incorrect chunk index returned!")

        with self.assertRaises(KeyError, msg="No KeyError raised on non-existing identifier!"):
            store.find_chunk("asdf")

    def test_get_array(self):
        """get_array should return the arrays for the correct structures."""

        store = FlattenedStorage()

        for n, e, o in zip( ("first", None, "third"), self.even, self.odd):
            store.add_chunk(len(e), identifier=n, even=e, odd=o, sum=sum(e + o))

        self.assertTrue(np.array_equal(store.get_array("even", 0), self.even[0]),
                        "get_array returns wrong array for numeric index!")

        self.assertTrue(np.array_equal(store.get_array("even", "first"), self.even[0]),
                        "get_array returns wrong array for string identifier!")

        self.assertTrue(np.array_equal(store.get_array("even", "1"), self.even[1]),
                        "get_array returns wrong array for automatic identifier!")

        self.assertTrue(np.array_equal(store.get_array("sum", 0), sum(self.even[0] + self.odd[0])),
                        "get_array returns wrong array for numeric index!")

        self.assertTrue(np.array_equal(store.get_array("sum", "first"), sum(self.even[0] + self.odd[0])),
                        "get_array returns wrong array for string identifier!")

        self.assertTrue(np.array_equal(store.get_array("sum", "1"), sum(self.even[1] + self.odd[1])),
                        "get_array returns wrong array for automatic identifier!")

        with self.assertRaises(KeyError, msg="Non-existing identifier!"):
            store.get_array("even", "foo")

    def test_has_array(self):
        """hasarray should return correct information for added array; None otherwise."""

        store = FlattenedStorage()
        store.add_array("energy", per="chunk")
        store.add_array("forces", shape=(3,), per="element")

        info = store.has_array("energy")
        self.assertEqual(info["dtype"], np.float64, "has_array returns wrong dtype for per structure array.")
        self.assertEqual(info["shape"], (), "has_array returns wrong shape for per structure array.")
        self.assertEqual(info["per"], "chunk", "has_array returns wrong per for per structure array.")

        info = store.has_array("forces")
        self.assertEqual(info["dtype"], np.float64, "has_array returns wrong dtype for per atom array.")
        self.assertEqual(info["shape"], (3,), "has_array returns wrong shape for per atom array.")
        self.assertEqual(info["per"], "element", "has_array returns wrong per for per atom array.")

        self.assertEqual(store.has_array("missing"), None, "has_array does not return None for nonexisting array.")

class TestContainer(TestWithProject):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.elements = ("Fe", "Mg", "Al", "Cu", "Ti")
        cls.structures = [cls.project.create.structure.bulk(el).repeat(3) for el in cls.elements]

    def setUp(self):
        self.cont = StructureStorage(
                    num_structures=len(self.structures),
                    num_atoms=sum(len(s) for s in self.structures)
        )
        for s in self.structures:
            self.cont.add_structure(s, s.get_chemical_formula())

    def tearDown(self):
        del self.cont

    def test_len(self):
        """Length of container should be equal to number of calls to add_structure."""
        self.assertEqual(len(self.cont), len(self.structures))

    def test_get_elements(self):
        """get_elements() should return all unique chemical elements stored in its structures."""
        self.assertEqual(sorted(self.elements), sorted(self.cont.get_elements()),
                         "Results from get_elements() do not match added elements.")

    def test_set_array(self):
        """set_array should set the arrays for the correct structures and only those."""

        new_pbc = [True, False, True]
        self.cont.set_array("pbc", 0, new_pbc)
        self.assertTrue((self.cont.get_array("pbc", 0,) == new_pbc).all(),
                        f"Value from get_array {self.cont.get_array('pbc', 0)} does not match set value {new_pbc}")

        symbols = self.cont.get_array("symbols", 2)
        symbols[5:10] = 'Cu'
        self.cont.set_array("symbols", 2, symbols)
        self.assertTrue((self.cont.get_array("symbols", 2) == symbols).all(),
                        f"Value from get_array {self.cont.get_array('symbols', 0)} does not match set value {symbols}")

        self.cont.set_array("positions", 0, np.ones( (len(self.structures[0]), 3) ))
        for i, structure in enumerate(self.structures):
            if i == 0: continue

            self.assertTrue(np.allclose(self.cont.get_array("positions", i), structure.positions),
                            f"set_array modified arrray for different structure than instructured.")

    def test_get_structure(self):
        """Structure from get_structure should match thoes add with add_structure exactly."""

        for i, s in enumerate(self.structures):
            self.assertEqual(s, self.cont.get_structure(i),
                             "Added structure not equal to returned structure.")

    def test_translate_frame(self):
        """Using get_structure with the given identifiers should return the respective structure."""
        for s in self.structures:
            self.assertEqual(s, self.cont.get_structure(s.get_chemical_formula()),
                             "get_structure returned wrong structure for given identifier.")

    def test_add_structure(self):
        """add_structure(identifier=None) should set the current structure index as identifier"""

        for i, structure in enumerate(self.structures):
            self.cont.add_structure(structure)
            self.assertEqual(self.cont.get_array("identifier", len(self.structures) + i),
                             str(len(self.structures) + i),
                             "Default identifier is incorrect.")

    def test_add_structure_kwargs(self):
        """Additional kwargs given to add_structure should create appropriate custom arrays."""

        E = 3.14
        P = np.eye(3) * 2.72
        F = np.array([[1,3,5]] * len(self.structures[0]))
        R = np.ones(len(self.structures[0]))
        self.cont.add_structure(self.structures[0], self.structures[0].get_chemical_formula(),
                                energy=E, forces=F, pressure=P, fnord=R[None, :])
        self.assertEqual(self.cont.get_array("energy", self.cont.number_of_structures - 1), E,
                         "Energy returned from get_array() does not match energy passed to add_structure")
        got_F = self.cont.get_array("forces", self.cont.number_of_structures - 1)
        self.assertTrue(np.allclose(got_F, F),
                        f"Forces returned from get_array() {got_F} do not match forces passed to add_structure {F}")
        got_P = self.cont.get_array("pressure", self.cont.number_of_structures - 1)
        self.assertTrue(np.allclose(got_P, P),
                        f"Pressure returned from get_array() {got_P} does not match pressure passed to add_structure {P}")
        self.assertTrue("fnord" in self.cont._per_chunk_arrays,
                        "array 'fnord' not in per structure array, even though shape[0]==1")
        got_R = self.cont.get_array("fnord", self.cont.number_of_structures - 1)
        self.assertEqual(got_R.shape, R.shape,
                        f"array 'fnord' added with wrong shape {got_R.shape}, even though shape[0]==1 ({R.shape})")
        self.assertTrue((got_R == R).all(),
                        f"Fnord returned from get_array() {got_R} does not match fnord passed to add_structure {R}")

    def test_add_structure_spins(self):
        """If saved structures have spins, they should be saved and restored, too."""

        fe = self.structures[0].copy()

        cont = StructureStorage()
        spins = [1] * len(fe)
        fe.set_initial_magnetic_moments(spins)
        cont.add_structure(fe, "iron_spins")
        fe_read = cont.get_structure("iron_spins")
        self.assertTrue(fe_read.spins is not None,
                        "Spins not restored on added structure.")
        self.assertTrue(np.allclose(spins, fe_read.spins),
                        f"Spins restored on added structure not equal to original spins: {spins} {fe_read.spins}.")

        # repeat for vector spins
        cont = StructureStorage()
        spins = [(1,0,1)] * len(fe)
        fe.set_initial_magnetic_moments(spins)
        cont.add_structure(fe, "iron_spins")
        fe_read = cont.get_structure("iron_spins")
        self.assertTrue(fe_read.spins is not None,
                        "Spins not restored on added structure.")
        self.assertTrue(np.allclose(spins, fe_read.spins),
                        f"Spins restored on added structure not equal to original spins: {spins} {fe_read.spins}.")

    def test_hdf(self):
        """Containers written to, then read from HDF should match."""
        hdf = self.project.create_hdf(self.project.path, "test_hdf")
        self.cont.to_hdf(hdf)
        cont_read = StructureStorage()
        cont_read.from_hdf(hdf)

        self.assertEqual(len(self.cont), len(cont_read), "Container size not matching after reading from HDF.")
        self.assertEqual(self.cont.num_chunks, cont_read.num_chunks,
                         "num_chunks does not match after reading from HDF.")
        self.assertEqual(self.cont.num_elements, cont_read.num_elements,
                         "num_elements does not match after reading from HDF.")
        for s1, s2 in zip(self.cont.iter_structures(), cont_read.iter_structures()):
            self.assertEqual(s1, s2, "Structure from get_structure not matching after reading from HDF.")

        cont_read.to_hdf(hdf, "other_structures")
        cont_read.from_hdf(hdf, "other_structures")

        # bug regression: if you mess up reading some variables it might work fine when you use but it could write
        # itself wrongly to the HDF, thus double check here.
        self.assertEqual(len(self.cont), len(cont_read), "Container size not matching after reading from HDF twice.")
        self.assertEqual(self.cont.num_chunks, cont_read.num_chunks,
                         "num_structures does not match after reading from HDF twice.")
        self.assertEqual(self.cont.num_elements, cont_read.num_elements,
                         "num_atoms does not match after reading from HDF twice.")
        for s1, s2 in zip(self.cont.iter_structures(), cont_read.iter_structures()):
            self.assertEqual(s1, s2, "Structure from get_structure not matching after reading from HDF twice.")

        self.assertEqual(set(self.cont._per_element_arrays.keys()),
                         set(cont_read._per_element_arrays.keys()),
                         "per atom arrays read are not the same as written")
        self.assertEqual(set(self.cont._per_chunk_arrays.keys()),
                         set(cont_read._per_chunk_arrays.keys()),
                         "per structure arrays read are not the same as written")

        for n in self.cont._per_element_arrays:
            self.assertTrue((self.cont._per_element_arrays[n] == cont_read._per_element_arrays[n]).all(),
                            f"per atom array {n} read is not the same as writen")
        for n in self.cont._per_chunk_arrays:
            self.assertTrue((self.cont._per_chunk_arrays[n] == cont_read._per_chunk_arrays[n]).all(),
                            f"per structure array {n} read is not the same as writen")
