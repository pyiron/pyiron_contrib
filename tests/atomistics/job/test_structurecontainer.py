from unittest import TestCase
from pyiron_atomistics import Project
from pyiron_contrib.atomistics.atomistics.job.structurecontainer import StructureContainer
import os.path
import numpy as np

class TestContainer(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project = Project(os.path.dirname(__file__))
        cls.structures = [cls.project.create.structure.bulk(el).repeat(3) for el in ("Fe", "Mg", "Al", "Cu", "Ti")]

    def setUp(self):
        self.cont = StructureContainer(
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

    def test_add_array(self):
        """Custom arrays added with add_array should be properly allocated with matching shape, dtype and fill"""

        self.cont.add_array("energy", per="structure")
        self.cont.add_array("forces", shape=(3,), per="atom")
        self.cont.add_array("fnorble", shape=(), dtype=np.int64, fill=0, per="atom")

        self.assertTrue("energy" in self.cont._per_structure_arrays,
                        "no 'energy' array present after adding it with add_array()")
        self.assertEqual(self.cont._per_structure_arrays["energy"].shape, (self.cont._num_structures_alloc,),
                        "'energy' array has wrong shape")

        self.assertTrue("forces" in self.cont._per_atom_arrays,
                        "no 'forces' array present after adding it with add_array()")
        self.assertEqual(self.cont._per_atom_arrays["forces"].shape, (self.cont._num_atoms_alloc, 3),
                        "'forces' array has wrong shape")

        self.assertEqual(self.cont._per_atom_arrays["fnorble"].dtype, np.int64,
                         "'fnorble' array has wrong dtype after adding it with add_array()")
        self.assertTrue((self.cont._per_atom_arrays["fnorble"] == 0).all(),
                         "'fnorble' array not initialized with given fill value.")

    def test_get_structure(self):
        """Structure from get_structure should match thoes add with add_structure exactly."""

        for i, s in enumerate(self.structures):
            self.assertEqual(s, self.cont.get_structure(i),
                             "Added structure not equal to returned structure.")

    def test_add_structure_kwargs(self):
        """Additional kwargs given to add_structure should create appropriate custom arrays."""

        E = 3.14
        P = np.eye(3) * 2.72
        F = np.array([[1,3,5]] * len(self.structures[0]))
        R = np.ones(len(self.structures[0]))
        self.cont.add_structure(self.structures[0], self.structures[0].get_chemical_formula(),
                                energy=E, forces=F, pressure=P, fnord=R[None, :])
        self.assertEqual(self.cont.get_array("energy", self.cont.num_structures - 1), E,
                         "Energy returned from get_array() does not match energy passed to add_structure")
        self.assertTrue(np.allclose(self.cont.get_array("forces", self.cont.num_structures - 1), F),
                        "Forces returned from get_array() does not match forces passed to add_structure")
        self.assertTrue(np.allclose(self.cont.get_array("pressure", self.cont.num_structures - 1), P),
                        "Pressure returned from get_array() does not match pressure passed to add_structure")
        self.assertTrue("fnord" in self.cont._per_structure_arrays,
                        "array 'fnord' not in per structure array, even though shape[0]==1")
        self.assertEqual(self.cont.get_array("fnord", self.cont.num_structures - 1).shape, R.shape,
                        "array 'fnord' added with wrong shape, even though shape[0]==1")
        self.assertTrue((self.cont.get_array("fnord", self.cont.num_structures - 1) == R).all(),
                        "Fnord returned from get_array() does not match fnord passed to add_structure")

    def test_resize(self):
        """A dynamically resized container should behave exactly as a pre-allocated container."""

        cont_static = self.cont
        cont_dynamic = StructureContainer(num_structures=2, num_atoms=10)

        for s in self.structures:
            cont_dynamic.add_structure(s, s.get_chemical_formula())

        self.assertEqual(cont_static.current_atom_index, cont_dynamic.current_atom_index,
                         "Dynamic container doesn't have the same current atom after adding structures.")
        self.assertEqual(cont_static.current_structure_index, cont_dynamic.current_structure_index,
                         "Dynamic container doesn't have the same current structure after adding structures.")
        self.assertTrue( (cont_static.symbols[:cont_static.current_atom_index] \
                            == cont_dynamic.symbols[:cont_dynamic.current_atom_index]).all(),
                        "Array of chemical symbols not equal after adding structures.")
        self.assertTrue(np.isclose(cont_static.positions[:cont_static.current_atom_index],
                                   cont_static.positions[:cont_dynamic.current_atom_index]).all(),
                        "Array of chemical symbols not equal after adding structures.")
        self.assertTrue(np.isclose(cont_static.cells[:cont_static.current_structure_index],
                                   cont_dynamic.cells[:cont_dynamic.current_structure_index]).all(),
                        "Array of chemical symbols not equal after adding structures.")
        self.assertTrue( (cont_static.identifiers[:cont_static.current_structure_index] \
                            == cont_dynamic.identifiers[:cont_dynamic.current_structure_index]).all(),
                        "Array of chemical symbols not equal after adding structures.")
