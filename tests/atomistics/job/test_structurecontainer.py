from unittest import TestCase
from pyiron_atomistics import Project
from pyiron_contrib.atomistics.atomistics.job.structurecontainer import StructureContainer
import os.path
import numpy as np

class TestContainer(TestCase):

    def test_resize(self):
        """A dynamically resized container should behave exactly as a pre-allocated container."""

        project = Project(os.path.dirname(__file__))
        structures = [project.create.structure.bulk(el).repeat(3) for el in ("Fe", "Mg", "Al", "Cu", "Ti")]
        cont_static = StructureContainer(
                    num_structures=len(structures),
                    num_atoms=sum(len(s) for s in structures)
        )
        cont_dynamic = StructureContainer(num_structures=2, num_atoms=10)

        for s in structures:
            cont_static.add_structure(s, s.get_chemical_formula())
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
