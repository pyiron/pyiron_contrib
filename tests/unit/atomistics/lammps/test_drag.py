import unittest
from pyiron_atomistics import Project
from pyiron_contrib.atomistics.lammps.drag import setup_lmp_input

class TestDrag(unittest.TestCase):
    def setUp(self):
        pr = Project(".")
        self.lmp = pr.create.job.Lammps("test")
        self.lmp.structure = pr.create.structure.bulk('Ni', cubic=True)

    def test_calc_minimize(self):
        self.assertRaises(ValueError, setup_lmp_input, self.lmp)

    def test_min_style(self):
        self.lmp.calc_minimize()
        self.assertEqual(self.lmp.input.control["min_style"], "cg")
        setup_lmp_input(self.lmp)
        self.assertEqual(self.lmp.input.control["min_style"], "quickmin")

    def test_direction(self):
        self.lmp.calc_minimize()
        setup_lmp_input(self.lmp, direction=[0, 0, 1])
        self.assertEqual(self.lmp.input.control["variable___fx_free"], " equal 0")


if __name__ == '__main__':
    unittest.main()
