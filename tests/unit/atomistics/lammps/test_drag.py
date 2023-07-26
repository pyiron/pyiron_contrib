import unittest
from pyiron_atomistics import Project
from pyiron_contrib.atomistics.lammps.drag import setup_lmp_input

class TestDrag(unittest.TestCase):
    def setUp(self):
        pr = Project(".")
        bulk = pr.create.structure.bulk('Ni', cubic=True)
        self.lmp = pr.create.job.Lammps("test")
        self.lmp.structure = bulk.repeat(2)

    def test_calc_minimize(self):
        self.assertRaises(ValueError, setup_lmp_input, self.lmp)


if __name__ == '__main__':
    unittest.main()
