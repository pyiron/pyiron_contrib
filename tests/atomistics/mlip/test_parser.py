import unittest

import numpy as np

from pyiron_contrib.atomistics.mlip.parser import potential

mtp08 = """
MTP
version = 1.1.0
potential_name = MTP1m
scaling = 7.430083706879999e+00
species_count = 1
potential_tag = 
radial_basis_type = RBChebyshev
	min_dist = 2.000000000000000e+00
	max_dist = 5.000000000000000e+00
	radial_basis_size = 8
	radial_funcs_count = 2
	radial_coeffs
		0-0
			{3.554069395582601e-01, 7.665830760050807e-01, 3.256060114004741e-01, 3.547676784062508e-01, 1.750861385166761e-01, 1.398528343990471e-01, 5.653452154334131e-02, 2.732746746903122e-02}
			{5.327986310319617e-01, -3.867927012903804e-01, 6.605896831670657e-01, -3.452742717327723e-01, 9.518803436004511e-02, -2.918125711726744e-02, 2.562297027862194e-02, 1.884483253906242e-02}
alpha_moments_count = 18
alpha_index_basic_count = 11
alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 2, 0, 0}, {0, 1, 1, 0}, {0, 1, 0, 1}, {0, 0, 2, 0}, {0, 0, 1, 1}, {0, 0, 0, 2}, {1, 0, 0, 0}}
alpha_index_times_count = 14
alpha_index_times = {{0, 0, 1, 11}, {1, 1, 1, 12}, {2, 2, 1, 12}, {3, 3, 1, 12}, {4, 4, 1, 13}, {5, 5, 2, 13}, {6, 6, 2, 13}, {7, 7, 1, 13}, {8, 8, 2, 13}, {9, 9, 1, 13}, {0, 10, 1, 14}, {0, 11, 1, 15}, {0, 12, 1, 16}, {0, 15, 1, 17}}
alpha_scalar_moments = 9
alpha_moment_mapping = {0, 10, 11, 12, 13, 14, 15, 16, 17}
species_coeffs = {-7.206554037754401e-02}
moment_coeffs = {-4.997232021481571e-02, 2.983004039435163e-03, 3.943225926195725e-04, -4.917912150526871e-02, -8.472293821242558e-05, 2.193369043858843e-04, -1.002075759201661e-05, -3.061450593696996e-02, 3.785744244383398e-07}
"""

mtp08_missing_key = """
MTP
version = 1.1.0
potential_name = MTP1m
scaling = 7.430083706879999e+00
species_count = 1
potential_tag = 
radial_basis_type = RBChebyshev
	min_dist = 2.000000000000000e+00
	max_dist = 5.000000000000000e+00
	radial_basis_size = 8
	radial_funcs_count = 2
	radial_coeffs
		0-0
			{3.554069395582601e-01, 7.665830760050807e-01, 3.256060114004741e-01, 3.547676784062508e-01, 1.750861385166761e-01, 1.398528343990471e-01, 5.653452154334131e-02, 2.732746746903122e-02}
			{5.327986310319617e-01, -3.867927012903804e-01, 6.605896831670657e-01, -3.452742717327723e-01, 9.518803436004511e-02, -2.918125711726744e-02, 2.562297027862194e-02, 1.884483253906242e-02}
alpha_index_basic_count = 11
alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 2, 0, 0}, {0, 1, 1, 0}, {0, 1, 0, 1}, {0, 0, 2, 0}, {0, 0, 1, 1}, {0, 0, 0, 2}, {1, 0, 0, 0}}
alpha_index_times_count = 14
alpha_index_times = {{0, 0, 1, 11}, {1, 1, 1, 12}, {2, 2, 1, 12}, {3, 3, 1, 12}, {4, 4, 1, 13}, {5, 5, 2, 13}, {6, 6, 2, 13}, {7, 7, 1, 13}, {8, 8, 2, 13}, {9, 9, 1, 13}, {0, 10, 1, 14}, {0, 11, 1, 15}, {0, 12, 1, 16}, {0, 15, 1, 17}}
alpha_scalar_moments = 9
alpha_moment_mapping = {0, 10, 11, 12, 13, 14, 15, 16, 17}
species_coeffs = {-7.206554037754401e-02}
moment_coeffs = {-4.997232021481571e-02, 2.983004039435163e-03, 3.943225926195725e-04, -4.917912150526871e-02, -8.472293821242558e-05, 2.193369043858843e-04, -1.002075759201661e-05, -3.061450593696996e-02, 3.785744244383398e-07}
"""

class TestMlipParser(unittest.TestCase):

    def test_parse(self):
        try:
            result = potential(mtp08)
            for pair in result["radial"]["funcs"]:
                self.assertTrue(isinstance(result["radial"]["funcs"][pair], np.ndarray),
                                "Entry list not parsed as numpy array!")
            self.assertTrue(isinstance(result["alpha_index_basic"], np.ndarray),
                            "Entry list not parsed as numpy array!")
            self.assertTrue(isinstance(result["alpha_index_times"], np.ndarray),
                            "Entry list not parsed as numpy array!")
            self.assertTrue(isinstance(result["alpha_moment_mapping"], np.ndarray),
                            "Entry list not parsed as numpy array!")
            self.assertTrue(isinstance(result["species_coeffs"], np.ndarray),
                            "Entry list not parsed as numpy array!")
            self.assertTrue(isinstance(result["moment_coeffs"], np.ndarray),
                            "Entry list not parsed as numpy array!")
        except ValueError:
            self.fail("Well-formated potential should not raise an error.")

        with self.assertRaises(ValueError, msg="Misformated potential should raise an error"):
            potential(mtp08_missing_key)
