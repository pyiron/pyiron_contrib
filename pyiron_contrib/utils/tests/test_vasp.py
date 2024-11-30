import unittest
import tempfile
import os
import shutil

from utils.vasp import (find_vasp_directories)

class TestFindVaspDirectories(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_files()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        dir1_path = os.path.join(self.temp_dir, "dir1")
        os.makedirs(dir1_path)

        vasp_log_path = os.path.join(dir1_path, "asdf.jpg")
        with open(vasp_log_path, "w") as file:
            file.write("This is a jpg file")

        incar_path = os.path.join(dir1_path, "pasdf.txt")
        with open(incar_path, "w") as file:
            file.write("This is random text file")

        dir2_path = os.path.join(self.temp_dir, "dir2")
        os.makedirs(dir2_path)

        vasp_log_path = os.path.join(dir2_path, "vasp.log")
        with open(vasp_log_path, "w") as file:
            file.write("This is vasp.log")

        potcar_path = os.path.join(dir2_path, "POTCAR")
        with open(potcar_path, "w") as file:
            file.write("This is POTCAR")

        dir3_path = os.path.join(self.temp_dir, "dir3")
        os.makedirs(dir3_path)

        incar_path = os.path.join(dir3_path, "INCAR")
        with open(incar_path, "w") as file:
            file.write("This is INCAR")

        kpoints_path = os.path.join(dir3_path, "KPOINTS")
        with open(kpoints_path, "w") as file:
            file.write("This is KPOINTS")

        outcar_path = os.path.join(dir3_path, "OUTCAR")
        with open(outcar_path, "w") as file:
            file.write("This is OUTCAR")

    def test_find_vasp_directories(self):
        parent_dir = self.temp_dir
        filenames = ["vasp.log", "INCAR", "POTCAR", "CONTCAR", "KPOINTS", "OUTCAR"]
        all_present = False
        extract_tarballs = True

        directories = find_vasp_directories(parent_dir, filenames, all_present, extract_tarballs)

        self.assertEqual(len(directories), 2)

        expected_dirs = ["dir2", "dir3"]
        for dir_name in expected_dirs:
            self.assertIn(dir_name, [os.path.basename(dir) for dir in directories])
            
    def test_find_vasp_directories_negative(self):
        # Create a temporary empty directory to test the negative case
        empty_dir = tempfile.mkdtemp()

        # Call the function with empty directory and all_present=True
        parent_dir = empty_dir
        filenames = ["vasp.log", "INCAR", "POTCAR", "CONTCAR", "KPOINTS", "OUTCAR"]
        all_present = True
        extract_tarballs = True

        directories = find_vasp_directories(parent_dir, filenames, all_present, extract_tarballs)

        # Assert that the function returns an empty list as there are no directories that meet the criteria
        self.assertEqual(len(directories), 0)

if __name__ == "__main__":
    unittest.main()
