import os
import unittest
from pyiron_contrib.project.project import Project
from pyiron_base import InputList

class TestProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        cls.project = Project(os.path.join(cls.current_dir, "sub_folder"))

    def tearDown(self):
        self.project.remove(enable=True)

    def test_open_file_browser(self):
        self.project.open_file_browser()

    def test_projectinfo(self):
        self.assertTrue(isinstance(self.project.project_info, InputList))
        self.project.project_info = {"some": "meta", "data": 1}
        self.assertTrue(isinstance(self.project.project_info, InputList))
        self.assertEqual(self.project.project_info["some"], "meta")

    def test_save_projectinfo(self):
        self.project.project_info = {"some": "meta", "data": 1}
        self.project._save_projectinfo()

    def test_load_projectinfo(self):
        self.project.project_info = {"some": "meta", "data": 1}
        self.project._save_projectinfo()
        self.project.project_info = None
        self.project._load_projectinfo()
        self.assertEqual(self.project.project_info["some"], "meta")

    def test_metadata(self):
        self.assertTrue(isinstance(self.project.metadata, InputList))
        self.project.metadata = {"some": "meta", "data": 1}
        self.assertTrue(isinstance(self.project.metadata, InputList))
        self.assertEqual(self.project.metadata["some"], "meta")

    def test_save_metadata(self):
        self.project.metadata = {"some": "meta", "data": 1}
        self.project.save_metadata()

    def test_load_metadata(self):
        self.project.metadata = {"some": "meta", "data": 1}
        self.project.save_metadata()
        self.project.metadata = None
        self.project.load_metadata()
        self.assertEqual(self.project.metadata["some"], "meta")

    def test_open(self):
        new_project = self.project.open("sub")
        self.assertTrue(isinstance(new_project, Project))
        self.assertTrue(hasattr(new_project, "metadata"))

if __name__ == "__main__":
    unittest.main()
