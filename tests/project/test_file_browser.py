import unittest
import os

from pyiron_contrib.project.file_browser import FileBrowser
from pyiron_contrib.generic.data import Data
from pyiron_contrib.project.project import Project


class TestFileBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up project and file browser classes."""
        cls.filebrowser = FileBrowser()
        cls.current_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        cls.project = Project(os.path.join(cls.current_dir, "sub_folder"))

    @classmethod
    def tearDown(self):
        """Tear down test classes."""
        self.project.remove(enable=True)

    def test_data(self):
        """Test data attribute by setting a _clickedFile by hand."""
        self.filebrowser._clickedFiles = ["../static/test_data.txt"]
        data = self.filebrowser.data()
        self.assertTrue(isinstance(data[0], Data))

    def test_project_path_init(self):
        """Test for the init of the file browser using a project as argument."""
        filebrowser = FileBrowser(project=self.project)
        self.assertEqual(filebrowser.path, self.project.path)
        self.assertTrue(filebrowser.fix_storage_sys)
        self.assertTrue(filebrowser.hdf_as_dirs)

    def test_init(self):
        """Test different init scenarios."""
        filebrowser = FileBrowser(s3path="", fix_s3_path=False,  storage_system="local",
                        fix_storage_sys=False, hdf_as_dirs=False, hide_hdf=False)
        self.assertEqual(filebrowser.path, self.current_dir)
        # Since no S3 credentials are provided:
        self.assertTrue(filebrowser.fix_storage_sys)
        self.assertFalse(filebrowser.hide_hdf)
        self.assertFalse(filebrowser.hdf_as_dirs)
        self.assertFalse(filebrowser.fix_s3_path)

        filebrowser = FileBrowser(s3path='some/random/path', fix_s3_path=True, storage_system='S3',
                                  fix_storage_sys=True, hdf_as_dirs=True)
        self.assertTrue(filebrowser.fix_storage_sys)
        self.assertFalse(filebrowser.hide_hdf)
        self.assertTrue(filebrowser.hdf_as_dirs)
        self.assertTrue(filebrowser.fix_s3_path)
        self.assertEqual(filebrowser.s3path, 'some/random/path')
        # Since no S3 credentials are provided:
        self.assertEqual(filebrowser.data_sys, 'local')

    def test_configure(self):
        """Test configure."""
        filebrowser = FileBrowser(project=self.project)
        filebrowser.configure(s3path=None, fix_s3_path=None, storage_system=None, fix_storage_sys=None,
                              hdf_as_dirs=None, hide_hdf=None)
        # This should change nothing, thus:
        self.assertEqual(filebrowser.path, self.current_dir)
        self.assertTrue(filebrowser.fix_storage_sys)
        self.assertFalse(filebrowser.hide_hdf)
        self.assertFalse(filebrowser.hdf_as_dirs)
        self.assertFalse(filebrowser.fix_s3_path)
        self.assertTrue(filebrowser.s3path, '')

        filebrowser.configure(s3path='/some/path/', fix_s3_path=None, storage_system=None, fix_storage_sys=None,
                              hdf_as_dirs=None, hide_hdf=None)
        # This should change nothing, thus:
        self.assertEqual(filebrowser.path, self.current_dir)
        self.assertTrue(filebrowser.fix_storage_sys)
        self.assertFalse(filebrowser.hide_hdf)
        self.assertFalse(filebrowser.hdf_as_dirs)
        self.assertFalse(filebrowser.fix_s3_path)
        self.assertEqual(filebrowser.s3path, 'some/path')

        filebrowser.configure(s3path=None, fix_s3_path=False, storage_system=None, fix_storage_sys=None,
                              hdf_as_dirs=None, hide_hdf=None)
        self.assertEqual(filebrowser.path, self.current_dir)
        self.assertFalse(filebrowser.fix_storage_sys)
        self.assertFalse(filebrowser.hide_hdf)
        self.assertFalse(filebrowser.hdf_as_dirs)
        # Change only here:
        self.assertFalse(filebrowser.fix_s3_path)
        self.assertEqual(filebrowser.s3path, 'some/path')

        filebrowser.configure(s3path=None, fix_s3_path=None, storage_system=None, fix_storage_sys=None,
                              hdf_as_dirs=True, hide_hdf=None)
        self.assertEqual(filebrowser.path, self.current_dir)
        self.assertFalse(filebrowser.fix_storage_sys)
        self.assertFalse(filebrowser.hide_hdf)
        # Change only here:
        self.assertTrue(filebrowser.hdf_as_dirs)
        self.assertFalse(filebrowser.fix_s3_path)
        self.assertEqual(filebrowser.s3path, 'some/path')

        self.assertRaises(AttributeError, filebrowser.configure(storage_system='S3'))

if __name__ == '__main__':
    unittest.main()
