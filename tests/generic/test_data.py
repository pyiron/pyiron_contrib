import unittest

from pyiron_contrib.generic.filedata import FileData

class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = FileData(source="../static/test_data.txt")

    def test___init__(self):
        """Test init of data class"""
        # Test if init from setUpClass is as expected
        self.assertFalse(self.data._hasdata)
        self.assertEqual(self.data.filename, "test_data.txt")
        self.assertEqual(self.data.filetype, "txt")
        self.assertRaises(ValueError, FileData())
        data = FileData(source="../static/test_data.txt", metadata={"some": "dict"})
        self.assertFalse(data._hasdata)
        self.assertEqual(data.filename, "test_data.txt")
        self.assertEqual(data.filetype, "txt")
        self.assertEqual(data.metadata["some"], "dict")
        with open("../static/test_data.txt") as f:
            some_data = f.readlines()
        self.assertRaises(ValueError, FileData(data=some_data))
        data = FileData(data=some_data, filename="test_data.dat")
        self.assertTrue(data._hasdata)
        self.assertEqual(data.filetype, "dat")
        data = FileData(data=some_data, filename="test_data.dat", filetype="txt")
        self.assertEqual(data.filetype, "txt")

    def test_data(self):
        """Test data property of FileData"""
        with open("../static/test_data.txt", "rb") as f:
            some_data = f.readlines()
        self.assertEqual(self.data.data, some_data)

    def test_data_as_numpy_array(self):
        """Test data conversion to numpy"""
        self.assertTrue(self.data.data_as_np_array()[0] == 'some text')



if __name__ == '__main__':
    unittest.main()
