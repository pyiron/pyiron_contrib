import unittest

from pyiron_contrib.generic.data import Data

class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data(source="../static/test_data.txt")

    def test___init__(self):
        """Test init of data class"""
        # Test if init from setUpClass is as expected
        self.assertFalse(self.data.hasdata)
        self.assertEqual(self.data.filename, "test_data.txt")
        self.assertEqual(self.data.filetype, "txt")
        self.assertRaises(ValueError, Data())
        data = Data(source="../static/test_data.txt", metadata={"some": "dict"})
        self.assertFalse(data.hasdata)
        self.assertEqual(data.filename, "test_data.txt")
        self.assertEqual(data.filetype, "txt")
        self.assertEqual(data.metadata["some"], "dict")
        data = Data(source="../static/test_data.txt", storedata=True)
        self.assertTrue(data.hasdata)
        with open("../static/test_data.txt") as f:
            some_data = f.readlines()
        self.assertRaises(ValueError, Data(data=some_data))
        data = Data(data=some_data, filename="test_data.dat")
        self.assertTrue(data.hasdata)
        self.assertEqual(data.filetype, "dat")
        data = Data(data=some_data, filename="test_data.dat", filetype="txt")
        self.assertEqual(data.filetype, "txt")

    def test_data(self):
        """Test data property of Data"""
        with open("../static/test_data.txt", "rb") as f:
            some_data = f.read()
        self.assertEqual(self.data.data, some_data)

    def test_data_as_numpy_array(self):
        """Test data conversion to numpy"""
        self.assertTrue(self.data.data_as_np_array() is None)



if __name__ == '__main__':
    unittest.main()
