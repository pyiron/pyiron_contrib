import unittest
import tempfile
import os
import shutil
import numpy as np
import tarfile
import filecmp

# Import the function to be tested
from utils.generic import (chunk_list,
                           search_line_in_file,
                           parse_lines,
                           find_directories_with_files,
                           extract_tarball,
                           find_and_extract_tarballs_parallel,
                           extract_files_from_tarball,
                           extract_files_from_tarballs_parallel,
                           find_and_extract_files_from_tarballs_parallel,
                           compress_directory,
                           compress_directory_parallel,
                           cleanup_dir,
                           compress_and_cleanup,
                           find_and_compress_directories_parallel,
                           is_line_in_file)

class TestChunkList(unittest.TestCase):
    def test_chunk_list(self):
        # Test with a list that is evenly divisible by n
        lst1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result1 = chunk_list(lst1, 3)
        self.assertEqual(result1, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]])

        # Test with a list that is not evenly divisible by n
        lst2 = [11, 12, 13, 14, 15, 16, 17]
        result2 = chunk_list(lst2, 4)
        self.assertEqual(result2, [[11, 12, 13, 14], [15, 16, 17]])

        # Test with an empty list
        lst3 = []
        result3 = chunk_list(lst3, 5)
        self.assertEqual(result3, [])

        # Test with a single element list
        lst4 = [99]
        result4 = chunk_list(lst4, 2)
        self.assertEqual(result4, [[99]])

        # Test with a large list and large n value
        lst5 = list(range(1, 1001))
        result5 = chunk_list(lst5, 100)
        expected_result5 = [list(range(1, 101)), list(range(101, 201)), list(range(201, 301)),
                            list(range(301, 401)), list(range(401, 501)), list(range(501, 601)),
                            list(range(601, 701)), list(range(701, 801)), list(range(801, 901)),
                            list(range(901, 1001))]
        self.assertEqual(result5, expected_result5)

class TestSearchLineInFile(unittest.TestCase):
    def setUp(self):
        # Create a temporary file and write some contents for testing
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'test_file.txt')
        with open(self.temp_file, 'w') as file:
            file.write('This is the first line.\n')
            file.write('This is the second line.\n')
            file.write('This is the third line.\n')

    def tearDown(self):
        # Remove the temporary directory and files after the test
        shutil.rmtree(self.temp_dir)

    def test_search_line_in_file(self):
        # Test searching for an existing line in the file
        result = search_line_in_file(self.temp_file, 'second line')
        self.assertTrue(result)

        # Test searching for a non-existing line in the file
        result = search_line_in_file(self.temp_file, 'fourth line')
        self.assertFalse(result)

    def test_search_line_in_file_with_depth(self):
        # Test searching for a line with a specified depth
        result = search_line_in_file(self.temp_file, 'first line', search_depth=2, reverse=True)
        self.assertFalse(result)

    def test_search_line_in_file_reverse(self):
        # Test searching for a line in reverse order
        result = search_line_in_file(self.temp_file, 'third line', reverse=True)
        self.assertTrue(result)

    def test_search_line_in_file_file_not_found(self):
        # Test handling of file not found scenario
        result = search_line_in_file('non_existing_file.txt', 'line')
        self.assertFalse(result)

class TestParseLines(unittest.TestCase):
    def test_parse_lines(self):
        flist = [
            "Header line\n",
            "Trigger Start line\n",
            "1.0 2.0 3.0\n",
            "4.0 5.0 6.0\n",
            "Trigger End line\n",
            "Footer line\n"
        ]
        trigger_start = "Trigger Start"
        trigger_end = "Trigger End"

        result = parse_lines(flist, trigger_start, trigger_end)
        expected = [["1.0 2.0 3.0\n", "4.0 5.0 6.0\n"]]
                
        np.testing.assert_array_equal(result, expected)

    def test_parse_lines_triggers_but_no_data(self):
        flist = [
            "Header line\n",
            "Trigger Start line\n",
            "Trigger End line\n",
            "Footer line\n"
        ]
        trigger_start = "Trigger Start"
        trigger_end = "Trigger End"

        result = parse_lines(flist, trigger_start, trigger_end)
        expected = [[]]
        
        np.testing.assert_array_equal(result, expected)
        
    def test_parse_lines_no_data(self):
        flist = [
            "Header line\n",
            "Footer line\n"
        ]
        trigger_start = "Trigger Start"
        trigger_end = "Trigger End"

        result = parse_lines(flist, trigger_start, trigger_end)
        expected = []
        
        np.testing.assert_array_equal(result, expected)
    
    def test_parse_lines_no_endtrigger(self):
        flist = [
            "Header line\n",
            "Trigger Start",
            "1.0 2.0 3.0\n"
        ]
        trigger_start = "Trigger Start"
        trigger_end = "Trigger End"

        result = parse_lines(flist, trigger_start, trigger_end)
        expected = [["1.0 2.0 3.0\n"]]
        
        np.testing.assert_array_equal(result, expected)
                
    def test_parse_lines_multiple_blocks(self):
        flist = [
            "Header line\n",
            "Trigger Start line\n",
            "1.0 2.0 3.0\n",
            "4.0 5.0 6.0\n",
            "Trigger End line\n",
            "Some other line\n",
            "Trigger Start line\n",
            "7.0 8.0 9.0\n",
            "10.0 11.0 12.0\n",
            "Trigger End line\n",
            "Footer line\n"
        ]
        trigger_start = "Trigger Start"
        trigger_end = "Trigger End"

        result = parse_lines(flist, trigger_start, trigger_end)
        expected = [["1.0 2.0 3.0\n", "4.0 5.0 6.0\n"],
                    ["7.0 8.0 9.0\n", "10.0 11.0 12.0\n"]]
        
        np.testing.assert_array_equal(result, expected)
        
class TestFindDirectoriesWithFiles(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_files()

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_files(self):
        dir1 = os.path.join(self.temp_dir, "dir1")
        dir2 = os.path.join(self.temp_dir, "dir2")
        dir3 = os.path.join(self.temp_dir, "dir3")
        dir4 = os.path.join(self.temp_dir, "dir4")
        
        os.makedirs(dir1)
        os.makedirs(dir2)
        os.makedirs(dir3)
        os.makedirs(dir4)
        
        with open(os.path.join(dir1, "file1.txt"), 'w') as file:
            file.write("This is file 1 in dir 1")
            
        with open(os.path.join(dir1, "file2.txt"), 'w') as file:
            file.write("This is file 2 in dir 1")
            
        with open(os.path.join(dir2, "file2.txt"), 'w') as file:
            file.write("This is file 2")

        with open(os.path.join(dir3, "file3.txt"), 'w') as file:
            file.write("This is file 3")

        with open(os.path.join(dir4, "file9.txt"), "w") as file:
            file.write("This is file 9")

    def test_find_all_files_present(self):
        result = find_directories_with_files(self.temp_dir, ["file1.txt", "file2.txt", "file3.txt"], all_present=False)
        expected = [os.path.join(self.temp_dir, "dir1"), os.path.join(self.temp_dir, "dir2"), os.path.join(self.temp_dir, "dir3")]
        self.assertTrue(set(result) == set(expected))

    def test_find_some_files_present(self):
        result = find_directories_with_files(self.temp_dir, ["file1.txt", "file2.txt"], all_present=True)
        expected = [os.path.join(self.temp_dir, "dir1")]
        self.assertTrue(set(result) == set(expected))

    def test_find_any_files_present(self):
        result = find_directories_with_files(self.temp_dir, ["file2.txt", "file3.txt"], all_present=False)
        expected = [os.path.join(self.temp_dir, "dir1"), os.path.join(self.temp_dir, "dir2"), os.path.join(self.temp_dir, "dir3")]
        self.assertTrue(set(result) == set(expected))

    def test_find_no_files_present(self):
        result = find_directories_with_files(self.temp_dir, ["file4.txt", "file5.txt"], all_present=True)
        expected = []
        self.assertTrue(set(result) == set(expected))

class TestExtractTarball(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'test_file.txt')
        with open(self.temp_file, 'w') as file:
            file.write('This is the first line.\n')
        self.create_test_tarball()

    def tearDown(self):
        os.remove(self.test_tarball_path)
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_tarball(self):
        self.test_tarball_path = os.path.join(self.temp_dir, "test.tar.gz")
        with tarfile.open(self.test_tarball_path, "w:gz") as tar:
            tar.add(self.temp_file, arcname="dir1/test_file.txt")

    def test_extract_tarball(self):
        extraction_path = os.path.join(self.temp_dir, "extracted")
        extract_tarball(self.test_tarball_path, extraction_path)
        extracted_file_path = os.path.join(extraction_path, "dir1", "test_file.txt")
        self.assertTrue(os.path.exists(extracted_file_path))

class TestFindAndExtractTarballsParallel(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, 'test_file.txt')
        with open(self.temp_file, 'w') as file:
            file.write('This is the first line.\n')
        self.create_test_tarballs()

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_tarballs(self):
        dir1 = os.path.join(self.temp_dir, "dir1")
        dir2 = os.path.join(self.temp_dir, "dir2")
        os.makedirs(dir1)
        os.makedirs(dir2)

        # Create test tarball 1
        test_tarball_path1 = os.path.join(dir1, "test1.tar.gz")
        with tarfile.open(test_tarball_path1, "w:gz") as tar:
            tar.add(self.temp_file, arcname="test_file1.txt")

        # Create test tarball 2
        test_tarball_path2 = os.path.join(dir2, "test2.tar.gz")
        with tarfile.open(test_tarball_path2, "w:gz") as tar:
            tar.add(self.temp_file, arcname="test_file2.txt")

        # Create test tarball 3 (.tar.bz2)
        test_tarball_path3 = os.path.join(dir2, "test3.tar.bz2")
        with tarfile.open(test_tarball_path3, "w:bz2") as tar:
            tar.add(self.temp_file, arcname="test_file3.txt")
            
    def test_find_and_extract_tarballs_parallel(self):
        parent_dir = self.temp_dir
        tarball_extension = ".tar.gz"

        find_and_extract_tarballs_parallel(parent_dir, tarball_extension)
        
        extracted_file_path1 = os.path.join(self.temp_dir, "dir1", "test_file1.txt")
        self.assertTrue(os.path.exists(extracted_file_path1))

        extracted_file_path2 = os.path.join(self.temp_dir, "dir2", "test_file2.txt")
        self.assertTrue(os.path.exists(extracted_file_path2))
        
        extracted_file_path3 = os.path.join(self.temp_dir, "dir2", "test_file3.txt")
        self.assertFalse(os.path.exists(extracted_file_path3))
 
class TestExtractFilesFromTarball(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_tarball()

    def tearDown(self):
        os.remove(self.test_tarball_path)
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_tarball(self):
        self.test_tarball_path = os.path.join(self.temp_dir, "test.tar.gz")
        test_file_path = os.path.join(os.path.dirname(__file__), "test_file.txt")
        
        with open(test_file_path, "w") as file:
            file.write("This is the content of the test file.")
            
        with tarfile.open(self.test_tarball_path, "w:gz") as tar:
            tar.add(test_file_path, arcname="dir1/test_file.txt")

    def test_extract_files_from_tarball(self):
        tarball_filepath = self.test_tarball_path
        filenames = ["test_file.txt"]
        suffix = None

        extracted_filepaths = extract_files_from_tarball(tarball_filepath, filenames, suffix)

        extracted_file_path = os.path.join(self.temp_dir, "dir1", "test_file.txt")
        self.assertTrue(os.path.exists(extracted_file_path))
        self.assertListEqual(extracted_filepaths, [extracted_file_path])
               
class TestExtractFilesFromTarballsParallel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_tarballs()
        self.tarball_paths = [os.path.join(self.temp_dir, "dir1", "test1.tar.gz"),
                              os.path.join(self.temp_dir, "dir2", "test2.tar.gz")]
    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_tarballs(self):
        dir1 = os.path.join(self.temp_dir, "dir1")
        dir2 = os.path.join(self.temp_dir, "dir2")
        os.makedirs(dir1)
        os.makedirs(dir2)

        # Create test tarball 1 (.tar.gz)
        test_tarball_path1 = os.path.join(dir1, "test1.tar.gz")
        with open(os.path.join(dir1, "file1.txt"), 'w') as file:
            file.write("This is file 1")

        with tarfile.open(test_tarball_path1, "w:gz") as tar:
            tar.add(os.path.join(dir1, "file1.txt"), arcname="file1.txt")

        # Create test tarball 2 (.tar.gz)
        test_tarball_path2 = os.path.join(dir2, "test2.tar.gz")
        with open(os.path.join(dir2, "file2.txt"), 'w') as file:
            file.write("This is file 2")

        with tarfile.open(test_tarball_path2, "w:gz") as tar:
            tar.add(os.path.join(dir2, "file2.txt"), arcname="file2.txt")

    def test_extract_files_from_tarballs_parallel(self):
        filenames = [
            ["file1.txt"],  # Extract single file from first tarball
            ["file2.txt"],  # Extract single file from second tarball
        ]
        suffix = False

        extract_files_from_tarballs_parallel(self.tarball_paths, filenames, suffix)

        extracted_file_path1 = os.path.join(self.temp_dir, "dir1", "file1.txt")
        self.assertTrue(os.path.exists(extracted_file_path1))

        extracted_file_path2 = os.path.join(self.temp_dir, "dir2", "file2.txt")
        self.assertTrue(os.path.exists(extracted_file_path2))

    def test_extract_files_with_leading_dot(self):

        filenames = ["./file1.txt", "./file2.txt"]

        extract_files_from_tarballs_parallel(self.tarball_paths, filenames)

        for i, filename in enumerate(filenames):
            extracted_filepath = os.path.join(os.path.dirname(self.tarball_paths[i]), filename[:2])
            self.assertTrue(os.path.exists(extracted_filepath))

    def test_extract_files_no_suffix(self):
        filenames = ["file1.txt", "file2.txt"]

        extract_files_from_tarballs_parallel(self.tarball_paths, filenames)

        for i, filename in enumerate(filenames):
            extracted_filepath = os.path.join(os.path.dirname(self.tarball_paths[i]), filename)
            # print(f"the expected extraction path for this file {filename} is {extracted_filepath}")
            self.assertTrue(os.path.exists(extracted_filepath))

class TestFindAndExtractFilesFromTarballsParallel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_tarballs()

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_tarballs(self):
        dir1 = os.path.join(self.temp_dir, "dir1")
        dir2 = os.path.join(self.temp_dir, "dir2")
        os.makedirs(dir1)
        os.makedirs(dir2)

        # Create test tarball 1 (.tar.gz)
        test_tarball_path1 = os.path.join(dir1, "test1.tar.gz")
        with open(os.path.join(dir1, "file1.txt"), 'w') as file:
            file.write("This is file 1")

        with tarfile.open(test_tarball_path1, "w:gz") as tar:
            tar.add(os.path.join(dir1, "file1.txt"), arcname="file1.txt")

        # Create test tarball 2 (.tar.gz)
        test_tarball_path2 = os.path.join(dir2, "test2.tar.gz")
        with open(os.path.join(dir2, "file2.txt"), 'w') as file:
            file.write("This is file 2")

        with tarfile.open(test_tarball_path2, "w:gz") as tar:
            tar.add(os.path.join(dir2, "file2.txt"), arcname="file2.txt")

    def test_find_and_extract_files_from_tarballs_parallel(self):
        parent_dir = self.temp_dir
        extension = ".tar.gz"
        filenames = ["file1.txt", "file2.txt"]
        suffix = False

        find_and_extract_files_from_tarballs_parallel(parent_dir, extension, filenames, suffix)

        extracted_file_path1 = os.path.join(self.temp_dir, "dir1", "file1.txt")
        self.assertTrue(os.path.exists(extracted_file_path1))

        extracted_file_path2 = os.path.join(self.temp_dir, "dir2", "file2.txt")
        self.assertTrue(os.path.exists(extracted_file_path2))
        
class TestCompressDirectory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_directory()

    def tearDown(self):
        # print(f"{self.temp_dir} is the TestCompressDirectory path")
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_directory(self):
        dir_path = os.path.join(self.temp_dir, "test_dir")
        os.makedirs(dir_path)

        # Create test files in the directory
        file1_path = os.path.join(dir_path, "file1.txt")
        with open(file1_path, 'w') as file:
            file.write("This is file 1")

        file2_path = os.path.join(dir_path, "file2.txt")
        with open(file2_path, 'w') as file:
            file.write("This is file 2")

    def test_compress_directory(self):
        directory_path = os.path.join(self.temp_dir, "test_dir")
        exclude_files = ["file2.txt"]
        exclude_file_patterns = []
        print_message = False
        inside_dir = True

        compress_directory(directory_path, exclude_files, exclude_file_patterns, print_message, inside_dir)

        compressed_file_path = os.path.join(self.temp_dir, "test_dir/test_dir.tar.gz")
        self.assertTrue(os.path.exists(compressed_file_path))

        with tarfile.open(compressed_file_path, "r:gz") as tar:
            file_names = tar.getnames()
            self.assertTrue(any(name.endswith("file1.txt") for name in file_names))
            self.assertFalse(any(name.endswith("file2.txt") for name in file_names))
            
class TestCompressDirectoryParallel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_directories()

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_directories(self):
        dir1_path = os.path.join(self.temp_dir, "dir1")
        dir2_path = os.path.join(self.temp_dir, "dir2")
        os.makedirs(dir1_path)
        os.makedirs(dir2_path)

        # Create test files in the directories
        file1_path = os.path.join(dir1_path, "file1.txt")
        with open(file1_path, "w") as file:
            file.write("This is file 1")

        file2_path = os.path.join(dir2_path, "file2.txt")
        with open(file2_path, "w") as file:
            file.write("This is file 2")

    def test_compress_directory_parallel(self):
        directory_paths = [
            os.path.join(self.temp_dir, "dir1"),
            os.path.join(self.temp_dir, "dir2")
        ]
        exclude_files = [["file1.txt"], ["file2.txt"]]
        exclude_file_patterns = []
        print_message = [False]
        inside_dir = [True]

        compress_directory_parallel(directory_paths, exclude_files, exclude_file_patterns, print_message, inside_dir)

        compressed_file_path1 = os.path.join(self.temp_dir, "dir1/dir1.tar.gz")
        self.assertTrue(os.path.exists(compressed_file_path1))

        with tarfile.open(compressed_file_path1, "r:gz") as tar:
            file_names = tar.getnames()
            self.assertTrue(any(name.endswith("file1.txt") for name in file_names))
            self.assertFalse(any(name.endswith("file2.txt") for name in file_names))

        compressed_file_path2 = os.path.join(self.temp_dir, "dir2/dir2.tar.gz")
        self.assertTrue(os.path.exists(compressed_file_path2))

        with tarfile.open(compressed_file_path2, "r:gz") as tar:
            file_names = tar.getnames()
            self.assertFalse(any(name.endswith("file1.txt") for name in file_names))
            self.assertTrue(any(name.endswith("file2.txt") for name in file_names))

class TestCleanupDir(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_files()

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_files(self):
        file1_path = os.path.join(self.temp_dir, "file1.txt")
        with open(file1_path, "w") as file:
            file.write("This is file 1")

        file2_path = os.path.join(self.temp_dir, "file2.txt")
        with open(file2_path, "w") as file:
            file.write("This is file 2")

        file3_path = os.path.join(self.temp_dir, "file3.txt")
        with open(file3_path, "w") as file:
            file.write("This is file 3")

    def test_cleanup_dir_keep(self):
        directory_path = self.temp_dir
        keep = True
        files = ["file1.txt"]
        file_patterns = []
        
        self.create_test_files()
        cleanup_dir(directory_path, keep, files, file_patterns)

        file1_path = os.path.join(self.temp_dir, "file1.txt")
        self.assertTrue(os.path.exists(file1_path))

        file2_path = os.path.join(self.temp_dir, "file2.txt")
        self.assertFalse(os.path.exists(file2_path))

        file3_path = os.path.join(self.temp_dir, "file3.txt")
        self.assertFalse(os.path.exists(file3_path))

    def test_cleanup_dir_remove(self):
        self.create_test_files()
        directory_path = self.temp_dir
        keep = False
        files = ["file1.txt"]
        file_patterns = []

        cleanup_dir(directory_path, keep, files, file_patterns)
        file1_path = os.path.join(self.temp_dir, "file1.txt")
        self.assertFalse(os.path.exists(file1_path))

        file2_path = os.path.join(self.temp_dir, "file2.txt")
        self.assertTrue(os.path.exists(file2_path))

        file3_path = os.path.join(self.temp_dir, "file3.txt")
        self.assertTrue(os.path.exists(file3_path))
        
class TestCompressAndCleanup(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_files()

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_files(self):
        file1_path = os.path.join(self.temp_dir, "file1.txt")
        with open(file1_path, "w") as file:
            file.write("This is file 1")

        file2_path = os.path.join(self.temp_dir, "file2.txt")
        with open(file2_path, "w") as file:
            file.write("This is file 2")

        file3_path = os.path.join(self.temp_dir, "file3.txt")
        with open(file3_path, "w") as file:
            file.write("This is file 3")

    def test_compress_and_cleanup_keep(self):
        directory_path = self.temp_dir
        exclude_files_from_tarball = []
        exclude_filepatterns_from_tarball = []
        keep_after = True
        files = ["file1.txt"]
        file_patterns = []
        print_msg = False
        inside_dir = True

        compress_and_cleanup(directory_path, exclude_files_from_tarball, exclude_filepatterns_from_tarball,
                             keep_after, files, file_patterns, print_msg, inside_dir)

        file1_path = os.path.join(self.temp_dir, "file1.txt")
        self.assertTrue(os.path.exists(file1_path))

        file2_path = os.path.join(self.temp_dir, "file2.txt")
        self.assertFalse(os.path.exists(file2_path))

        file3_path = os.path.join(self.temp_dir, "file3.txt")
        self.assertFalse(os.path.exists(file3_path))

        compressed_file_path = os.path.join(self.temp_dir, os.path.basename(self.temp_dir) + ".tar.gz")
        self.assertTrue(os.path.exists(compressed_file_path))

    def test_compress_and_cleanup_remove(self):
        directory_path = self.temp_dir
        exclude_files_from_tarball = []
        exclude_filepatterns_from_tarball = []
        keep_after = False
        files = ["file1.txt"]
        file_patterns = []
        print_msg = False
        inside_dir = True

        compress_and_cleanup(directory_path, exclude_files_from_tarball, exclude_filepatterns_from_tarball,
                             keep_after, files, file_patterns, print_msg, inside_dir)

        file1_path = os.path.join(self.temp_dir, "file1.txt")
        self.assertFalse(os.path.exists(file1_path))

        file2_path = os.path.join(self.temp_dir, "file2.txt")
        self.assertTrue(os.path.exists(file2_path))

        file3_path = os.path.join(self.temp_dir, "file3.txt")
        self.assertTrue(os.path.exists(file3_path))

        compressed_file_path = os.path.join(self.temp_dir, os.path.basename(self.temp_dir) + ".tar.gz")
        self.assertTrue(os.path.exists(compressed_file_path))

        # compressed_dir_path = os.path.join(self.temp_dir, "test_dir")
        # self.assertFalse(os.path.exists(compressed_dir_path))

class TestCompressDirectoryParallel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_directories()

    def tearDown(self):
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_directories(self):
        dir1_path = os.path.join(self.temp_dir, "dir1")
        os.makedirs(dir1_path)

        file1_path = os.path.join(dir1_path, "file1.txt")
        with open(file1_path, "w") as file:
            file.write("This is file 1")

        dir2_path = os.path.join(self.temp_dir, "dir2")
        os.makedirs(dir2_path)

        file2_path = os.path.join(dir2_path, "file2.txt")
        with open(file2_path, "w") as file:
            file.write("This is file 2")

    def test_compress_directory_parallel(self):
        directory_paths = [os.path.join(self.temp_dir, "dir1"), os.path.join(self.temp_dir, "dir2")]
        exclude_files = []
        exclude_file_patterns = []
        print_message = False
        inside_dir = True

        compress_directory_parallel(directory_paths, exclude_files, exclude_file_patterns, print_message, inside_dir)

        compressed_file_path1 = os.path.join(self.temp_dir, "dir1/dir1.tar.gz")
        self.assertTrue(os.path.exists(compressed_file_path1))

        compressed_file_path2 = os.path.join(self.temp_dir, "dir2/dir2.tar.gz")
        self.assertTrue(os.path.exists(compressed_file_path2))

        with tarfile.open(compressed_file_path1, "r:gz") as tar:
            tar.extractall(self.temp_dir)

        extracted_file_path1 = os.path.join(self.temp_dir, "dir1", "file1.txt")
        self.assertTrue(os.path.exists(extracted_file_path1))

        with tarfile.open(compressed_file_path2, "r:gz") as tar:
            tar.extractall(self.temp_dir)

        extracted_file_path2 = os.path.join(self.temp_dir, "dir2", "file2.txt")
        self.assertTrue(os.path.exists(extracted_file_path2))

        self.assertTrue(filecmp.cmp(extracted_file_path1, os.path.join(self.temp_dir, "dir1", "file1.txt")))
        self.assertTrue(filecmp.cmp(extracted_file_path2, os.path.join(self.temp_dir, "dir2", "file2.txt")))

class TestFindAndCompressDirectoriesParallel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_directories()

    def tearDown(self):
        # print(f"{self.temp_dir} is the TestCompressDirectory path")
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)

    def create_test_directories(self):
        dir1_path = os.path.join(self.temp_dir, "dir1")
        os.makedirs(dir1_path)

        file1_path = os.path.join(dir1_path, "file1.txt")
        with open(file1_path, "w") as file:
            file.write("This is file 1")

        dir2_path = os.path.join(self.temp_dir, "dir2")
        os.makedirs(dir2_path)

        file2_path = os.path.join(dir2_path, "file2.txt")
        with open(file2_path, "w") as file:
            file.write("This is file 2")

        dir3_path = os.path.join(self.temp_dir, "dir3")
        os.makedirs(dir3_path)

        file3_path = os.path.join(dir3_path, "file3.txt")
        with open(file3_path, "w") as file:
            file.write("This is file 3")

    def test_find_and_compress_directories_parallel(self):
        parent_dir = self.temp_dir
        valid_dir_if_filenames = ["file1.txt", "file2.txt"]
        exclude_files_from_tarball = []
        exclude_filepatterns_from_tarball = []
        keep_after = True
        files = []
        file_patterns = []
        print_msg = False
        inside_dir = True
        all_present = False
        
        find_and_compress_directories_parallel(parent_dir, valid_dir_if_filenames, all_present, exclude_files_from_tarball,
                                               exclude_filepatterns_from_tarball, keep_after, files, file_patterns,
                                               print_msg, inside_dir)

        compressed_file_path1 = os.path.join(self.temp_dir, "dir1/dir1.tar.gz")
        self.assertTrue(os.path.exists(compressed_file_path1))

        compressed_file_path2 = os.path.join(self.temp_dir, "dir2/dir2.tar.gz")
        self.assertTrue(os.path.exists(compressed_file_path2))

        compressed_file_path3 = os.path.join(self.temp_dir, "dir3/dir3.tar.gz")
        self.assertFalse(os.path.exists(compressed_file_path3))

        with tarfile.open(compressed_file_path1, "r:gz") as tar:
            tar.extractall(self.temp_dir)

        extracted_file_path1 = os.path.join(self.temp_dir, "dir1", "file1.txt")
        self.assertTrue(os.path.exists(extracted_file_path1))

        with tarfile.open(compressed_file_path2, "r:gz") as tar:
            tar.extractall(self.temp_dir)

        extracted_file_path2 = os.path.join(self.temp_dir, "dir2", "file2.txt")
        self.assertTrue(os.path.exists(extracted_file_path2))
        
        self.assertFalse(os.path.exists(compressed_file_path3))

        self.assertTrue(filecmp.cmp(extracted_file_path1, os.path.join(self.temp_dir, "dir1", "file1.txt")))
        self.assertTrue(filecmp.cmp(extracted_file_path1, os.path.join(self.temp_dir, "dir1", "file1.txt")))                  
          
class TestIsLineInFile(unittest.TestCase):

    def test_exact_match_line_present(self):
        # Create a temporary file with some lines
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("Line 1\n")
            temp_file.write("Line 2\n")
            temp_file.write("Line 3\n")
            temp_file.write("Line 4\n")

        filepath = temp_file.name
        line_to_search = "Line 2"
        exact_match = True

        result = is_line_in_file(filepath, line_to_search, exact_match)
        self.assertTrue(result)

    def test_exact_match_line_not_present(self):
        filepath = "path/to/nonexistent/file.txt"
        line_to_search = "Hello, world!"
        exact_match = True

        result = is_line_in_file(filepath, line_to_search, exact_match)
        self.assertFalse(result)

    def test_partial_match_line_present(self):
        # Create a temporary file with some lines
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("Hello, world!\n")
            temp_file.write("Goodbye, world!\n")

        filepath = temp_file.name
        line_to_search = "Hello"
        exact_match = False

        result = is_line_in_file(filepath, line_to_search, exact_match)
        self.assertTrue(result)

    def test_partial_match_line_not_present(self):
        # Create a temporary file with some lines
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("Hello, world!\n")
            temp_file.write("Goodbye, world!\n")

        filepath = temp_file.name
        line_to_search = "Hey"
        exact_match = False

        result = is_line_in_file(filepath, line_to_search, exact_match)
        self.assertFalse(result)
                                    
if __name__ == '__main__':
    unittest.main()