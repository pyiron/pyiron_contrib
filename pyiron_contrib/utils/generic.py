import os
import tarfile
from multiprocessing import Pool, cpu_count
import fnmatch

from functools import wraps

from utils.parallel import parallelise

from monty.os.path import find_exts
from monty.io import zopen

def chunk_list(lst, n):
    """
    Split a list into smaller chunks with a maximum size of n.

    Parameters:
        lst (list): The list to be chunked.
        n (int): The maximum size of each chunk.

    Returns:
        list: A list of sub-lists, each containing elements from the original list.

    Example:
        >>> my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> chunked_list = chunk_list(my_list, 3)
        >>> print(chunked_list)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def search_line_in_file(filename, line_to_search, search_depth=None, reverse=True):
    """
    Searches for a specific line in a file.

    Parameters:
        filename (str): The path of the file to search.
        line_to_search (str): The line to search within the file.
        search_depth (int, optional): The maximum number of lines to search. Defaults to None (search all lines).
        reverse (bool, optional): Determines whether to search in reverse order. Defaults to True.

    Returns:
        bool: True if the line is found, False otherwise.

    Usage:
        # Search for a specific line in a file
        search_line_in_file("/path/to/file.txt", "target line")

        # Search for a specific line in a file, limiting the search depth
        search_line_in_file("/path/to/file.txt", "target line", search_depth=100)

        # Search for a specific line in a file in reverse order
        search_line_in_file("/path/to/file.txt", "target line", reverse=True)

    Note:
        - The function opens the specified file in read mode and reads its lines.
        - If `reverse` is True, it searches the lines in reverse order.
        - If `search_depth` is provided, it limits the maximum number of lines to search.
        - The function checks if the `line_to_search` is present in each line, stripped of leading and trailing whitespace.
        - If the line is found, it returns True. Otherwise, it returns False.
        - If the file is not found, the function returns False.
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            if reverse:
                lines = reversed(lines)  # Reverse the lines           
            count = 0
            for line in lines:
                if search_depth is not None and count >= search_depth:
                    break                
                if line_to_search in line.strip():
                    return True                
                count += 1            
            return False
    except FileNotFoundError:
        # print("File not found:", filename)
        return False

def parse_lines(flist, trigger_start, trigger_end, recursive=True):
    """
    Parses lines from a list of strings based on start and end triggers and returns the parsed data.

    Parameters:
        flist (list): A list of strings representing the lines to parse.
        trigger_start (str): The trigger string indicating the start of the data block.
        trigger_end (str): The trigger string indicating the end of the data block.
        recursive (bool, optional): Determines whether to parse recursively for multiple data blocks. Defaults to True.

    Returns:
        list: A list of parsed data blocks.

    Usage:
        # Parse lines between specific start and end triggers
        parse_lines(lines, "START", "END")

        # Parse lines between specific start and end triggers recursively
        parse_lines(lines, "START", "END", recursive=True)

        # Parse lines between specific start and end triggers without recursion
        parse_lines(lines, "START", "END", recursive=False)

    Note:
        - The function iterates over the lines in the `flist` list and identifies the data blocks based on the specified start and end triggers.
        - It returns a list of parsed data blocks, where each data block is a list of lines between a start trigger and an end trigger.
        - If `recursive` is True, the function continues parsing for multiple data blocks, even after finding an end trigger.
        - If `recursive` is False, the function stops parsing after finding the first end trigger.
        - If no data blocks are found, an empty list is returned.
    """
    parsing = False
    any_data = False
    data = []
    for line in flist:
        if trigger_end in line:
            parsing = False
            data.append(data_block)
            if not recursive:
                break
            else:
                continue
        if parsing:
            data_block.append(line)
        if trigger_start in line:
            any_data = True
            parsing = True
            data_block = []
    if not any_data:
        data = []
    if parsing and not trigger_end in line:
        data.append(data_block)

    return data

def find_directories_with_files(parent_dir, filenames, all_present=True):
    """
    Finds directories in a parent directory that contain specified files.

    Parameters:
        parent_dir (str): The path of the parent directory to search for directories.
        filenames (list): A list of filenames to search for within the directories.
        all_present (bool, optional): Determines whether all the filenames should be present in each directory. Defaults to True.

    Returns:
        list: A list of directories that contain the specified files.

    Usage:
        # Find directories containing specific files, requiring all files to be present
        find_directories_with_files("/path/to/parent", ["file1.txt", "file2.jpg"])

        # Find directories containing specific files, requiring at least one file to be present
        find_directories_with_files("/path/to/parent", ["file1.txt", "file2.jpg"], all_present=False)

    Note:
        - The function searches for directories in the `parent_dir` directory using the `os.walk` function.
        - It checks if the specified `filenames` are present in each directory.
        - If `all_present` is True, the function includes only the directories that contain all the specified files.
        - If `all_present` is False, the function includes the directories that contain at least one of the specified files.
        - The function returns a list of directories that meet the specified conditions.
    """
    directories = []
    file_set = set(filenames)  # Convert filenames to a set for efficient membership checking

    for root, dirs, files in os.walk(parent_dir):
        # Check if the intersection of file_set and files is not empty
        if all_present and file_set.intersection(files) == file_set:
            directories.append(root)
        elif not all_present and file_set.intersection(files):
            directories.append(root)

    return directories

def extract_tarball(tarball_filepath, extraction_path):
    """
    Extracts the contents of a tarball file to the specified extraction path.

    Parameters:
        tarball_filepath (str): The path of the tarball file to extract.
        extraction_path (str): The path where the contents of the tarball will be extracted.

    Usage:
        # Extract a tarball file to a specific extraction path
        extract_tarball("/path/to/tarball.tar.gz", "/path/to/extraction")

    Note:
        - The function opens the tarball file using the `tarfile` module with read mode and gzip compression.
        - It extracts all the contents of the tarball to the specified extraction path.
        - The directory structure within the tarball will be preserved in the extraction process.
    """
    try:
        with tarfile.open(tarball_filepath, "r:gz") as tar:
            tar.extractall(extraction_path)
    except:
        # EOFError
        # EOFError: Compressed file ended before the end-of-stream marker was reached
        a = 0

def find_and_extract_tarballs_parallel(parent_dir, extensions=(".tar.gz")):
    """
    Finds tarball files with specified extensions in a directory and extracts them in parallel.

    Parameters:
        parent_dir (str): The path of the parent directory to search for tarball files.
        extensions (tuple or list, optional): A tuple or list of extensions to search for. Defaults to (".tar.gz").

    Usage:
        # Find and extract tarball files with default extensions in a directory
        find_and_extract_tarballs_parallel("/path/to/directory")

        # Find and extract tarball files with custom extensions in a directory
        find_and_extract_tarballs_parallel("/path/to/directory", extensions=(".tar", ".tgz"))

    Note:
        - The function searches for tarball files with the specified extensions in the `parent_dir` directory using the `find_exts` function.
        - It creates a list of extraction filepaths by getting the directory paths of the tarball files.
        - The `parallelise` function is used to perform the extraction process in parallel, using the `extract_tarball` function for each tarball file.
        - The function extracts the tarball files in parallel, preserving the directory structure within the tarballs.
    """
    filepaths = find_exts(top=parent_dir, exts=extensions)
    extraction_filepaths = [os.path.dirname(filepath) for filepath in filepaths]
    parallelise(extract_tarball, filepaths, extraction_filepaths)

def extract_files_from_tarball(tarball_filepath, filenames, suffix=None, prefix=None):
    """
    Extracts specific files from a tarball file and optionally renames them with a suffix.

    Parameters:
        tarball_filepath (str): The path of the tarball file from which to extract files.
        filenames (list): A list of filenames to extract from the tarball.
        suffix (str, optional): An optional suffix to append to the extracted filenames. Defaults to None.

    Returns:
        list: A list of extracted filepaths.

    Usage:
        # Extract specific files from a tarball without renaming
        extract_files_from_tarball("/path/to/tarball.tar.gz", ["file1.txt", "file2.jpg"])

        # Extract specific files from a tarball and add a suffix to the extracted filenames
        extract_files_from_tarball("/path/to/tarball.tar.gz", ["file1.txt", "file2.jpg"], suffix="extracted")

    Note:
        - The function opens the tarball file using the `tarfile` module with read mode and gzip compression.
        - It iterates over the specified `filenames` and extracts the matching files from the tarball to the directory containing the tarball file.
        - If the extracted filename starts with "./", it is modified to remove the leading "./".
        - If `suffix` is provided, the extracted file is renamed by appending the suffix to the base filename.
        - The function returns a list of the extracted filepaths.
    """
    with tarfile.open(tarball_filepath, "r:gz") as tar:
        extracted_filepaths = []
        for filename in filenames:
            matching_names = [name for name in tar.getnames() if name.endswith(filename)]
            for name in matching_names:
                tar.extract(name, path=os.path.dirname(tarball_filepath))
                if name.startswith("./"):
                    extracted_filepath = os.path.join(os.path.dirname(tarball_filepath), name[2:])
                else:
                    extracted_filepath = os.path.join(os.path.dirname(tarball_filepath), name)
                if suffix:
                    new_path = os.path.join(os.path.dirname(extracted_filepath), os.path.basename(extracted_filepath) + "_" + suffix)
                    os.rename(extracted_filepath, new_path)
                    extracted_filepath = new_path
                if prefix:
                    new_path = os.path.join(prefix + "_" + os.path.dirname(extracted_filepath), os.path.basename(extracted_filepath))
                    os.rename(extracted_filepath, new_path)
                    extracted_filepath = new_path
                extracted_filepaths.append(extracted_filepath)

    return extracted_filepaths

def extract_files_from_tarballs_parallel(tarball_paths, filenames, suffix=False):
    """
    Extracts specific files from multiple tarball files in parallel and optionally renames them with suffixes.

    Parameters:
        tarball_paths (list): A list of tarball file paths.
        filenames (list or str): The filenames to extract from the tarball(s). If a list, it should match the number of tarball_paths.
                                 If a string, it will be used for all tarball_paths.
        suffix (bool, optional): Determines whether to append suffixes to the extracted filenames. Defaults to False.

    Usage:
        # Extract "file1.txt" and "file2.jpg" from two tarball files in parallel
        tarballs = ["/path/to/tarball1.tar.gz", "/path/to/tarball2.tar.gz"]
        extract_files_from_tarballs_parallel(tarballs, ["file1.txt", "file2.jpg"], suffix=True)

        # Extract "data.csv" from three tarball files without appending suffixes
        tarballs = ["/path/to/tarball1.tar.gz", "/path/to/tarball2.tar.gz", "/path/to/tarball3.tar.gz"]
        extract_files_from_tarballs_parallel(tarballs, "data.csv", suffix=False)

    Note:
        - The function first validates the format of the filenames input.
        - If filenames is a list, it checks if it is a list of strings or a list of lists with lengths matching the number of tarball_paths.
        - If filenames is a string, it creates a list with the same filename for each tarball_path.
        - If suffix is True, it generates suffixes from the tarball filenames by removing the ".tar" extension.
        - If suffix is False, None is used as the suffix for all tarball_paths.
        - Finally, it parallelizes the extraction process by calling the `parallelise()` function with the `extract_files_from_tarball` function,
          the lists of tarball_paths, filenames, and suffixes as arguments.
    """
    if isinstance(filenames, list):
        if isinstance(filenames[0], str):
            filenames = [filenames] * len(tarball_paths)
        elif isinstance(filenames[0], list):
            if len(filenames) != len(tarball_paths):
                raise ValueError("The length of filenames should match the number of tarball_paths.")
        else:
            raise ValueError("Invalid format for filenames.")
    else:
        raise ValueError("Invalid format for filenames.")

    if suffix:
        suffixes = [os.path.basename(filepath).split(".tar")[0] for filepath in tarball_paths]
    else:
        suffixes = [None for _ in tarball_paths]

    parallelise(extract_files_from_tarball, tarball_paths, filenames, suffixes)

def find_and_extract_files_from_tarballs_parallel(parent_dir,
                                                  extension=".tar.gz",
                                                  filenames=[],
                                                  suffix=False,
                                                  prefix=False):
    """
    Finds and extracts specific files from multiple tarball files within a parent directory using parallel processing.

    Parameters:
        parent_dir (str): The path of the parent directory to search for tarball files.
        extension (str, optional): The file extension of the tarball files to search for. Defaults to ".tar.gz".
        filenames (str or list, optional): The filenames to extract from the tarball(s). If a string, it will be used for all tarball files.
                                           If a list, it should have the same length as the number of tarball files found in the parent directory.
                                           Defaults to an empty list, which means all files will be extracted.
        suffix (bool, optional): Determines whether to append suffixes to the extracted filenames. Defaults to False.
        prefix (bool, optional): Determines whether to prepend prefixes to the extracted filenames. Defaults to False.

    Usage:
        # Extract all files from .tar.gz files within a parent directory in parallel
        find_and_extract_files_from_tarballs_parallel("/path/to/parent_directory")

        # Extract specific files from .tar.gz files within a parent directory in parallel, with suffixes appended
        find_and_extract_files_from_tarballs_parallel("/path/to/parent_directory", filenames=["file1.txt", "file2.jpg"], suffix=True)

    Note:
        - The function searches for tarball files within the specified `parent_dir` using the provided `extension`.
        - It finds and extracts specific `filenames` from the tarball files, either all files or the specified files.
        - If `suffix` is True, the extracted filenames will be appended with suffixes.
        - The extraction process is parallelized using the `parallelise()` function and the `extract_files_from_tarball` function.
    """
    filepaths = find_exts(top=parent_dir, exts=(extension))
    filenames = [filenames]*len(filepaths)
    if suffix:
        # Really ugly so this only works with .tar.gz files for now
        suffixes = [os.path.basename(filepath).split(".tar")[0] for filepath in filepaths]
    else:
        suffixes = [None for _ in filepaths]
        
    if prefix:
        # Really ugly so this only works with .tar.gz files for now
        prefixes = [os.path.basename(filepath).split(".tar")[0] for filepath in filepaths]
    else:
        prefixes = [None for _ in filepaths]        
    parallelise(extract_files_from_tarball, filepaths, filenames, suffixes, prefixes)
    
def compress_directory(directory_path,
                       exclude_files = [],
                       exclude_file_patterns = [],
                       print_message=True,
                       inside_dir=True):
    """
    Compresses a directory and its contents into a tarball with gzip compression.

    Parameters:
        directory_path (str): The path of the directory to compress.
        exclude_files (list, optional): A list of filenames to exclude from the compression. Defaults to an empty list.
        exclude_file_patterns (list, optional): A list of file patterns (glob patterns) to match against filenames and exclude from the compression. Defaults to an empty list.
        print_message (bool, optional): Determines whether to print a message indicating the compression. Defaults to True.
        inside_dir (bool, optional): Determines whether the output tarball should be placed inside the source directory or in the same directory as the source directory. Defaults to True.

    Usage:
        # Compress a directory and place the resulting tarball inside the directory
        compress_directory("/path/to/source_directory")

        # Compress a directory and place the resulting tarball in the same directory as the source directory
        compress_directory("/path/to/source_directory", inside_dir=False)

        # Compress a directory and exclude specific files from the compression
        compress_directory("/path/to/source_directory", exclude_files=["file1.txt", "file2.jpg"])

        # Compress a directory and exclude files matching specific file patterns from the compression
        compress_directory("/path/to/source_directory", exclude_file_patterns=["*.txt", "*.log"], inside_dir=False)

    Note:
        - The function creates a tarball with gzip compression of the directory and its contents.
        - The resulting tarball will be placed either inside the source directory (if inside_dir is True) or in the same directory as the source directory (if inside_dir is False).
        - Files specified in the `exclude_files` list and those matching the `exclude_file_patterns` will be excluded from the compression.
        - The `print_message` parameter controls whether a message indicating the compression is printed. By default, it is set to True.
    """
    if inside_dir:
        output_file = os.path.join(directory_path, os.path.basename(directory_path) + '.tar.gz')
    else:
        output_file = os.path.join(os.path.dirname(directory_path), os.path.basename(directory_path) + '.tar.gz')
    with tarfile.open(output_file, "w:gz") as tar:
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Exclude the output tarball from being added
                if file_path == output_file:
                    continue
                if any(fnmatch.fnmatch(file, pattern) for pattern in exclude_file_patterns):
                    continue
                if file in exclude_files:
                    continue
                arcname = os.path.join(os.path.basename(directory_path), os.path.relpath(file_path, directory_path))
                tar.add(file_path, arcname=arcname)
                # tar.add(file_path, arcname=os.path.relpath(file_path, directory_path))      
                # print(f"{file} added")
    if print_message:
        print(f"Compressed directory: {directory_path}")

def compress_directory_parallel(directory_paths,
                                exclude_files=None,
                                exclude_file_patterns=None,
                                print_message=None,
                                inside_dir=None):
    """
    Compresses multiple directories and their contents into tarballs with gzip compression in parallel.

    Parameters:
        directory_paths (list): A list of directory paths to compress.
        exclude_files (str or list, optional): A filename or a list of filenames to exclude from each compression. Defaults to None.
        exclude_file_patterns (str or list, optional): A file pattern (glob pattern) or a list of patterns to match against filenames and exclude from each compression. Defaults to None.
        print_message (bool or list, optional): Determines whether to print a message indicating the compression for each directory. Defaults to None.
        inside_dir (bool or list, optional): Determines whether the output tarball should be placed inside each directory or in the same directory as each directory. Defaults to None.

    Usage:
        # Compress multiple directories and place the resulting tarballs inside each directory
        compress_directory_parallel(["/path/to/dir1", "/path/to/dir2", "/path/to/dir3"])

        # Compress multiple directories and place the resulting tarballs in the same directory as each directory
        compress_directory_parallel(["/path/to/dir1", "/path/to/dir2", "/path/to/dir3"], inside_dir=False)

        # Compress multiple directories and exclude specific files from each compression
        compress_directory_parallel(["/path/to/dir1", "/path/to/dir2", "/path/to/dir3"], exclude_files=["file1.txt", "file2.jpg"])

        # Compress multiple directories and exclude files matching specific file patterns from each compression
        compress_directory_parallel(["/path/to/dir1", "/path/to/dir2", "/path/to/dir3"], exclude_file_patterns=["*.txt", "*.log"], inside_dir=False)

    Note:
        - The function compresses each directory and its contents into a tarball with gzip compression using the `compress_directory` function.
        - Files specified in the `exclude_files` parameter and those matching the `exclude_file_patterns` will be excluded from each compression.
        - The resulting tarballs will be placed either inside each source directory (if inside_dir is True) or in the same directory as each source directory (if inside_dir is False).
        - The `print_message` parameter controls whether a message indicating the compression is printed for each directory.
        - The function parallelizes the compression process using the `parallelise()` function and the `compress_directory` function.
    """
    exclude_files = [exclude_files] * len(directory_paths) if exclude_files is not None else exclude_files
    exclude_file_patterns = [exclude_file_patterns] * len(directory_paths) if exclude_file_patterns is not None else exclude_file_patterns
    print_message = [print_message] * len(directory_paths) if print_message is not None else print_message
    inside_dir = [inside_dir] * len(directory_paths) if inside_dir is not None else inside_dir
    parallelise(compress_directory, directory_paths, exclude_files, exclude_file_patterns, print_message, inside_dir)
       
def cleanup_dir(directory_path, keep=True, files=[], file_patterns=[]):    
    """
    Cleans up files in a directory based on specified conditions.

    Parameters:
        directory_path (str): The path of the directory to perform cleanup.
        keep (bool, optional): Determines whether to keep the directory. If True, the directory will be kept. If False, it will be deleted. Defaults to True.
        files (list, optional): A list of filenames to exclude from the cleanup operation. Defaults to an empty list.
        file_patterns (list, optional): A list of file patterns (glob patterns) to match against filenames and exclude from the cleanup operation. Defaults to an empty list.

    Usage:
        # Clean up files in a directory, keeping all files
        cleanup_dir("/path/to/directory")

        # Clean up files in a directory, excluding specific files
        cleanup_dir("/path/to/directory", files=["file1.txt", "file2.jpg"])

        # Clean up files in a directory, deleting the directory and cleaning up specific files
        cleanup_dir("/path/to/directory", keep=False, files=["file1.txt"])

    Note:
        - The function performs a cleanup operation in the specified directory based on the provided conditions.
        - If `keep` is True, the directory will be kept, and only files not specified in `files` or matching any of the `file_patterns` will be removed.
        - If `keep` is False, the directory will be deleted, and all files specified in `files` or matching any of the `file_patterns` will be removed.
        - The `files` parameter is a list of filenames to exclude from the cleanup operation.
        - The `file_patterns` parameter is a list of file patterns (glob patterns) to match against filenames and exclude from the cleanup operation.
    """
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            if keep:
                matched_pattern = False
                for pattern in file_patterns:
                    if fnmatch.fnmatch(file, pattern):
                        matched_pattern = True
                        break
                if matched_pattern or file in files:
                    continue
                os.remove(file_path)
            else:
                should_remove = False
                for pattern in file_patterns:
                    if fnmatch.fnmatch(file, pattern):
                        should_remove = True
                        break
                if should_remove or file in files:
                    os.remove(file_path)
                
def compress_and_cleanup(directory_path, 
                         exclude_files_from_tarball=[],
                         exclude_filepatterns_from_tarball=[],
                         keep_after=True,
                         files=[],
                         file_patterns=[],
                         print_msg=False,
                         inside_dir=True):
    """
    Compresses a directory and its contents into a tarball with gzip compression, and performs cleanup operations.

    Parameters:
        directory_path (str): The path of the directory to compress.
        exclude_files_from_tarball (list, optional): A list of filenames to exclude from the compression. Defaults to an empty list.
        exclude_filepatterns_from_tarball (list, optional): A list of file patterns (glob patterns) to match against filenames and exclude from the compression. Defaults to an empty list.
        keep_after (bool, optional): Determines whether to keep the directory after compression. If True, the directory will be kept. If False, it will be deleted. Defaults to True.
        files (list, optional): A list of filenames to include in the cleanup operation. Defaults to an empty list.
        file_patterns (list, optional): A list of file patterns (glob patterns) to match against filenames and include in the cleanup operation. Defaults to an empty list.
        print_msg (bool, optional): Determines whether to print a message indicating the compression for the directory. Defaults to False.
        inside_dir (bool, optional): Determines whether the output tarball should be placed inside the directory or in the same directory as the directory. Defaults to True.

    Usage:
        # Compress a directory and perform cleanup, keeping the directory and cleaning up all files except the compressed tarball
        compress_and_cleanup("/path/to/source_directory")

        # Compress a directory and perform cleanup, deleting the directory and cleaning up specific files
        compress_and_cleanup("/path/to/source_directory", keep_after=False, files=["file1.txt"])

    Note:
        - The function compresses the specified directory and its contents into a tarball with gzip compression using the `compress_directory` function.
        - Files specified in the `exclude_files_from_tarball` list and those matching the `exclude_filepatterns_from_tarball` will be excluded from the compression.
        - After compression, the function performs a cleanup operation using the `cleanup_dir` function to remove files from the directory based on the `keep_after`, `files`, and `file_patterns` parameters.
        - If `keep_after` is True, the directory will be kept, and the tarball filename will be added to the `file_patterns` to ensure it is not deleted.
        - If `keep_after` is False, the directory will be deleted, and only the files specified in `files` or matching the `file_patterns` will be cleaned up.
        - The `print_msg` parameter controls whether a message indicating the compression is printed for the directory.
        - The `inside_dir` parameter determines whether the output tarball should be placed inside the directory (True) or in the same directory as the directory (False).
    """
    compress_directory(directory_path=directory_path,
                       exclude_files=exclude_files_from_tarball,
                       exclude_file_patterns=exclude_filepatterns_from_tarball,
                       print_message=print_msg,
                       inside_dir=inside_dir)
    # Add the newly compressed directory to the exceptions, or we'll remove it!
    if keep_after:
        file_patterns += [f"{os.path.basename(directory_path)}.tar.gz"]
    else:
        file_patterns = file_patterns
    cleanup_dir(directory_path=directory_path,
                keep=keep_after,
                files=files,
                file_patterns=file_patterns)
    
def find_and_compress_directories_parallel(parent_dir,
                                           valid_dir_if_filenames,
                                           all_present=False,
                                           exclude_files_from_tarball=[],
                                           exclude_filepatterns_from_tarball=[],
                                           keep_after=True,
                                           files=[],
                                           file_patterns=[],
                                           print_msg=False,
                                           inside_dir=True):
    """
    Finds directories containing specific files, and compresses each directory and its contents into tarballs with gzip compression in parallel.

    Parameters:
        parent_dir (str): The parent directory to search for directories.
        valid_dir_if_filenames (list): A list of filenames that a directory must contain to be considered valid for compression.
        all_present (bool, optional): Determines whether all filenames in valid_dir_if_filenames must be present in a directory for it to be considered valid. Defaults to False.
        exclude_files_from_tarball (str or list, optional): A filename or a list of filenames to exclude from each compression. Defaults to an empty list.
        exclude_filepatterns_from_tarball (str or list, optional): A file pattern (glob pattern) or a list of patterns to match against filenames and exclude from each compression. Defaults to an empty list.
        keep_after (bool or list, optional): Determines whether to keep the directory after compression. If True, the directory will be kept. If False, it will be deleted. Defaults to True.
        files (str or list, optional): A filename or a list of filenames to include in each compression. Defaults to an empty list, which means all files will be included.
        file_patterns (str or list, optional): A file pattern (glob pattern) or a list of patterns to match against filenames and include in each compression. Defaults to an empty list.
        print_msg (bool or list, optional): Determines whether to print a message indicating the compression for each directory. Defaults to False.
        inside_dir (bool or list, optional): Determines whether the output tarball should be placed inside each directory or in the same directory as each directory. Defaults to True.

    Usage:
        # Find and compress directories containing specific files, placing the resulting tarballs inside each directory
        find_and_compress_directories_parallel("/path/to/parent_directory", ["file1.txt", "file2.jpg"])

        # Find and compress directories containing specific files, excluding certain files from each compression, and placing the resulting tarballs in the same directory as each directory
        find_and_compress_directories_parallel("/path/to/parent_directory", ["file1.txt", "file2.jpg"], exclude_files_from_tarball=["file3.txt"], inside_dir=False)

        # Find and compress directories containing specific files, keeping the directories after compression, and including only specific files in each compression
        find_and_compress_directories_parallel("/path/to/parent_directory", ["file1.txt", "file2.jpg"], keep_after=True, files=["file1.txt"])

    Note:
        - The function searches for directories within the specified `parent_dir` that contain the specified `valid_dir_if_filenames`.
        - The `all_present` parameter determines whether all filenames in `valid_dir_if_filenames` must be present in a directory for it to be considered valid.
        - Files specified in the `exclude_files_from_tarball` parameter and those matching the `exclude_filepatterns_from_tarball` will be excluded from each compression.
        - The `keep_after` parameter determines whether the directory should be kept (True) or deleted (False) after compression.
        - Files specified in the `files` parameter and those matching the `file_patterns` will be included in each compression. If empty, all files will be included.
        - The `print_msg` parameter controls whether a message indicating the compression is printed for each directory.
        - The `inside_dir` parameter determines whether the output tarball should be placed inside each directory (True) or in the same directory as each directory (False).
        - The function parallelizes the compression process using the `parallelise()` function and the `compress_and_cleanup` function.
    """
    # I've no idea how to do this better, lol. I'm assuming kwargs or args or some better way of feeding kwargs into pool.map exists, but I've no idea
    dirs_to_compress = find_directories_with_files(parent_dir=parent_dir, filenames=valid_dir_if_filenames, all_present=all_present)
    exclude_files_from_tarball = [exclude_files_from_tarball] * len(dirs_to_compress)
    exclude_filepatterns_from_tarball = [exclude_filepatterns_from_tarball] * len(dirs_to_compress)
    keep_after = [keep_after] * len(dirs_to_compress)
    files = [files] * len(dirs_to_compress)
    file_patterns = [file_patterns] * len(dirs_to_compress)
    print_msg = [print_msg] * len(dirs_to_compress)
    inside_dir = [inside_dir] * len(dirs_to_compress)
    
    parallelise(compress_and_cleanup,
                dirs_to_compress,
                exclude_files_from_tarball,
                exclude_filepatterns_from_tarball,
                keep_after,
                files,
                file_patterns,
                print_msg,
                inside_dir)

def is_line_in_file(filepath, line, exact_match=True):
    """
    Check if a line is present in a file.

    Args:
        filepath (str): The path to the file.
        line (str): The line to search for in the file.
        exact_match (bool, optional): Determines whether the search should be an exact match (default: True).

    Returns:
        bool: True if the line is found in the file, False otherwise.

    Example:
        >>> filepath = 'path/to/your/file.txt'
        >>> line_to_search = 'Hello, world!'
        >>> exact_match = True  # Toggle this flag to change between exact and partial match

        >>> if is_line_in_file(filepath, line_to_search, exact_match):
        ...     print("Line found in the file.")
        ... else:
        ...     print("Line not found in the file.")
    """
    try:
        with open(filepath, 'r') as file:
            for file_line in file:
                if exact_match and line == file_line.strip():
                    return True
                elif not exact_match and line in file_line:
                    return True
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
    return False
