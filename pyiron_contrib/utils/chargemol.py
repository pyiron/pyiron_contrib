from utils.generic import parse_lines, search_line_in_file, find_filepaths_in_dir_with_files, extract_file_from_tarball
from pymatgen.core import Structure, Element
import pandas as pd
import os

@parallelize
def parse_DDEC6(filepaths):
    """
    Parses DDEC6 output files and returns a Structure object and bond matrix.

    Args:
        filepaths (str or list): The path(s) to the DDEC6 output file(s) to be parsed.

    Returns:
        tuple: A tuple containing the Structure object and bond matrix.
            - The Structure object represents the atomic structure of the system
              and contains information about the lattice, atomic coordinates,
              and atomic numbers.
            - The bond matrix is a DataFrame that provides information about the
              bonding interactions in the system, including bond indices, bond lengths,
              and other properties.

    Raises:
        FileNotFoundError: If the specified file(s) do not exist.

    Example:
        filepaths = ["output1.txt", "output2.txt"]
        structure, bond_matrix = parse_DDEC6(filepaths)
        print(structure)
        print(bond_matrix)

    Note:
        - The function reads the specified DDEC6 output file(s) and extracts relevant
          information to create a Structure object and bond matrix.
        - The function expects the DDEC6 output files to be in a specific format and
          relies on certain trigger lines to identify the relevant sections.
        - The structure lattice is parsed from the lines between the "vectors" and
          "direct_coords" triggers.
        - The atomic fractional coordinates are parsed from the lines between the
          "direct_coords" and "totnumA" triggers.
        - The atomic numbers are parsed from the lines between the "(Missing core
          electrons will be inserted using stored core electron reference densities.)"
          and "Finished the check for missing core electrons." triggers.
        - The atomic numbers are converted to element symbols using the pymatgen
          Element.from_Z() method.
        - The Structure object is created using the parsed lattice, atomic numbers,
          and fractional coordinates.
        - The bond matrix is parsed from the lines between the "The final bond pair
          matrix is" and "The legend for the bond pair matrix follows:" triggers.
        - The bond matrix is returned as a pandas DataFrame with the specified column
          names.

    """
    flist = open(filepaths).readlines()
    structure_lattice = parse_lines(flist, trigger_start="vectors", trigger_end="direct_coords")
    structure_frac_coords = parse_lines(flist, trigger_start="direct_coords", trigger_end="totnumA")
    structure_atomic_no = parse_lines(flist, trigger_start="(Missing core electrons will be inserted using stored core electron reference densities.)", trigger_end=" Finished the check for missing core electrons.")
    # Convert atomic numbers to element symbols
    structure_atomic_no = [Element.from_Z(atomic_number[1]).symbol for atomic_number in structure_atomic_no]
    # Create the Structure object
    structure = Structure(structure_lattice, structure_atomic_no, structure_frac_coords)
    
    data_column_names = ['atom1',\
                'atom2',\
                'repeata',\
                'repeatb',\
                'repeatc',\
                'min-na',\
                'max-na',\
                'min-nb',\
                'max-nb',\
                'min-nc',\
                'max-nc',\
                'contact-exchange',\
                'avg-spin-pol-bonding-term',\
                'overlap-population',\
                'isoaepfcbo',\
                'coord-term-tanh',\
                'pairwise-term',\
                'exp-term-comb-coord-pairwise',\
                'bond-idx-before-self-exch',\
                'final_bond_order']
    
    bond_matrix = parse_lines(flist, trigger_start="The final bond pair matrix is", trigger_end="The legend for the bond pair matrix follows:")
    bond_matrix = pd.DataFrame(bond_matrix, columns=data_column_names)
    
    return structure, bond_matrix

def check_valid_chargemol_output(vasp_ddec_analysis_output_filepath):
    """
    Checks if a VASP DDEC analysis output file indicates successful completion of Chargemol.

    Args:
        vasp_ddec_analysis_output_filepath (str): The path to the VASP DDEC analysis output file.

    Returns:
        bool: True if Chargemol analysis has successfully finished, False otherwise.

    Example:
        output_filepath = "vasp_ddec_analysis_output.txt"
        result = check_valid_chargemol_output(output_filepath)
        if result:
            print("Chargemol analysis finished successfully.")
        else:
            print("Chargemol analysis did not finish.")

    Notes:
        - The function reads the VASP DDEC analysis output file and searches for a specific line
          indicating the completion of the Chargemol analysis.
        - If the specified line is found, the function returns True, indicating successful completion.
        - If the specified line is not found, the function returns False, indicating that the Chargemol
          analysis did not finish or encountered an error.
        - The function assumes that the VASP DDEC analysis output file follows a specific format and
          contains the necessary information.

    """
    convergence = search_line_in_file(vasp_ddec_analysis_output_filepath, "Finished chargemol in")

    return convergence
    
def find_chargemol_dirs(filepath):
    """
    Find directories with Chargemol output files in the specified filepath.

    Args:
        filepath (str): The path to the directory to search for Chargemol output files.

    Returns:
        tuple: A tuple containing two lists:
            - The first list contains paths to directories with Chargemol output files that indicate successful completion.
            - The second list contains paths to directories with Chargemol output files that did not indicate successful completion.

    Example:
        directory_path = "/path/to/directory"
        converged_dirs, non_converged_dirs = find_chargemol_dirs(directory_path)
        print("Converged directories:")
        for converged_dir in converged_dirs:
            print(converged_dir)
        print("Non-converged directories:")
        for non_converged_dir in non_converged_dirs:
            print(non_converged_dir)

    Notes:
        - The function searches for directories in the specified filepath that contain Chargemol output files.
        - The Chargemol output files are expected to have the name "VASP_DDEC_analysis.output" and be located
          in the same directory as an "INCAR" file.
        - The function uses the helper function `find_filepaths_in_dir_with_files` to find directories with "INCAR" files.
        - For each directory found, the function checks if the corresponding "VASP_DDEC_analysis.output" file indicates
          successful completion using the `check_valid_chargemol_output` function.
        - Directories with successful Chargemol completion are added to the converged list, while directories without
          successful completion are added to the non-converged list.
        - The function returns the converged and non-converged lists as a tuple.

    """
    whole_list = find_filepaths_in_dir_with_files(filepath, ["INCAR"])
    whole_list = [os.path.join(os.path.dirname(path), "VASP_DDEC_analysis.output") for path in whole_list]
    converged_list = []
    non_converged_list = []
    for file in whole_list:
        if check_valid_chargemol_output(file):
            converged_list.append(file)
        else:
            non_converged_list.append(file)
    
    return converged_list, non_converged_list
    
def run_scrape(filepath):
    """
    Run the Chargemol scraping process on the specified filepath.

    Args:
        filepath (str): The path to the directory to run the Chargemol scraping process on.

    Returns:
        tuple: A tuple containing two elements:
            - The first element is a list of filepaths to directories with successful Chargemol completion.
            - The second element is the result of parsing the Chargemol output files.

    Example:
        directory_path = "/path/to/directory"
        filepaths, results = run_scrape(directory_path)
        print("Filepaths with successful Chargemol completion:")
        for filepath in filepaths:
            print(filepath)
        print("Parsing results:")
        print(results)

    Notes:
        - The function runs the Chargemol scraping process on the specified filepath.
        - It first calls the `find_chargemol_dirs` function to find directories with Chargemol output files,
          separating them into filepaths with successful completion and non-converged filepaths.
        - The successful completion filepaths are then passed to the `parse_DDEC6` function to parse the Chargemol output files.
        - The parsing results are returned along with the filepaths.

    """
    filepaths, non_converged_filepaths = find_chargemol_dirs(filepath)
    results = parse_DDEC6(filepaths)
    # Pad the results with None if the lengths don't match
    max_length = len(filepaths+non_converged_filepaths)
    results += [None] * (max_length - len(results))
    return filepaths, results

@parallelize
def extract_DDEC6_output(filepath):
    extract_file_from_tarball(filepath, filename="VASP_DDEC")