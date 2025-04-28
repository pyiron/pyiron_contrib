import os
import glob
import time
import tarfile
import re

from pymatgen.core import Structure
from pyiron_atomistics.vasp.outcar import Outcar
from pymatgen.io.vasp import Vasprun, Kpoints, Incar, Potcar
import numpy as np
import pandas as pd

import utils.generic as gen_tools

from utils.parallel import parallelise

def find_vasp_directories(parent_dir,
                          filenames=["vasp.log", "INCAR", "POTCAR", "CONTCAR", "KPOINTS", "OUTCAR", "vasprun.xml"],
                          all_present=False,
                          extract_tarballs=True):
    """
    Finds directories in a parent directory that contain specified files.

    Parameters:
        parent_dir (str): The path of the parent directory to search for directories.
        filenames (list): A list of filenames to search for within the directories.
        all_present (bool, optional): Determines whether all the specified filenames should be present in each directory to be considered a match. Defaults to True.

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
    if extract_tarballs:
        gen_tools.find_and_extract_files_from_tarballs_parallel(parent_dir=parent_dir, 
                                                                extension=".tar.gz",
                                                                filenames=filenames,                                                            
                                                                suffix=None,
                                                                prefix=None)
   
    directories =  gen_tools.find_directories_with_files(parent_dir=parent_dir,
                                          filenames=filenames,
                                          all_present=all_present)

    return directories

def read_OUTCAR(filename="OUTCAR",
                free_energy=True,
                energy_zero=True,
                structures=True):
    """
    Read information from the OUTCAR file and related VASP structure files.

    Parameters:
        filename (str, optional): The path of the OUTCAR file to read. Defaults to "OUTCAR".

    Returns:
        pandas.DataFrame: A DataFrame containing the parsed data from the OUTCAR and related structure files.

    Usage:
        # Read data from the default OUTCAR file "OUTCAR"
        df = read_OUTCAR()

        # Read data from a specific OUTCAR file
        df = read_OUTCAR("/path/to/OUTCAR")

    Note:
        - The function attempts to read information from the specified OUTCAR file using the `Outcar` class from pymatgen.
        - If successful, it collects data such as energies, ionic step structures, forces, stresses, magnetization moments, and SCF step counts.
        - The function searches for related structure files (with extensions .vasp, CONTCAR, and POSCAR) in the same directory as the OUTCAR file.
        - If a related structure file is found, it is parsed using the `Structure` class from pymatgen.
        - The parsed data is stored in a pandas DataFrame with columns for job name, file path, ionic step structures, energies, forces, stresses, magnetization moments, SCF step counts, and convergence.
        - If any part of the parsing encounters an error, the corresponding DataFrame entry will have NaN values.
    """
    outcar = Outcar()
    outcar.from_file(filename = filename)

    structure_name = os.path.basename(os.path.dirname(filename))
    
    try:
        energies = outcar.parse_dict["energies"]
    except:
        energies = np.nan
        
    # create a list of file extensions to search for
    extensions = [".vasp", "CONTCAR", "POSCAR"]
    # create an empty list to store matching files
    structure_files = []
    # walk through the directory and check each file's name
    for root, dirs, files in os.walk(os.path.dirname(filename)):
        for file in files:
            if any(extension in file for extension in extensions):
                structure_files.append(os.path.join(root, file))
    for structure_file in structure_files:
        try:
            structure = Structure.from_file(structure_file)
            break
        except:
            pass
        
    try:
        ionic_step_structures = np.array([Structure(cell, structure.species, outcar.parse_dict["positions"][i], coords_are_cartesian=True).to_json()
                                            for i, cell in enumerate(outcar.parse_dict["cells"])])
    except:
        ionic_step_structures = np.nan
    
    try:
        energies_zero =  outcar.parse_dict["energies_zero"]
    except:
        energies_zero = np.nan
        
    try:
        forces = outcar.parse_dict["forces"]
    except:
        forces = np.nan
        
    try:
        stresses = outcar.parse_dict["stresses"]
    except:
        stresses = np.nan
        
    try:
        magmoms = np.array(outcar.parse_dict["final_magmoms"])
    except:
        magmoms = np.nan
        
    try:
        scf_steps = [len(i) for i in outcar.parse_dict["scf_energies"]]
    except:
        scf_steps = np.nan
        
    df = pd.DataFrame([[structure_name,
                        filename,
                        ionic_step_structures,
                        energies,
                        energies_zero,
                        forces,
                        stresses,
                        magmoms,
                        scf_steps]],
                columns = ["job_name",
                            "filepath",
                            "structures",
                            "energy",
                            "energy_zero",
                            "forces",
                            "stresses",
                            "magmoms",
                            "scf_steps"])
    return df

def parse_VASP_directory(directory,
                      INCAR_filename="INCAR",
                      KPOINTS_filename="KPOINTS",
                      POTCAR_filename="POTCAR",
                      OUTCAR_filename="OUTCAR",
                      vasprunxml_filename="vasprun.xml",
                      vasplog_filename="vasp.log"):
    
    # Find file matching pattern
    structure_files = glob.glob(os.path.join(directory, "starter*.vasp"))
    
    if len(structure_files) > 0:
        init_structure = Structure.from_file(structure_files[0])
    else:
        init_structure = None

    # OUTCAR first
    try:
        df = read_OUTCAR(filename=os.path.join(directory, OUTCAR_filename))
    except:
        df = pd.DataFrame([[np.nan,
                    directory,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan]],
            columns = ["job_name",
                        "filepath",
                        "structures",
                        "energy",
                        "energy_zero",
                        "forces",
                        "stresses",
                        "magmoms",
                        "scf_steps"])
        
    convergence = check_convergence(directory=directory,
                                    filename_vasprun=vasprunxml_filename,
                                    filename_vasplog=vasplog_filename)    
    # INCAR
    try:
        incar = Incar.from_file(os.path.join(directory, INCAR_filename)).as_dict()
    except:
        incar = np.nan
        
    try:
        # KPOINTS
        kpoints = Kpoints.from_file(os.path.join(directory, KPOINTS_filename)).as_dict()
    except:
        try:
            kspacing = incar["KSPACING"]
            kpoints = f"KSPACING: {kspacing}"
        except:
            kpoints = np.nan

    try:
        element_list, element_count, electron_of_potcar = grab_electron_info(directory_path=directory,
                                                                            potcar_filename=POTCAR_filename)
    except:
        element_list = np.nan
        element_count = np.nan
        electron_of_potcar = np.nan

        
    try:
        electron_count = get_total_electron_count(directory_path=directory)
    except:
        electron_count = np.nan
        
    df["element_list"] = [element_list]
    df["element_count"] = [element_count]
    df["potcar_electron_count"] = [electron_of_potcar]
    df["total_electron_count"] = [electron_count]
    df["convergence"] = [convergence]
    
    df["kpoints"] = [kpoints]
    df["incar"] = [incar]

    return df


def check_convergence(directory, filename_vasprun="vasprun.xml", filename_vasplog="vasp.log", backup_vasplog = "error.out"):
    """
    Check the convergence status of a VASP calculation.

    Args:
        directory (str): The directory containing the VASP files.
        filename_vasprun (str, optional): The name of the vasprun.xml file (default: "vasprun.xml").
        filename_vasplog (str, optional): The name of the vasp.log file (default: "vasp.log").

    Returns:
        bool: True if the calculation has converged, False otherwise.

    Raises:
        FileNotFoundError: If neither vasprun.xml nor vasp.log is found.

    Example:
        >>> convergence_status = check_convergence(directory="/path/to/vasp_files")
        >>> if convergence_status:
        ...     print("Calculation has converged.")
        ... else:
        ...     print("Calculation has not converged.")
    """
    try:
        vr = Vasprun(filename=os.path.join(directory, filename_vasprun))
        return vr.converged
    except:
        line_converged = "reached required accuracy - stopping structural energy minimisation"
        try:
            converged = gen_tools.is_line_in_file(filepath=os.path.join(directory, filename_vasplog),
                                        line=line_converged,
                                        exact_match=False)
            return converged
        except:
            try:
                converged = gen_tools.is_line_in_file(filepath=os.path.join(directory, backup_vasplog),
                            line=line_converged,
                            exact_match=False)
                return converged
            except:
                return False

def element_count_ordered(structure):
    site_element_list = [site.species_string for site in structure]
    past_element = site_element_list[0]
    element_list = [past_element]
    element_count = []
    count = 0
    for element in site_element_list:
        if element == past_element:
            count += 1
        else:
            element_count.append(count)
            element_list.append(element)
            count = 1
            past_element = element
    element_count.append(count)
    return element_list, element_count 

def _try_read_structure(directory_path, structure_filenames = ["CONTCAR", ".vasp", "POSCAR"]):    
    structure_files = []
    # walk through the directory and check each file's name
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(filename) for filename in structure_filenames):
                structure_files.append(os.path.join(root, file))
    structure = None
    for structure_file in structure_files:
        try:
            structure = Structure.from_file(structure_file)
            break
        except:
            pass
        if structure == None:
            print(f"no structure found in {directory_path}")
            structure = np.nan
    return structure

def grab_electron_info(directory_path, line_before_elec_str="PAW_PBE", potcar_filename = "POTCAR"):
    
    structure = _try_read_structure(directory_path=directory_path)
    if structure != None:
        element_list, element_count = element_count_ordered(structure)
        
    electron_of_potcar = []
    
    with open(os.path.join(directory_path, potcar_filename), 'r') as file:
        lines = file.readlines()  # Read the lines from the file
        should_append = False  # Flag to determine if the next line should be appended
        for line in lines:
            stripped_line = line.strip()  # Remove leading and trailing whitespace
            if should_append:
                electron_of_potcar.append(float(stripped_line))
                should_append = False  # Reset the flag
            if stripped_line.startswith(line_before_elec_str):
                should_append = True  # Set the flag to append the next line
        
    return element_list, element_count, electron_of_potcar

def get_total_electron_count(directory_path, line_before_elec_str="PAW_PBE", potcar_filename = "POTCAR"):
    ele_list, ele_count, electron_of_potcar = grab_electron_info(directory_path=directory_path, line_before_elec_str=line_before_elec_str, potcar_filename=potcar_filename)
    total_electron_count = np.dot(ele_count, electron_of_potcar)
    return total_electron_count

class DatabaseGenerator():
    
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        
    def build_database(self, extract_directories = True, cleanup=False, keep_filenames_after_cleanup = [], keep_filename_patterns_after_cleanup = [], max_dir_count = None, df_filename = None):
        
        start_time = time.time()
        
        dirs = find_vasp_directories(parent_dir=self.parent_dir, extract_tarballs=extract_directories)
        
        print(f"The total number of vasp directories that we are building the database out of is {len(dirs)}")
        
        if max_dir_count:

            pkl_filenames = []
            
            for i, chunks in enumerate(gen_tools.chunk_list(dirs, max_dir_count)):
                step_time = time.time()
                df = pd.concat(parallelise(parse_VASP_directory, chunks))
                if df_filename:
                    db_filename = f"{i}_{df_filename}.pkl"
                else:
                    db_filename = f"{i}.pkl"
                pkl_filenames.append(os.path.join(self.parent_dir, db_filename))
                df.to_pickle(os.path.join(self.parent_dir, db_filename))
                step_taken_time = np.round(step_time - time.time(),3)
                print(f"Step {i}: {step_taken_time} seconds taken for {len(chunks)} parse steps")
                
            df = pd.concat([pd.read_pickle(partial_df) for partial_df in pkl_filenames])
            df.to_pickle(os.path.join(self.parent_dir, f"vasp_database.pkl"))
            
        else:
            
            df = pd.concat(parallelise(parse_VASP_directory, dirs))
            results = [df]
            if df_filename:
                df.to_pickle(os.path.join(self.parent_dir, f"vasp_database.pkl"))
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # not optional - keep the tarballs/zips..
        keep_filename_patterns_after_cleanup += ".tar.gz"
        keep_filename_patterns_after_cleanup += ".tar.bz2"
        keep_filename_patterns_after_cleanup += ".zip"

        if cleanup:
            gen_tools.cleanup_dir(directory_path=dirs, keep=True, files=[], file_patterns=[])
            parallelise(gen_tools.cleanup_dir, dirs, [True] * len(dirs), keep_filenames_after_cleanup*len(dirs), keep_filename_patterns_after_cleanup*len(dirs))
        
        print("Elapsed time:", np.round(elapsed_time,3), "seconds")

        return df