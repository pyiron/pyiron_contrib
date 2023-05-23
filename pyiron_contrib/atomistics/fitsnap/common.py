import numpy as np
from pybispectrum import calc_bispectrum_names
from fitsnap3lib.scrapers.ase_funcs import get_apre, create_shared_arrays


def ase_scraper(
    s, frames, energies, forces, stresses=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
):
    """
    Custom function to allocate shared arrays used in Calculator and build the internal list of
    dictionaries `data` of configuration info. Customized version of `fitsnap3lib.scrapers.ase_funcs`.

    Args:
        s: fitsnap instance.
        frames: list or array of ASE atoms objects.
        energies: array of energies.
        forces: array of forces for all configurations.
        stresses: array of stresses for all configurations.

    Creates a list of data dictionaries `s.data` suitable for fitsnap descriptor calculation.
    If running in parallel, this list will be distributed over procs, so that each proc will have a
    portion of the list.
    """

    create_shared_arrays(s, frames)
    s.data = [
        collate_data(a, e, f, s)
        for (a, e, f, s) in zip(frames, energies, forces, stresses)
    ]


def collate_data(atoms, energy, forces, stresses):
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args:
        atoms: ASE atoms object for a single configuration of atoms.
        energy: energy of a configuration.
        forces: numpy array of forces for a configuration.
        stresses: numpy array of stresses for a configuration.

    Returns a fitsnap data dictionary for a single configuration.
    """

    # make a data dictionary for this config

    apre = get_apre(cell=atoms.cell)
    R = np.dot(np.linalg.inv(atoms.cell), apre)

    positions = np.matmul(atoms.get_positions(), R)
    cell = apre.T

    data = {}
    data["PositionsStyle"] = "angstrom"
    data["AtomTypeStyle"] = "chemicalsymbol"
    data["StressStyle"] = "bar"
    data["LatticeStyle"] = "angstrom"
    data["EnergyStyle"] = "electronvolt"
    data["ForcesStyle"] = "electronvoltperangstrom"
    data["Group"] = "Displaced_BCC"
    data["File"] = None
    data["Stress"] = stresses
    data["Positions"] = positions
    data["Energy"] = energy
    data["AtomTypes"] = atoms.get_chemical_symbols()
    data["NumAtoms"] = len(atoms)
    data["Forces"] = forces
    data["QMLattice"] = cell
    data["test_bool"] = 0
    data["Lattice"] = cell
    data["Rotation"] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    data["Translation"] = np.zeros((len(atoms), 3))
    data["eweight"] = 1.0
    data["fweight"] = 1.0 / 150.0
    data["vweight"] = 0.0

    return data


def subsample_twojmax(total_bispect, twojmax_lst):
    bi_spect_names_str_lst = [str(lst) for lst in total_bispect]
    twojmax_master_str_lst = [
        [str(lst) for lst in calc_bispectrum_names(twojmax=tjm)] for tjm in twojmax_lst
    ]
    ind_lst = [
        [desc in desc_lst for desc in bi_spect_names_str_lst]
        for desc_lst in twojmax_master_str_lst
    ]
    return ind_lst
