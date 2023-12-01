import os
from io import StringIO
import tempfile
import shutil
import contextlib
import pandas as pd
import numpy as np

from pyiron_base import PythonTemplateJob
from pyiron import ase_to_pyiron

from ase.optimize import LBFGSLineSearch
import ase.io

from xtb.ase.calculator import XTB

@contextlib.contextmanager
def make_temp_directory(prefix=None):
    """make_temp_directory function to make a temporary directory and change there.
    Needed for xtb jobs that create input read/written from files.

    Parameters
    ----------
    prefix : str, optional
        path prefix to temporary folder, by default None

    Yields
    ------
    temp_dir : str
        name of the temporary directory
    """
    mycwd = os.getcwd()
    try:
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(mycwd)
        shutil.rmtree(temp_dir)

def convert_ase_xyz(ase_atoms):
    """convert_ase_xyz

    Parameters
    ----------
    ase_atoms : ase.Atoms
        ase atoms to write to xyz string

    Returns
    -------
    outstring : str
        xyz file string
    """
    outstring = '{}\n\n'.format(len(ase_atoms))
    for atom in ase_atoms:
        outstring += '{} {} {} {}\n'.format(atom.symbol,
                                            atom.position[0],
                                            atom.position[1],
                                            atom.position[2])
    outstring = outstring.strip('\n')
    return outstring

def convert_xyz_ase(structure_str):
    """convert_xyz_ase

    Parameters
    ----------
    structure_str : str
        xyz file string

    Returns
    -------
    ase_atoms : ase.Atoms
        ase atoms to write to xyz string
    """
    return ase.io.read(StringIO(structure_str), format="xyz")

inputDict = {
    'structure':'2\n\nH 0.00 0.00 0.00\nH 2.00 2.00 2.00',
    'uhf':0,
    'charge':0,
    'solvent':'none',
    'method':'GFN2-xTB',
    'relax':False,
    'fmax':0.1,
    'steps':100,
    'xtb_electronic_temperature':300,
    'xtb_max_iterations':250,
    'xtb_accuracy':1,
}

class XtbJob(PythonTemplateJob):
    def __init__(self, project, job_name):
        super(XtbJob, self).__init__(project, job_name)
        self.input.update(inputDict)
        self.resultsdf = pd.DataFrame({})

    def run_static(self):
        # These are set to ensure xTB is run serially. Helpful when trying to run several jobs at the same time.
        os.environ["MKL_NUM_THREADS"]="1"
        os.environ["NUMEXPR_NUM_THREADS"]="1"
        os.environ["OMP_NUM_THREADS"]="1"
        ase_atoms = convert_xyz_ase(self.input['structure'])
        calc = XTB(method=self.input['method'], solvent=self.input['solvent'],
                           max_iterations=self.input['xtb_max_iterations'],
                           electronic_temperature=self.input['xtb_electronic_temperature'],
                           accuracy=self.input['xtb_accuracy'])
        uhf_vect = np.zeros(len(ase_atoms))
        uhf_vect[0] = self.input["uhf"]
        charge_vect = np.zeros(len(ase_atoms))
        charge_vect[0] = self.input['charge']
        ase_atoms.set_initial_charges(charge_vect)
        ase_atoms.set_initial_magnetic_moments(uhf_vect)
        ase_atoms.calc = calc
        if inputDict['relax']:
            with make_temp_directory(
                        prefix='/tmp/') as _:
                dyn = LBFGSLineSearch(ase_atoms,
                        trajectory='temp.traj',
                        logfile='tmp.log')
                try:
                    dyn.run(fmax=self.input['fmax'],steps=self.input['steps'])
                except Exception as e:
                    pass
        else:
            with make_temp_directory(
                        prefix='/tmp/') as _:
                try:
                    ase_atoms.get_total_energy()
                except:
                    pass
        if len(ase_atoms.calc.results) > 0:
            results = ase_atoms.calc.results
            results.update({"xtb_completed":True})
            results.update({"structure":convert_ase_xyz(ase_atoms)})
            results = {key:([val] if hasattr(val,'shape') else val) for key,val in results.items()}
            resultsdf = pd.DataFrame(results)
            self.resultsdf = resultsdf
            structure_pyiron = ase_to_pyiron(ase_atoms)
            with self.project_hdf5.open("output/generic") as h5out:
                structure_pyiron.to_hdf(h5out)
            self.resultsdf.to_hdf(self.project_hdf5.file_name,key=self.job_name+'/output/xtb')

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf, group_name)
        if (self['output'] is not None) and ('xtb' in self['output'].list_groups()) and \
            (self.resultsdf.shape[0] == 0):
            self.resultsdf = pd.read_hdf(self.project_hdf5.file_name,
                    key=self.job_name+'/output/xtb')