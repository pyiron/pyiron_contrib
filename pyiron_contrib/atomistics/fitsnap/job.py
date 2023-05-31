import os
from mpi4py import MPI
import pandas as pd
import numpy as np
from fitsnap3lib.fitsnap import FitSnap
from pyiron_base import TemplateJob, DataContainer
from fitsnap3lib.scrapers.ase_funcs import ase_scraper

default_input = settings = \
{
### PYIRON-Specific Keywords and Options:
"PYIRON_PERFORMFIT": True,
"PYIRON_SAVE_DESCRIPTORS": True,
"PYIRON_LOAD_DESCRIPTORS": False, # Pass in dictionary with {'a':np.ndarray,'b':np.ndarray,'w':np.ndarry}
### Fitsnap Parameters
"BISPECTRUM":
    {
    "numTypes": 1,
    "twojmax": 6,
    "rcutfac": 4.67637,
    "rfac0": 0.99363,
    "rmin0": 0.0,
    "wj": 1.0,
    "radelem": 0.5,
    "type": "Ta",
    "wselfallflag": 0,
    "chemflag": 0,
    "bzeroflag": 0,
    "quadraticflag": 0,
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSSNAP",
    "energy": 1,
    "force": 1,
    "stress": 1
    },
"ESHIFT":
    {
    "Ta": 0.0
    },
"SOLVER":
    {
    "solver": "SVD",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"OUTFILE":
    {
    "metrics": "Ta_metrics.md",
    "potential": "Ta_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "atom_style": "atomic",
    "pair_style": "hybrid/overlay zero 10.0 zbl 4.0 4.8",
    "pair_coeff1": "* * zero",
    "pair_coeff2": "* * zbl 73 73"
    },
"EXTRAS":
    {
    "dump_descriptors": 0,
    "dump_truth": 0,
    "dump_weights": 0,
    "dump_dataframe": 0
    },
"MEMORY":
    {
    "override": 0
    },
}

class FitsnapJob(TemplateJob):
    def __init__(self, project, job_name):
        super(FitsnapJob, self).__init__(project, job_name)
        self.__version__ = "0.1"
        self.__name__ = "FitsnapJob"
        self._compress_by_default = False # Make sure to not compress.
        self.input.update(default_input)
        self._lst_of_struct = [] # List of ASE atoms containing flagged info (energy, forces, stress, etc.)
        self._potential = None
        
    @property
    def list_of_structures(self):
        return self._lst_of_struct

    @list_of_structures.setter
    def list_of_structures(self, structure_lst):
        self._lst_of_struct = structure_lst

    @property # Set potential property to potential.
    def potential(self):
        return self.get_potential()

    def save_pyiron_lammps_potential_dict(self):
        """save_pyiron_lammps_potential_dict 
        Load generated fitsnap potential into lammps files.

        Necessary right now - but can be 
        """
        snapmod = self.input['OUTFILE']['potential'] + '.mod'
        snapparam = self.input['OUTFILE']['potential'] + '.snapparam'
        snapcoeff = self.input['OUTFILE']['potential'] + '.snapcoeff'
        
        outsnapparam = os.path.join(self.working_directory, snapparam)
        outsnapcoeff = os.path.join(self.working_directory, snapcoeff)

        self._potential = {
        'Name': [ 'Snap_Potential' ],
        'Filename': [ [outsnapparam, outsnapcoeff, snapmod] ],
        'Model': [ 'Custom' ],
        'Species': [ self.input['BISPECTRUM']['type'].split() ],
        'Config': [ ['include {}\n'.format(snapmod)] ]
        }
        with self.project_hdf5.open("output") as hdf_output:
            hdf_output["lammps_potential"] = self._potential

    def run_static(self):
        comm = MPI.COMM_WORLD
        input_dict = self.input.to_builtin()
        fs = FitSnap(input_dict, comm=comm, arglist=["--overwrite"])

        if isinstance(self.input['PYIRON_LOAD_DESCRIPTORS'],dict):
            if any([self.input['PYIRON_LOAD_DESCRIPTORS'].get('a',None) is None,
                    self.input['PYIRON_LOAD_DESCRIPTORS'].get('b',None) is None,
                    self.input['PYIRON_LOAD_DESCRIPTORS'].get('w',None) is None]):
                raise ValueError('At least "a", "b", and "w" matrices must be passed to "PYIRON_LOAD_DESCRIPTORS" dictionary.')
            else:
                a_array = self.input['PYIRON_LOAD_DESCRIPTORS']['a']
                b_array = self.input['PYIRON_LOAD_DESCRIPTORS']['b']
                w_array = self.input['PYIRON_LOAD_DESCRIPTORS']['w'] 
                fs.pt.create_shared_array(name="a", size1=a_array.shape[0], 
                                        size2=a_array.shape[1], tm=0)
                fs.pt.create_shared_array(name="b", size1=b_array.shape[0], size2=1, tm=0)
                fs.pt.create_shared_array(name="w", size1=w_array.shape[0], size2=1, tm=0)
                fs.pt.shared_arrays["a"].array = a_array
                fs.pt.shared_arrays["b"].array = b_array
                fs.pt.shared_arrays["w"].array = w_array
                fs.pt.fitsnap_dict['Testing'] = [False] * a_array.shape[0]
        else:
            data = ase_scraper(
                self._lst_of_struct
            )
            fs.process_configs(data=data)
        if self.input["PYIRON_SAVE_DESCRIPTORS"]:
            with self.project_hdf5.open("output") as hdf_output:
                hdf_output["a"] = fs.pt.shared_arrays['a'].array
                hdf_output["b"] = fs.pt.shared_arrays['b'].array
                hdf_output["w"] = fs.pt.shared_arrays['w'].array
        if self.input['PYIRON_PERFORM_FIT']: 
            fs.perform_fit()
            fs.output.write_lammps(fs.solver.fit)
            self.save_pyiron_lammps_potential_dict()
        self.status.finished = True

    def get_potential(self):
        if self._potential is not None:
            return pd.DataFrame(self._potential)

    def to_hdf(self, hdf=None, group_name=None):
        super(FitsnapJob, self).to_hdf(hdf=hdf, group_name=group_name)

    def from_hdf(self, hdf=None, group_name=None):
        super(FitsnapJob, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("output") as hdf_output:
            if "lammps_potential" in hdf_output.list_nodes():
                self._potential = hdf_output["lammps_potential"]
