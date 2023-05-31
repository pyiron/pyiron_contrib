from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from pyiron_base import TemplateJob, DataContainer
from fitsnap3lib.scrapers.ase_funcs import ase_scraper

default_input = settings = \
{
### PYIRON-Specific Keywords and Options:
"PYIRON_PERFORMFIT": True,
"PYIRON_SAVE_DESCRIPTORS": True,
"PYIRON_LOAD_DESCRIPTORS": False, # Pass in dictionary with {'a':np.ndarray,'b':np.ndarray,'w':np.ndarry}
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
        self.input.update(default_input)
        self._lst_of_struct = [] # List of ASE atoms containing flagged info (energy, forces, stress, etc.)
        
    @property
    def list_of_structures(self):
        return self._lst_of_struct

    @list_of_structures.setter
    def list_of_structures(self, structure_lst):
        self._lst_of_struct = structure_lst

    def run_static(self):
        comm = MPI.COMM_WORLD
        input_dict = self.input.to_builtin()
        fs = FitSnap(input_dict, comm=comm, arglist=["--overwrite"])

        if isinstance(self.input['PYIRON_LOAD_DESCRIPTORS'],dict):
            fs.pt.create_shared_array(name="a", size1=self.input['PYIRON_LOAD_DESCRIPTORS']['a'].shape[0], 
                                      size2=self.input['PYIRON_LOAD_DESCRIPTORS']['a'].shape[1], tm=0)
            fs.pt.create_shared_array(name="b", size1=self.input['PYIRON_LOAD_DESCRIPTORS']['b'].shape[0], size2=1, tm=0)
            fs.pt.create_shared_array(name="w", size1=self.input['PYIRON_LOAD_DESCRIPTORS']['c'].shape[0], size2=1, tm=0)
            fs.pt.shared_arrays["a"].array = self.input['PYIRON_LOAD_DESCRIPTORS']['a']
            fs.pt.shared_arrays["b"].array = self.input['PYIRON_LOAD_DESCRIPTORS']['b']
            fs.pt.shared_arrays["w"].array = self.input['PYIRON_LOAD_DESCRIPTORS']['w']
            fs.pt.fitsnap_dict['Testing'] = [False] * self.input['PYIRON_LOAD_DESCRIPTORS']['a'].shape[0]
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
        self.status.finished = True

    def to_hdf(self, hdf=None, group_name=None):
        super(FitsnapJob, self).to_hdf(hdf=hdf, group_name=group_name)

    def from_hdf(self, hdf=None, group_name=None):
        super(FitsnapJob, self).from_hdf(hdf=hdf, group_name=group_name)

    def to_hdf(self, hdf=None, group_name=None):
        super(FitsnapJob, self).to_hdf(hdf=hdf, group_name=group_name)

    def from_hdf(self, hdf=None, group_name=None):
        super(FitsnapJob, self).from_hdf(hdf=hdf, group_name=group_name)
