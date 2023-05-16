from mpi4py import MPI
from fitsnap3lib.fitsnap import FitSnap
from pyiron_base import PythonTemplateJob, DataContainer
from pyiron_lanl.fitsnap.common import ase_scraper


default_input = settings = {
    "BISPECTRUM": {
        "numTypes": 1,
        "twojmax": 8,
        "rcutfac": 4.812302818,
        "rfac0": 0.99363,
        "rmin0": 0.0,
        "wj": 1.0,
        "radelem": 0.5,
        "type": "Be",
        "wselfallflag": 0,
        "chemflag": 0,
        "bzeroflag": 0,
        "quadraticflag": 0,
    },
    "CALCULATOR": {
        "calculator": "LAMMPSSNAP",
        "energy": 1,  # Calculate energy descriptors
        "force": 1,  # Calculate force descriptors
        "stress": 0,  # Calculate virial descriptors
    },
    "REFERENCE": {
        "units": "metal",
        "atom_style": "atomic",
        "pair_style": "hybrid/overlay zero 10.0 zbl 4.0 4.8",
        "pair_coeff1": "* * zero",
        "pair_coeff2": "1 1 zbl 74 74",
    },
    "SOLVER": {"solver": "SVD", "compute_testerrs": 1, "detailed_errors": 1},
    "EXTRAS": {
        "dump_descriptors": 0,
        "dump_truth": 0,
        "dump_weights": 0,
        "dump_dataframe": 0,
    },
    "MEMORY": {"override": 0},
}


class FitsnapJob(PythonTemplateJob):
    def __init__(self, project, job_name):
        super(FitsnapJob, self).__init__(project, job_name)
        self.__version__ = "0.1"
        self.__name__ = "FitsnapJob"
        self.input.update(default_input)
        self._lst_of_struct = []
        self._lst_of_energies = []
        self._lst_of_forces = []
        self._coefficients = []

    @property
    def list_of_structures(self):
        return self._lst_of_struct

    @list_of_structures.setter
    def list_of_structures(self, structure_lst):
        self._lst_of_struct = structure_lst

    @property
    def list_of_energies(self):
        return self._lst_of_energies

    @list_of_energies.setter
    def list_of_energies(self, energies):
        self._lst_of_energies = energies

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def list_of_forces(self):
        return self._lst_of_forces

    @list_of_forces.setter
    def list_of_forces(self, forces):
        self._lst_of_forces = forces

    def run_static(self):
        comm = MPI.COMM_WORLD
        input_dict = self.input.to_builtin()
        snap = FitSnap(input_dict, comm=comm, arglist=["--overwrite"])
        ase_scraper(
            snap, self._lst_of_struct, self._lst_of_energies, self._lst_of_forces
        )
        snap.process_configs()
        snap.solver.perform_fit()
        self._coefficients = snap.solver.fit
        self.status.finished = True

    def to_hdf(self, hdf=None, group_name=None):
        super(FitsnapJob, self).to_hdf(hdf=hdf, group_name=group_name)

    def from_hdf(self, hdf=None, group_name=None):
        super(FitsnapJob, self).from_hdf(hdf=hdf, group_name=group_name)
