import os
from pyiron_base import DataContainer, state
from pyiron_atomistics.atomistics.structure.atoms import pyiron_to_ase, ase_to_pyiron
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob
from pycp2k import CP2K


pyiron_dict = {
    "global": {"run_type": "ENERGY_FORCE"},
    "force_eval": {"method": "Quickstep", "print_forces_section": "ON"},
    "dft": {
        "qs": {"eps": 1.0e-10},
        "mgrid": {"ngrids": 4, "cutoff": 300, "rel_cutoff": 60},
        "xc": {"functional": "PADE"},
    },
    "scf": {
        "scf_guess": "ATOMIC",
        "eps": 1.0e-7,
        "max": 300,
        "diagonalization": {"algorthim": "STANDARD"},
        "mixing": {
            "method": "BROYDEN_MIXING",
            "alpha": 0.4,
            "n_broyden": 8,
        },
    },
    "kind": {"basis_set": "DZVP-GTH-PADE", "potential": "GTH-PADE-q4"},
}


class Cp2kJob(AtomisticGenericJob):
    def __init__(self, project, job_name):
        super(Cp2kJob, self).__init__(project, job_name)
        self.__name__ = "cp2k"
        self.input = DataContainer(pyiron_dict, table_name="control_dict")

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the ExampleJob object in the HDF5 File

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(Cp2kJob, self).to_hdf(hdf=hdf, group_name=group_name)
        self._structure_to_hdf()
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the ExampleJob object in the HDF5 File

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(Cp2kJob, self).from_hdf(hdf=hdf, group_name=group_name)
        self._structure_from_hdf()
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)

    def write_input(self):
        calc = CP2K()
        calc.working_directory = self.working_directory
        calc.project_name = "pyiron"
        CP2K_INPUT = calc.CP2K_INPUT
        GLOBAL = CP2K_INPUT.GLOBAL
        FORCE_EVAL = (
            CP2K_INPUT.FORCE_EVAL_add()
        )  # Repeatable items have to be first created
        SUBSYS = FORCE_EVAL.SUBSYS
        DFT = FORCE_EVAL.DFT
        SCF = DFT.SCF
        GLOBAL.Run_type = self.input["global"]["run_type"]
        FORCE_EVAL.Method = self.input["force_eval"]["method"]
        FORCE_EVAL.PRINT.FORCES.Section_parameters = self.input["force_eval"][
            "print_forces_section"
        ]
        DFT.Basis_set_file_name = os.path.join(
            state.settings.resource_paths[0], "cp2k", "potentials", "BASIS_SET"
        )
        DFT.Potential_file_name = os.path.join(
            state.settings.resource_paths[0], "cp2k", "potentials", "GTH_POTENTIALS"
        )
        DFT.QS.Eps_default = self.input["dft"]["qs"]["eps"]
        DFT.MGRID.Ngrids = self.input["dft"]["mgrid"]["ngrids"]
        DFT.MGRID.Cutoff = self.input["dft"]["mgrid"]["cutoff"]
        DFT.MGRID.Rel_cutoff = self.input["dft"]["mgrid"]["rel_cutoff"]
        DFT.XC.XC_FUNCTIONAL.Section_parameters = self.input["dft"]["xc"]["functional"]
        SCF.Scf_guess = self.input["scf"]["scf_guess"]
        SCF.Eps_scf = self.input["scf"]["eps"]
        SCF.Max_scf = self.input["scf"]["max"]
        SCF.DIAGONALIZATION.Section_parameters = "ON"
        SCF.DIAGONALIZATION.Algorithm = self.input["scf"]["diagonalization"][
            "algorthim"
        ]
        SCF.MIXING.Section_parameters = "T"
        SCF.MIXING.Method = self.input["scf"]["mixing"]["method"]
        SCF.MIXING.Alpha = self.input["scf"]["mixing"]["alpha"]
        SCF.MIXING.Nbroyden = self.input["scf"]["mixing"]["n_broyden"]
        for el in set(self.structure.get_chemical_symbols()):
            KIND = SUBSYS.KIND_add(el)
        KIND.Basis_set = self.input["kind"]["basis_set"]
        KIND.Potential = self.input["kind"]["potential"]
        calc.create_cell(SUBSYS, pyiron_to_ase(self.structure))
        calc.create_coord(SUBSYS, pyiron_to_ase(self.structure))
        calc.write_input_file()

    def collect_output(self):
        pass
