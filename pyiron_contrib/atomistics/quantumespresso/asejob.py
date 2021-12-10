import os
from shutil import copyfile
from ase import io
from ase.calculators.espresso import Espresso
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob
from pyiron_atomistics.atomistics.structure.atoms import pyiron_to_ase, ase_to_pyiron
from pyiron_base import GenericParameters, Settings, Executable


s = Settings()


class QuantumEspressoInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(QuantumEspressoInput, self).__init__(
            input_file_name=input_file_name, table_name="input", comment_char="#"
        )

    def load_default(self):
        """
        Loading the default settings for the input file.
        """
        input_str = """\
kpoints [3,3,3]
tstress True
tprnfor True
"""
        self.load_string(input_str)


class QuantumEspresso(AtomisticGenericJob):
    def __init__(self, project, job_name):
        super(QuantumEspresso, self).__init__(project, job_name)
        self.__name__ = "QuantumEspresso"
        self.input = QuantumEspressoInput()
        self.pseudopotentials = {
            'Na': 'Na.pbe-spn-rrkjus_psl.1.0.0.UPF',
            'Cl': 'Cl.pbe-nl-rrkjus_psl.1.0.0.UPF'
        }

    def write_input(self):
        calc = Espresso(
            label="pyiron",
            pseudopotentials=self.pseudopotentials,
            tstress=self.input["tstress"],
            tprnfor=self.input["tprnfor"],
            kpts=self.input["kpoints"]
        )
        calc.directory = self.working_directory
        calc.write_input(atoms=pyiron_to_ase(self.structure))

    def collect_output(self):
        output = io.read(os.path.join(self.working_directory, 'pyiron.pwo'))
        with self.project_hdf5.open("output") as hdf5_output:
            for k in output.calc.results.keys():
                hdf5_output[k] = output.calc.results[k]

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the ExampleJob object in the HDF5 File

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(QuantumEspresso, self).to_hdf(hdf=hdf, group_name=group_name)
        self._structure_to_hdf()
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.to_hdf(hdf5_input)
            hdf5_input["potential"] = self.pseudopotentials

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the ExampleJob object in the HDF5 File

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(QuantumEspresso, self).from_hdf(hdf=hdf, group_name=group_name)
        self._structure_from_hdf()
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)
            self.pseudopotentials = hdf5_input["potential"]
