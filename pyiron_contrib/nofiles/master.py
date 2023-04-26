import pandas
from pympipool import Pool
from pyiron_base import DataContainer, GenericJob, state
from pyiron_atomistics.project import Project
from pyiron_atomistics.atomistics.structure.atoms import (
    Atoms,
    pyiron_to_ase,
    ase_to_pyiron,
)
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob


class GenericJobNoFiles(GenericJob):
    def __init__(self, project, job_name):
        if not state.database.database_is_disabled:
            raise RuntimeError(
                "To run a `Without` job, the database must first be disabled. Please "
                "`from pyiron_base import state; "
                "state.update({'disable_database': True})`, and try again."
            )
        super(GenericJobNoFiles, self).__init__(project, job_name)

        # internal variables
        self._python_only_job = True
        self._interactive_disable_log_file = True

    def refresh_job_status(self):
        if not self._interactive_disable_log_file:
            super(GenericJobNoFiles, self).refresh_job_status()

    def to_hdf(self, hdf=None, group_name=None):
        """

        Args:
            hdf:
            group_name:

        Returns:

        """
        if not self._interactive_disable_log_file:
            super(GenericJobNoFiles, self).to_hdf(hdf=hdf, group_name=group_name)


class AtomisticGenericJobNoFiles(AtomisticGenericJob):
    def __init__(self, project, job_name):
        super(AtomisticGenericJobNoFiles, self).__init__(project, job_name)

        # internal variables
        self._python_only_job = True
        self._interactive_disable_log_file = False

    def refresh_job_status(self):
        if not self._interactive_disable_log_file:
            super(AtomisticGenericJobNoFiles, self).refresh_job_status()

    def to_hdf(self, hdf=None, group_name=None):
        """

        Args:
            hdf:
            group_name:

        Returns:

        """
        if not self._interactive_disable_log_file:
            super(AtomisticGenericJobNoFiles, self).to_hdf(
                hdf=hdf, group_name=group_name
            )
            self._structure_to_hdf()


class AtomisticStructureMasterNoFiles(AtomisticGenericJobNoFiles):
    def __init__(self, project, job_name):
        super(AtomisticStructureMasterNoFiles, self).__init__(project, job_name)
        self._lst_of_struct = []

    @property
    def list_of_structures(self):
        return self._lst_of_struct

    def from_hdf(self, hdf=None, group_name=None):
        super(AtomisticStructureMasterNoFiles, self).from_hdf(
            hdf=hdf, group_name=group_name
        )
        self._structure_from_hdf()
        with self.project_hdf5.open("output/structures") as hdf5_output:
            structure_names = hdf5_output.list_groups()
        for group in structure_names:
            with self.project_hdf5.open("output/structures/" + group) as hdf5_output:
                self._lst_of_struct.append(Atoms().from_hdf(hdf5_output))


class GenericStructureMasterNoFiles(GenericJobNoFiles):
    def __init__(self, project, job_name):
        super(GenericStructureMasterNoFiles, self).__init__(project, job_name)
        self._lst_of_struct = []

    @property
    def list_of_structures(self):
        return self._lst_of_struct

    def from_hdf(self, hdf=None, group_name=None):
        super(GenericStructureMasterNoFiles, self).from_hdf(
            hdf=hdf, group_name=group_name
        )
        self._structure_from_hdf()
        with self.project_hdf5.open("output/structures") as hdf5_output:
            structure_names = hdf5_output.list_groups()
        for group in structure_names:
            with self.project_hdf5.open("output/structures/" + group) as hdf5_output:
                self._lst_of_struct.append(Atoms().from_hdf(hdf5_output))


class SQSMasterMPI(AtomisticStructureMasterNoFiles):
    def __init__(self, project, job_name):
        super(SQSMasterMPI, self).__init__(project, job_name)

        # input
        self.input = DataContainer(table_name="custom_dict")
        self.input.mole_fraction_dict_lst = []

    def to_hdf(self, hdf=None, group_name=None):
        """

        Args:
            hdf:
            group_name:

        Returns:

        """
        if not self._interactive_disable_log_file:
            super(SQSMasterMPI, self).to_hdf(hdf=hdf, group_name=group_name)
            with self.project_hdf5.open("input") as h5in:
                self.input.to_hdf(h5in)

    def run_static(self):
        self.project_hdf5.create_working_directory()
        input_para_lst = [
            [
                i,
                mole_fraction_dict,
                pyiron_to_ase(self.structure),
                self.working_directory,
            ]
            for i, mole_fraction_dict in enumerate(self.input.mole_fraction_dict_lst)
        ]
        with Pool(cores=self.server.cores) as p:
            list_of_structures = p.map(
                function=generate_sqs_structures, lst=input_para_lst
            )
            self._lst_of_struct = [ase_to_pyiron(s) for s in list_of_structures]

        if not self._interactive_disable_log_file:
            for i, structure in enumerate(self._lst_of_struct):
                with self.project_hdf5.open(
                    "output/structures/structure_" + str(i)
                ) as h5:
                    structure.to_hdf(h5)
            self.status.finished = True
            self.project.db.item_update(self._runtime(), self.job_id)


class LAMMPSMinimizeMPI(GenericStructureMasterNoFiles):
    def __init__(self, project, job_name):
        super(LAMMPSMinimizeMPI, self).__init__(project, job_name)

        # input
        self.input = DataContainer(table_name="custom_dict")
        self.input.potential = ""
        self._structure_lst = []

    @property
    def structure_lst(self):
        return self._structure_lst

    @structure_lst.setter
    def structure_lst(self, structure_lst):
        self._structure_lst = structure_lst

    def to_hdf(self, hdf=None, group_name=None):
        """

        Args:
            hdf:
            group_name:

        Returns:

        """
        if not self._interactive_disable_log_file:
            super(LAMMPSMinimizeMPI, self).to_hdf(hdf=hdf, group_name=group_name)
            with self.project_hdf5.open("input") as h5in:
                self.input.to_hdf(h5in)
            with self.project_hdf5.open("input/structures") as hdf5_input:
                for ind, struct in enumerate(self.structure_lst):
                    struct.to_hdf(hdf=hdf5_input, group_name="s_" + str(ind))

    def run_static(self):
        self.project_hdf5.create_working_directory()
        input_para_lst = [
            [i, pyiron_to_ase(structure), self.input.potential, self.working_directory]
            for i, structure in enumerate(self._structure_lst)
        ]
        with Pool(cores=self.server.cores) as p:
            list_of_structures = p.map(
                function=minimize_structure_with_lammps, lst=input_para_lst
            )
            self._lst_of_struct = [ase_to_pyiron(s) for s in list_of_structures]

        if not self._interactive_disable_log_file:
            for i, structure in enumerate(self._lst_of_struct):
                with self.project_hdf5.open(
                    "output/structures/structure_" + str(i)
                ) as h5:
                    structure.to_hdf(h5)
            self.status.finished = True
            self.project.db.item_update(self._runtime(), self.job_id)


class LAMMPSElasticMPI(GenericJobNoFiles):
    def __init__(self, project, job_name):
        super(LAMMPSElasticMPI, self).__init__(project, job_name)

        # input
        self.input = DataContainer(table_name="custom_dict")
        self.input.potential = ""
        self.input.element_lst = []
        self._structure_lst = []
        self._results_df = None

    @property
    def structure_lst(self):
        return self._structure_lst

    @structure_lst.setter
    def structure_lst(self, structure_lst):
        self._structure_lst = structure_lst

    @property
    def results(self):
        return self._results_df

    def to_hdf(self, hdf=None, group_name=None):
        """

        Args:
            hdf:
            group_name:

        Returns:

        """
        if not self._interactive_disable_log_file:
            super(LAMMPSElasticMPI, self).to_hdf(hdf=hdf, group_name=group_name)
            with self.project_hdf5.open("input") as h5in:
                self.input.to_hdf(h5in)
            with self.project_hdf5.open("input/structures") as hdf5_input:
                for ind, struct in enumerate(self.structure_lst):
                    struct.to_hdf(hdf=hdf5_input, group_name="s_" + str(ind))

    def run_static(self):
        self.project_hdf5.create_working_directory()
        input_para_lst = [
            [
                i,
                pyiron_to_ase(structure),
                self.input.element_lst,
                self.input.potential,
                self.working_directory,
            ]
            for i, structure in enumerate(self._structure_lst)
        ]
        with Pool(cores=self.server.cores) as p:
            results = p.map(function=get_elastic_constants, lst=input_para_lst)

        self._results_df = convert_elastic_constants_to_dataframe(results)
        if not self._interactive_disable_log_file:
            self._results_df.to_hdf(
                self.project_hdf5._file_name, self.job_name + "/output/df"
            )

    def from_hdf(self, hdf=None, group_name=None):
        super(LAMMPSElasticMPI, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("output") as hdf5_output:
            if "df" in hdf5_output.list_groups():
                self._results_df = pandas.read_hdf(
                    self.project_hdf5._file_name, self.job_name + "/output/df"
                )


class LAMMPSMinimizeElasticMPI(AtomisticStructureMasterNoFiles):
    def __init__(self, project, job_name):
        super(LAMMPSMinimizeElasticMPI, self).__init__(project, job_name)

        # input
        self.input = DataContainer(table_name="custom_dict")
        self.input.mole_fraction_dict_lst = []
        self.input.potential = ""
        self.input.element_lst = []

    @property
    def results(self):
        return self._results_df

    def to_hdf(self, hdf=None, group_name=None):
        """

        Args:
            hdf:
            group_name:

        Returns:

        """
        if not self._interactive_disable_log_file:
            super(LAMMPSMinimizeElasticMPI, self).to_hdf(hdf=hdf, group_name=group_name)
            with self.project_hdf5.open("input") as h5in:
                self.input.to_hdf(h5in)

    def run_static(self):
        self.project_hdf5.create_working_directory()
        input_para_lst = [
            [
                i,
                mole_fraction_dict,
                pyiron_to_ase(self.structure),
                self.input.element_lst,
                self.input.potential,
                self.working_directory,
            ]
            for i, mole_fraction_dict in enumerate(self.input.mole_fraction_dict_lst)
        ]
        with Pool(cores=self.server.cores) as p:
            results = p.map(function=combined_function, lst=input_para_lst)

        self._results_df = convert_elastic_constants_to_dataframe(results)
        if not self._interactive_disable_log_file:
            self._results_df.to_hdf(
                self.project_hdf5._file_name, self.job_name + "/output/df"
            )

    def from_hdf(self, hdf=None, group_name=None):
        super(LAMMPSMinimizeElasticMPI, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("output") as hdf5_output:
            if "df" in hdf5_output.list_groups():
                self._results_df = pandas.read_hdf(
                    self.project_hdf5._file_name, self.job_name + "/output/df"
                )


def generate_sqs_structures(input_parameter):
    i, mole_fraction_dict, structure_template, working_directory = input_parameter

    # import
    import numpy as np
    from pyiron_atomistics import Project, ase_to_pyiron, pyiron_to_ase
    from pyiron_contrib.nofiles.sqs import SQSJobWithoutOutput

    # calculation
    if len(mole_fraction_dict) > 1:
        project = Project(working_directory)
        job = project.create_job(SQSJobWithoutOutput, "sqs_" + str(i))
        job._interactive_disable_log_file = True
        job.structure = ase_to_pyiron(structure_template)
        job.input["mole_fractions"] = mole_fraction_dict
        job.input["iterations"] = 1e6
        job.server.cores = 1
        job.run()
        structure_next = pyiron_to_ase(job._lst_of_struct[-1])
    else:
        # use ASE syntax
        structure_next = structure_template.copy()
        structure_next.symbols[:] = list(mole_fraction_dict.keys())[-1]

    # return value
    return structure_next


def minimize_structure_with_lammps(input_parameter):
    i, structure_next, potential, working_directory = input_parameter

    # import
    from pyiron_atomistics import Project, ase_to_pyiron, pyiron_to_ase
    from pyiron_contrib.nofiles.lammps import LammpsInteractiveWithoutOutput
    from mpi4py import MPI

    # calculation
    project = Project(working_directory)
    lmp_mini1 = project.create_job(
        LammpsInteractiveWithoutOutput, "lmp_mini_" + str(i), delete_existing_job=True
    )
    lmp_mini1.structure = ase_to_pyiron(structure_next)
    lmp_mini1.potential = potential
    lmp_mini1.calc_minimize(pressure=0.0)
    lmp_mini1.server.run_mode.interactive = True
    lmp_mini1.interactive_mpi_communicator = MPI.COMM_SELF
    lmp_mini1._interactive_disable_log_file = True  # disable lammps.log
    lmp_mini1.run()
    lmp_mini1.interactive_close()

    # return value
    return pyiron_to_ase(lmp_mini1.get_structure())


def get_elastic_constants(input_para):
    i, structure, element_lst, potential, working_directory = input_para

    # import
    from pyiron_atomistics import Project, ase_to_pyiron
    from pyiron_contrib.nofiles.lammps import LammpsInteractiveWithoutOutput
    from pyiron_contrib.nofiles.elastic import ElasticMatrixJobWithoutFiles
    from mpi4py import MPI

    # Elastic constants
    project = Project(working_directory)
    lmp_elastic = project.create_job(
        LammpsInteractiveWithoutOutput,
        "lmp_elastic_" + str(i),
        delete_existing_job=True,
    )
    lmp_elastic.structure = ase_to_pyiron(structure)
    lmp_elastic.potential = potential
    lmp_elastic.interactive_enforce_structure_reset = True
    lmp_elastic.interactive_mpi_communicator = MPI.COMM_SELF
    lmp_elastic.server.run_mode.interactive = True
    lmp_elastic._interactive_disable_log_file = True  # disable lammps.log
    elastic = lmp_elastic.create_job(
        ElasticMatrixJobWithoutFiles, "elastic_" + str(i), delete_existing_job=True
    )
    elastic._interactive_disable_log_file = True  # disable lammps.log
    elastic.run()

    # return value
    elastic_constants_lst = [
        elastic._data["C"][0][0],
        elastic._data["C"][0][1],
        elastic._data["C"][0][2],
        elastic._data["C"][0][3],
        elastic._data["C"][0][4],
        elastic._data["C"][0][5],
        elastic._data["C"][1][1],
        elastic._data["C"][1][2],
        elastic._data["C"][1][3],
        elastic._data["C"][1][4],
        elastic._data["C"][1][5],
        elastic._data["C"][2][2],
        elastic._data["C"][2][3],
        elastic._data["C"][2][4],
        elastic._data["C"][2][5],
        elastic._data["C"][3][3],
        elastic._data["C"][3][4],
        elastic._data["C"][3][5],
        elastic._data["C"][4][4],
        elastic._data["C"][4][5],
        elastic._data["C"][5][5],
    ]

    conc_lst = []
    for el in element_lst:
        if el in elastic.ref_job.structure.get_species_symbols():
            conc_lst.append(
                sum(
                    elastic.ref_job.structure.indices
                    == elastic.ref_job.structure.get_species_symbols()
                    .tolist()
                    .index(el)
                )
                / len(elastic.ref_job.structure.indices)
            )
        else:
            conc_lst.append(0.0)

    return elastic_constants_lst + conc_lst


def combined_function(input_parameter):
    (
        i,
        mole_fraction_dict,
        structure_template,
        element_lst,
        potential,
        working_directory,
    ) = input_parameter

    # import
    from pyiron_contrib.nofiles.master import (
        generate_sqs_structures,
        minimize_structure_with_lammps,
        get_elastic_constants,
    )

    # calculation
    structure_next = generate_sqs_structures(
        input_parameter=[i, mole_fraction_dict, structure_template, working_directory]
    )
    structure = minimize_structure_with_lammps(
        input_parameter=[i, structure_next, potential, working_directory]
    )
    results = get_elastic_constants(
        input_para=[i, structure, element_lst, potential, working_directory]
    )

    # return value
    return results


def convert_elastic_constants_to_dataframe(results):
    (
        c11_lst,
        c12_lst,
        c13_lst,
        c14_lst,
        c15_lst,
        c16_lst,
        c22_lst,
        c23_lst,
        c24_lst,
        c25_lst,
        c26_lst,
        c33_lst,
        c34_lst,
        c35_lst,
        c36_lst,
        c44_lst,
        c45_lst,
        c46_lst,
        c55_lst,
        c56_lst,
        c66_lst,
        conc_Fe_lst,
        conc_Ni_lst,
        conc_Cr_lst,
        conc_Co_lst,
        conc_Cu_lst,
    ) = zip(*results)

    return pandas.DataFrame(
        {
            "conc_Fe": conc_Fe_lst,
            "conc_Ni": conc_Ni_lst,
            "conc_Cr": conc_Cr_lst,
            "conc_Co": conc_Co_lst,
            "conc_Cu": conc_Cu_lst,
            "C11": c11_lst,
            "C12": c12_lst,
            "C13": c13_lst,
            "C14": c14_lst,
            "C15": c15_lst,
            "C16": c16_lst,
            "C22": c22_lst,
            "C23": c23_lst,
            "C24": c24_lst,
            "C25": c25_lst,
            "C26": c26_lst,
            "C33": c33_lst,
            "C34": c34_lst,
            "C35": c35_lst,
            "C36": c36_lst,
            "C44": c44_lst,
            "C45": c45_lst,
            "C46": c46_lst,
            "C55": c55_lst,
            "C56": c56_lst,
            "C66": c66_lst,
        }
    )
