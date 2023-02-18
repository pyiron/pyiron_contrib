import numpy as np
from pyiron_contrib.nofiles.lammps import LammpsInteractiveWithoutOutput
from pyiron_contrib.nofiles.elastic import ElasticMatrixJobWithoutFiles
from pyiron_contrib.nofiles.sqs import SQSJobWithoutOutput
from pyiron_atomistics import Project, ase_to_pyiron, pyiron_to_ase


def generate_sqs_structures(input_parameter):
    i, mole_fraction_dict, structure_template, working_directory = input_parameter

    # calculation
    if len(mole_fraction_dict) > 1:
        project = Project(working_directory)
        job = project.create_job(SQSJobWithoutOutput, "sqs_" + str(i))
        job._interactive_disable_log_file = True
        job.structure = ase_to_pyiron(structure_template)
        job.input['mole_fractions'] = mole_fraction_dict
        job.input['iterations'] = 1e6
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

    # calculation
    project = Project(working_directory)
    lmp_mini1 = project.create_job(LammpsInteractiveWithoutOutput, "lmp_mini_" + str(i),
                                   delete_existing_job=True)
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

    # Elastic constants
    project = Project(working_directory)
    lmp_elastic = project.create_job(LammpsInteractiveWithoutOutput, "lmp_elastic_" + str(i),
                                     delete_existing_job=True)
    lmp_elastic.structure = ase_to_pyiron(structure)
    lmp_elastic.potential = potential
    lmp_elastic.interactive_enforce_structure_reset = True
    lmp_elastic.interactive_mpi_communicator = MPI.COMM_SELF
    lmp_elastic.server.run_mode.interactive = True
    lmp_elastic._interactive_disable_log_file = True  # disable lammps.log
    elastic = lmp_elastic.create_job(ElasticMatrixJobWithoutFiles, "elastic_" + str(i),
                                     delete_existing_job=True)
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
                sum(elastic.ref_job.structure.indices == elastic.ref_job.structure.get_species_symbols().tolist().index(
                    el)) / len(elastic.ref_job.structure.indices))
        else:
            conc_lst.append(0.0)

    return elastic_constants_lst + conc_lst


def combined_function(input_parameter):
    i, mole_fraction_dict, structure_template, element_lst, potential, working_directory = input_parameter

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