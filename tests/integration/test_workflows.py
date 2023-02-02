
from unittest import TestCase
from pyiron_contrib import Project


class TestWorkflows(TestCase):
    def test_workflows(self):

        pr = Project("dir_if_theres_storage")
        wf = pr.create.workflow("subdir_if_theres_storage")
        nodes = pr.create.node

        wf.structure = nodes.atomistics.BulkStructure(element='Ni',
                                                        cubic=True,
                                                        repeat=3)

        wf.engine = nodes.atomistics.Lammps(structure=wf.structure.output.structure)

        wf.calc_md = nodes.atomistics.CalcMD(temperature=300,
                                               n_steps=1000,
                                               n_print=100,
                                               name="my_MD_run")
        wf.calc_md.input.job = wf.engine.output.job
        # Maybe file storage of input is default whitelisted
        wf.calc_md.output.steps.store()  # and file storage of output is default blacklisted
        wf.calc_md.output.temperature.store()  # so we turn on saving a couple things here

        wf.pressure = nodes.atomistics.Pressure(voigt=(1, 2, 3, 0, 0, 0))

        wf.plot = nodes.base.Plot2d(x=wf.calc_md.output.steps,
                                      y=wf.calc_md.output.temperature)
        wf.plot.output.plot.value.show()

        pr["subdir_if_theres_storage"].serialize()  # gets wf by name
        # Saves graph topology, and whatever IO we flagged to serialize
        # I have no idea yet how database connections should be made

        new_pr = Project(pr.name)
        new_wf = new_pr.create.workflow(wf.name)
        new_wf.deserialize()
        new_wf.plot.output.plot.value.show()
