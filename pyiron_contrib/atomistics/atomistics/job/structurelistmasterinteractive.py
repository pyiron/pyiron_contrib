# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department

"""
Job class to run a reference jobs on all structures in a given container via an interactive job.
"""

__author__ = "Marvin Poul"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Marvin Poul"
__email__ = "poul@mpie.de"
__status__ = "development"
__date__ = "Jun 14, 2021"


from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_base import GenericMaster, DataContainer


class StructureMasterInt(GenericMaster):
    """
    Runs given structures with given reference job.

    This example shows how to run a simple energy vs distance calculation for a dimer.

    >>> from pyiron_atomistics import Project
    >>> from pyiron_contrib.atomistics.atomistics.job.structurestorage import StructureStorage
    >>> pr = Project("container")
    >>> container = StructureStorage()
    >>> for d in [0.7, 0.8, 0.9, 1, 1.1, 1.2]:
    ...     container.add_structure(pr.create.structure.atoms(["Fe", "Fe"], positions=[[0,0,0], [0,0,d]], cell=[10,10,10])
    >>> ref = pr.create.job.Lammps("ref")
    >>> ref.structure = container.get_structure()
    >>> ref.calc_static()
    >>> master = pr.create.job.StructureContainerInteractive("master")
    >>> master.container = container
    >>> master.ref_job = ref
    >>> master.run()
    """

    def __init__(self, project=None, job_name=None):
        super().__init__(project=project, job_name=job_name)
        self.input = DataContainer(table_name="parameters")
        self.input.container = None

    @property
    def container(self):
        return self.input.container

    @container.setter
    def container(self, container):
        self.input.container = container

    def add_structure(self, structure):
        """
        Add a structure to the StructureStorage to calculate.  If no container set yet, create an empty one.

        Args:
            structure (Atoms): structure to add to calculation
        """
        if self.input.container is None:
            self.input.container = StructureStorage()
        self.input.container.add_structure(structure)

    def validate_ready_to_run(self):
        if self.input.container is None:
            raise ValueError("No structure container set!")
        if self.ref_job is None:
            raise ValueError("No reference job set!")
        self.ref_job.validate_ready_to_run()
        copy = self.ref_job.copy_to(
            project=self.project_hdf5,
            new_job_name=f"{self.name}_calculator",
            new_database_entry=True,
        )
        self.append(copy)

    def run_static(self):
        self.status.running = True

        j = self.pop()

        j.interactive_open()
        j.interactive_enforce_structure_reset = True
        for structure in self.container.iter_structures():
            j.structure = structure
            j.run()
        j.interactive_close()

        self.status.collect = True
        self.run()

    def collect_output(self):
        self.to_hdf()

    def write_input(self):
        pass

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
        self.input.to_hdf(hdf=self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf=hdf, group_name=group_name)
        self.input.from_hdf(hdf=self.project_hdf5)
