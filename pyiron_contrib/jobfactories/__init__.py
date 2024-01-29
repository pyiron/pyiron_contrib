from pyiron_base import HasStorage, GenericJob
from pyiron_atomistics import Atoms
from pyiron_contrib import Project

from abc import ABC, abstractmethod
import contextlib
from typing import Optional, Callable, Dict


class JobFactory(HasStorage, ABC):
    """
    A small class for high throughput job execution.

    This class and its sub classes can be used to specify pyiron jobs and their
    options with instantiating the actual job classes.  They can be easily
    saved into HDF.  Actual jobs can be created from them with :meth:`.run` or
    :meth:`.make`.  This allows them to be passed around in complex workflows
    which can then be written without reference to actual job classes, so that
    the underlying calculation can be swapped out easily.

    >>> from pyiron_atomistics import Project
    >>> pr = Project('.')
    >>> cu = pr.create.structure.bulk('Cu')

    >>> f = GenericJobFactory('Lammps')
    >>> f.attr.potential = 'my_potential'
    >>> f.calc_md(n_ionic_steps=1_000, temperature=100)
    >>> f.project = pr
    >>> j = f.run('my_lammps', modify=lambda job: job, structure=cu)

    would be roughly equivalent to

    >>> if 'my_lammps' not in pr.list_nodes() or pr.get_job_status(name) not in ['finished', 'submitted']:
    ...   j = pr.create.job.Lammps('my_lammps')
    ...   j.potential = 'my_potential'
    ...   j.structure = cu
    ...   j.calc_md(n_ionic_steps=1_000, temperature=100)
    ...   # modify is called here
    ...   j.run()
    ... else:
    ...   j = None

    For a single job this is ok syntax sugar, but for large numbers of
    submitted jobs this can become a substantial simplification, i.e.

    >>> import numpy as np
    >>> strains = np.linspace(-0.5, 1.5, 500)

    >>> for eps in strains:
    ...   j.run(f'my_lammps_{eps}', modify=lambda j: j['user/strain'] := eps and j, structure=cu.apply_strain(eps-1, return_box=True))

    is much easier to understand than a loop around the above "standard"
    construct.  While one could use functions to save a similar amount of
    typing, functions are much harder to compose and serialize long term.

    To call any method on a to be created job, just call the same method on the
    job factory.  To set any attribute on it, set it on :attr:`.attr` of the
    job factory.
    """

    def __init__(self):
        super().__init__()
        self.storage.create_group("input")
        self._project_nodes = None

    @abstractmethod
    def _get_hamilton(self):
        """
        Name of the job class that should be used, as registered to the
        pyiron_base job factory.
        """
        pass

    @property
    def hamilton(self):
        return self._get_hamilton()

    @property
    def attr(self):
        """
        Access to attributes, that should be set on the job after creation.
        """
        return self.storage.create_group("attributes")

    @property
    def project(self) -> Project:
        """
        Project that jobs should be created in by :meth:`.make`.
        """
        return self._project

    @project.setter
    def project(self, value: Project):
        self._project = value
        self._project_nodes = None

    @property
    def server(self):
        """
        Access to run time related options, the following are recognized:
            - queue
            - cores
            - run_time
        """
        return self.storage.create_group("server")

    @property
    def cores(self):
        return self.server.get("cores", self.storage.get("cores", None))

    @cores.setter
    def cores(self, cores):
        self.server.cores = cores

    @property
    def run_time(self):
        return self.server.get("run_time", self.storage.get("run_time", None))

    @run_time.setter
    def run_time(self, cores):
        self.server.run_time = cores

    @property
    def queue(self):
        return self.server.get("queue", self.storage.get("queue", None))

    @queue.setter
    def queue(self, cores):
        self.server.queue = cores

    def copy(self):
        """
        Return a deep copy.
        """
        copy = type(self)()
        copy.storage.clear()
        copy.storage.update(self.storage.copy())
        copy.project = self.project
        return copy

    def set_input(self, **kwargs):
        """
        Set attributes on the job's input after it is created.
        """
        for key, value in kwargs.items():
            self.input[key] = value
            # self.storage.input[key] = value

    @property
    def input(self):
        return self.storage.create_group("input")

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def wrapper(*args, **kwargs):
            d = self.storage.create_group(f"methods/{name}")
            d["args"] = args
            d["kwargs"] = kwargs

        return wrapper

    def _prepare_job(self, job, structure):
        if structure is not None:
            job.structure = structure
        if self.queue is not None:
            job.server.queue = self.queue
        if self.cores is not None:
            job.server.cores = self.cores
        if self.run_time is not None:
            job.server.run_time = self.run_time
        for k, v in self.storage.input.items():
            job.input[k] = v
        if "methods" in self.storage:
            for meth, ka in self.storage.methods.items():
                getattr(job, meth)(*ka.args, **ka.kwargs)
        if "attributes" in self.storage:
            for attr, val in self.storage.attributes.items():
                setattr(job, attr, val)
        return job

    def _project_list_nodes(self):
        if self._project_nodes is None:
            self._project_nodes = self.project.list_nodes()
        return self._project_nodes

    def make(
        self,
        name: str,
        modify: Callable[[GenericJob], GenericJob],
        structure: Atoms,
        delete_existing_job=False,
        delete_aborted_job=True,
    ) -> Optional[GenericJob]:
        """
        Create a new job if necessary.

        Args:
            name (str): name of the new job
            modify (str): a function to make individual changes to the job if it is created
            structure (Atoms): the structure to set on the job
            delete_existing_job, delete_aborted_job: passed through normal job creation

        Returns:
            GenericJob: if job newly created
            None: if job already existed and no action was taken
        """
        # short circuit if job already successfully ran
        if not delete_existing_job and (
            name in self._project_list_nodes()
            and self.project.get_job_status(name) in ["finished", "submitted"]
        ):
            return None

        job = getattr(self.project.create.job, self.hamilton)(
            name,
            delete_existing_job=delete_existing_job,
            delete_aborted_job=delete_aborted_job,
        )
        if not job.status.initialized:
            return None

        # FIXME: think about; when submitting large number of jobs with this
        # function that are all new, we can lose up 25% of run time by
        # recomputing this every time
        # adding new jobs, invalidate node cache
        # self._project_nodes = None

        job = self._prepare_job(job, structure)
        job = modify(job) or job
        return job

    def run(
        self,
        name: str,
        modify: Callable[[GenericJob], GenericJob],
        structure: Atoms,
        delete_existing_job: bool = False,
        delete_aborted_job: bool = True,
        silence: bool = True,
    ) -> Optional[GenericJob]:
        """
        First make a job, then run it if necessary.

        Args:
            name (str): name of the new job
            modify (str): a function to make individual changes to the job if it is created
            structure (Atoms): the structure to set on the job
            delete_existing_job, delete_aborted_job: passed through normal job creation
            silence (bool): redirect standard output while calling `job.run()`.

        Returns:
            GenericJob: if job newly created
            None: if job already existed and no action was taken
        """
        job = self.make(
            name, modify, structure, delete_existing_job, delete_aborted_job
        )
        if job is None:
            return
        if silence:
            with open("/dev/null", "w") as f, contextlib.redirect_stdout(f):
                job.run()
        else:
            job.run()
        return job


class GenericJobFactory(JobFactory):
    def __init__(self, hamilton=None):
        super().__init__()
        if hamilton is not None:
            self.storage.hamilton = hamilton

    def _get_hamilton(self):
        return self.storage.hamilton


class MasterJobFactory(GenericJobFactory):
    def set_ref_job(self, ref_job):
        self.storage.ref_job = ref_job

    def _prepare_job(self, job, structure):
        job.ref_job = self.storage.ref_job
        super()._prepare_job(job, structure)
        return job


class DftFactory(JobFactory):
    def set_empty_states(self, states_per_atom):
        self.storage.empty_states_per_atom = states_per_atom

    def _prepare_job(self, job, structure):
        job = super()._prepare_job(job, structure)
        if "empty_states_per_atom" in self.storage:
            job.input["EmptyStates"] = (
                len(structure) * self.storage.empty_states_per_atom + 3
            )
        return job


class VaspFactory(DftFactory):
    def __init__(self):
        super().__init__()
        self.storage.incar = {}
        self.storage.nband_nelec_map = None

    @property
    def incar(self):
        return self.storage.incar

    def enable_nband_hack(self, nelec: Dict[str, int]):
        """
        Set a per element NBANDS estimate.

        Structures far from (global) equilibrium may require more empty states than the default VASP provides.
        This allows to provide a mapping between elements and integers that give a "per element" NBAND that is summed
        over all atoms in a structure.
        """
        self.storage.nband_nelec_map = nelec

    def _get_hamilton(self):
        return "Vasp"

    def minimize_volume(self):
        self.calc_minimize(pressure=0.0, volume_only=True)

    def minimize_cell(self):
        self.calc_minimize()
        self.incar["ISIF"] = 5

    def minimize_internal(self):
        self.calc_minimize()

    def minimize_all(self):
        self.calc_minimize(pressure=0.0)

    def _prepare_job(self, job, structure):
        job = super()._prepare_job(job, structure)
        for k, v in self.incar.items():
            job.input.incar[k] = v
        if self.storage.nband_nelec_map is not None:
            # ensure we apply the hack only for structures where we know an NBAND estimate for all elements
            elems = set(self.storage.nband_nelec_map.keys())
            if elems.union(set(structure.get_chemical_symbols())) == elems:
                nelect = sum(
                    self.storage.nband_nelec_map[el]
                    for el in structure.get_chemical_symbols()
                )
                job.input.incar["NBANDS"] = nelect + len(structure)
        return job


class SphinxFactory(DftFactory):
    def _get_hamilton(self):
        return "Sphinx"


class LammpsFactory(JobFactory):
    @property
    def potential(self):
        return self.storage.potential

    @potential.setter
    def potential(self, value):
        self.storage.potential = value

    def _get_hamilton(self):
        return "Lammps"

    def _prepare_job(self, job, structure):
        super()._prepare_job(job, structure)
        job.potential = self.potential
        return job


class MlipFactory(LammpsFactory):
    def _get_hamilton(self):
        return "LammpsMlip"
