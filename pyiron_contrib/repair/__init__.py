import abc
import datetime
import re
import math
import os
from collections import defaultdict
from functools import wraps
import tarfile
from typing import Union, Iterable

from pyiron_atomistics import ase_to_pyiron
from ase.io import read as ase_read
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob
from pyiron_base import GenericJob, GenericMaster
from pyiron_snippets.logger import logger

from tqdm.auto import tqdm


class RepairError(Exception):
    pass


class NoMatchingTool(RepairError):
    pass


class RestartFailed(RepairError):
    pass


class FixFailed(RepairError):
    pass


def get_job_error_log(job):
    log = job["error.out"]
    if log is None:
        log = job["error.msg"]
    return log


class LineMatcher(abc.ABC):
    @abc.abstractmethod
    def match(self, line):
        pass


class FullLine(LineMatcher):
    def __init__(self, line):
        self._line = line
        # GOTCHA for next Marvin: what happens when the last line of a log
        # doesn't have a new line?
        if self._line[-1] != "\n":
            self._line += "\n"

    def match(self, line):
        return self._line == line


class PartialLine(LineMatcher):
    def __init__(self, line):
        self._line = line

    def match(self, line):
        return self._line in line


def match_in_error_log(match_lines, job):
    if isinstance(match_lines, (str, LineMatcher)):
        match_lines = [match_lines]
    match_lines = [
        m if isinstance(m, LineMatcher) else FullLine(m) for m in match_lines
    ]
    lines = get_job_error_log(job) or []
    return any(any(m.match(l) for m in match_lines) for l in lines)


class RepairTool(abc.ABC):

    @abc.abstractmethod
    def match(self, job: GenericJob) -> bool:
        """
        Return True if this tool can fix the job.
        """
        pass

    def fix_inplace(self, job: GenericJob, handyman) -> bool:
        """
        Try to fix a job in place without restarting it.

        Args:
            job (GenericJob): job to fix
            handyman: back reference so the tool can restart child jobs

        Returns:
            bool: True if job could be fixed in place, False otherwise
        """
        return False

    @abc.abstractmethod
    def fix(self, old_job: GenericJob, new_job: GenericJob):
        """
        Prepare a new job with the fix applied.
        """
        pass

    hamilton = "generic"
    """Name of job class that this tool can fix or 'generic'"""
    applicable_status = ("aborted",)
    """Tuple of job status strings that this tool can fix"""
    priority = 0
    """If multiple tools exist that can fix a job, pick the one with highest priority"""


class ConstructionSite:
    def __init__(self, fixing, hopeless, failed):
        self._fixing = fixing
        self._hopeless = hopeless
        self._failed = failed

    @property
    def fixing(self):
        return self._fixing

    @property
    def hopeless(self):
        return self._hopeless

    @property
    def failed(self):
        return self._failed


class HandyMan:

    def __init__(
        self, tools: Union[None, Iterable[RepairTool]] = None, suppress_fix_errors=True
    ):
        self.shed = defaultdict(list)
        self._suppress_fix_errors = suppress_fix_errors

        if tools is None:
            tools = DEFAULT_SHED
        if isinstance(tools, Iterable):
            for tool in tools:
                self.register(tool)
        else:
            raise TypeError(
                "tools must a list of RepairTools or a dict of status to RepairTools."
            )

    def register(self, tool, status=None):
        if status is None:
            status = tool.applicable_status
        if isinstance(status, str):
            status = [status]
        for s in status:
            self.shed[(s, tool.hamilton)].append(tool)

    def restart(self, job):
        new = job.restart()
        return new

    def fix_job(self, tool, job, graveyard=None):
        try:
            new_job = self.restart(job)
        except Exception as e:
            raise RestartFailed(e) from None

        history = job["user/handyman/history"] or []
        history.append(type(tool).__name__)
        new_job["user/handyman/last"] = history[-1]
        new_job["user/handyman/history"] = history

        try:
            tool.fix(job, new_job)
        except Exception as e:
            if self._suppress_fix_errors:
                raise FixFailed(e) from None
            else:
                raise

        new_job.write_input()
        new_job.save()
        new_job._restart_file_list = []
        new_job._restart_file_dict = {}

        mid = job.master_id
        pid = job.parent_id

        name = job.name
        if graveyard is None:
            job.remove()
        else:
            try:
                no = len(job.content["user/handyman/history"])
            except KeyError:
                no = 0
            job.rename(f"{job.name}_fix_{no}")
            job.move_to(graveyard)
        new_job.rename(name)

        new_job.master_id = mid
        new_job.parent_id = pid
        new_job.server.queue = "cmti"
        return new_job

    def find_tool(self, job):
        # try job specific tools first, otherwise sort by priority such that
        # highest comes first
        tools = sorted(
            self.shed[(job.status.string, job.__name__)],
            key=lambda tool: -tool.priority,
        ) + sorted(
            self.shed[(job.status.string, "generic")], key=lambda tool: -tool.priority
        )
        for tool in tools:
            try:
                if tool.match(job):
                    return tool
            except Exception as e:
                logger.warn(f"Matching {tool} on job {job.id} failed with {e}!")
        raise NoMatchingTool("Cannot find stuitable tool!")

    def fix_project(
        self, project, server_override={}, refresh=True, graveyard=None, **kwargs
    ):
        """
        Fix broken jobs.

        Args:
            project (Project): search this project for broken jobs
            server_override (dict): override these values on the restarted jobs
            graveyard (Project): move fixed projects here, instead of deleting
            **kwargs: pass through project.job_table when searching; use to
                      restrict what jobs are fixed
        """
        if refresh:
            project.refresh_job_status()

        hopeless = []
        failed = {}
        fixing = defaultdict(list)
        status_list = set([k[0] for k in self.shed.keys()])
        job_ids = tqdm(
            project.job_table(**kwargs).query("status.isin(@status_list)").id
        )
        for jid in job_ids:
            try:
                job = project.load(jid)
            except IndexError:
                logger.warning(f"Failed to load job {jid}, skipping.")
            try:
                if (
                    job.master_id is not None
                    and project.get_job_status(job.master_id) == "running"
                ):
                    logger.warning(f"Job {jid}'s master is still running, skipping!")
                    continue
                if (
                    job.parent_id is not None
                    and project.get_job_status(job.parent_id) == "running"
                ):
                    logger.warning(f"Job {jid}'s parent is still running, skipping!")
                    continue
            except IndexError:
                pass  # parent or master jobs don't exist anymore, ignore
            try:
                tool = self.find_tool(job)
                fixing[type(tool).__name__].append(job.id)
                if not tool.fix_inplace(job, self):
                    job = self.fix_job(tool, job, graveyard=graveyard)
                    for k, v in server_override.items():
                        setattr(job.server, k, v)
                    job.run()
            except NoMatchingTool:
                hopeless.append(job.id)
            except RepairError as e:
                failed[job.id] = e

        return ConstructionSite(fixing, hopeless, failed)


class TimeoutTool(RepairTool):

    def __init__(self, time_factor=2):
        """
        Increase runtime by this factor.
        """
        self._time_factor = time_factor

    applicable_status = ("aborted", "collect")

    def match(self, job: GenericJob) -> bool:
        return match_in_error_log(PartialLine("DUE TO TIME LIMIT"), job)

    @wraps(RepairTool.fix_inplace)
    def fix_inplace(self, job, handyman):
        if job.status.aborted and isinstance(job, GenericMaster):
            # first check if children can be fixed normally
            cs = handyman.fix_project(job.child_project)
            atleast_one = len(cs.fixing) > 0
            # if they cannot; assume they got killed during the timeout or
            # got created and didn't get to run
            for cid in cs.hopeless:
                child = job.load(cid)
                last = child["user/handyman/last"]
                if last != "RestartTool":
                    child = handyman.fix_job(RestartTool(), child)
                    child.run()
                    atleast_one = True
            # if we restarted at least one child, tell the handy man not to
            # restart this job
            return (
                atleast_one
                or job.child_project.get_jobs_status().get("submitted", 0) > 0
            )
        elif job.status.collect and not isinstance(job, GenericMaster):
            try:
                job.decompress()
            except (tarfile.ReadError, EOFError):
                # tarfile broken, ie. job got killed during compression
                # simply remove and collect again
                tfile = os.path.join(job.working_directory, job.job_name + ".tar.bz2")
                os.remove(tfile)
            job.run()
            return True
        else:
            return False

    def fix(self, old_job: GenericJob, new_job: GenericJob):
        for line in get_job_error_log(old_job):
            matches = re.findall("CANCELLED AT (.*) DUE TO TIME LIMIT", line)
            if len(matches) != 0:
                stop = datetime.datetime.strptime(matches[0], "%Y-%m-%dT%H:%M:%S")
                break
        run_time = stop - old_job.database_entry.timestart
        new_job.server.run_time = run_time.total_seconds() * self._time_factor


class RestartTool(RepairTool):
    """
    Naive tool that matches any job and simply restarts it w/o modification.

    Do not use by default, this is mainly a tool to be used by other tools
    internally.
    """

    def match(self, job):
        return True

    def fix(self, *_):
        pass

    applicable_status = ()


class AtomisticRepairTool(RepairTool, abc.ABC):
    """
    AtomisticGenericJob.restart() copies over the last structure to the
    initial structure of the new job.  That's not want we generally want,
    because e.g. for minimization or md we usually want the whole trajectory.

    This is a stop gap; the correct solution changes .restart() to .continue()
    in base and adds a new restart that just copies input files.
    """

    def __init__(self, *, copy_final_structure=False):
        super().__init__()
        self._copy_final_structure = copy_final_structure

    def match(self, job):
        return isinstance(job, AtomisticGenericJob)

    def fix(self, old_job: GenericJob, new_job: GenericJob):
        if (
            not self._copy_final_structure
            and old_job._generic_input["calc_mode"] != "static"
        ):
            new_job.structure = old_job.structure


class VaspTool(AtomisticRepairTool, abc.ABC):

    hamilton = "Vasp"


class VaspNbandsTool(VaspTool):

    def __init__(self, state_factor=2, **kwargs):
        """
        Increase the number of empty states by this factor.
        """
        super().__init__(**kwargs)
        self._state_factor = state_factor

    def match(self, job: GenericJob) -> bool:
        return super().match(job) and not job.nbands_convergence_check()

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        old_states = old_job["output/generic/dft/bands/occ_matrix"].shape[-1]
        n_elect = old_job.get_nelect()
        # double free states
        new_job.set_empty_states(
            math.ceil((old_states - n_elect // 2) * self._state_factor)
        )

        try:
            new_job.restart_file_list.append(str(old_job.files.CHGCAR))
            new_job.input.incar["ICHARG"] = 1
        except FileNotFoundError:
            # run didn't include CHGCAR file
            pass

    applicable_status = ("not_converged",)


class VaspDisableIsymTool(VaspTool):
    """
    Assorted symmetry errors, just turn symmetry off.
    """

    def match(self, job):
        return (
            super().match(job)
            and job.input.incar["ISYM"] != 0
            and match_in_error_log(
                [
                    " inverse of rotation matrix was not found (increase SYMPREC)       5",
                    " RHOSYG internal error: stars are not distinct, try to increase SYMPREC to e.g. ",
                    " POSMAP internal error: symmetry equivalent atom not found,",
                    " VERY BAD NEWS! internal error in subroutine PRICEL "
                    "(probably precision problem, try to change SYMPREC in INCAR ?):",
                    " VERY BAD NEWS! internal error in subroutine INVGRP:",
                    PartialLine(
                        "Inconsistent Bravais lattice types found for crystalline and"
                    ),
                    PartialLine("Found some non-integer element in rotation matrix"),
                ],
                job,
            )
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        new_job.input.incar["ISYM"] = 0

    applicable_status = ("aborted",)


class VaspSubspaceTool(VaspTool):
    """
    Lifted from custodian.
    """

    def match(self, job):
        return super().match(job) and match_in_error_log(
            PartialLine("ERROR in subspace rotation PSSYEVX"), job
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        new_job.input.incar["ALGO"] = "Normal"

    applicable_status = ("aborted",)


class VaspZbrentTool(VaspTool):
    """
    Lifted from custodian.
    """

    def match(self, job):
        if (
            job.input.incar["EDIFF"] <= 1e-6
            and job["user/handyman/last"] == "VaspZbrentTool"
        ):
            logger.warning(
                "Bailing to apply VaspZbrentTool! Already tried once and "
                "you didn't think it through this far yet!"
            )
        return match_in_error_log(
            [
                PartialLine("ZBRENT: fatal error in bracketing"),
                PartialLine("ZBRENT: fatal error: bracketing interval incorrect"),
            ],
            job,
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        ediff = old_job.input.incar.get("EDIFF", 1e-4)
        if ediff > 1e-6:
            new_job.input.incar["EDIFF"] = 1e-6
        else:
            contcar = ase_to_pyiron(ase_read(str(old_job.files.CONTCAR)))
            # VASP manual recommend to copy CONTCAR to POSCAR, but if we
            # overwrite the structure in pyiron directly with the one read from
            # CONTCAR we'll lose spins and charges
            new_job.structure.positions[:] = contcar.positions
            new_job.structure.cell[:] = contcar.cell
            # Job is a relaxation run, including positions
            if new_job.input.incar.get("ISIF", 2) not in [5, 6, 7]:
                new_job.structure.rattle(1e-2)
            # ZBRENT is caused by small and noisy forces; when starting on a
            # high symmetry configuration, forces start out small, confusing
            # the VASP minimizer; instead let's try to shake us away a bit from
            # equilibrium to get higher forces
        nelmin = old_job.input.incar["NELMIN"]
        if nelmin is None or nelmin < 8:
            new_job.input.incar["NELMIN"] = 8

    applicable_status = ("aborted",)


class VaspZpotrfZtrtri(VaspTool):
    """
    https://github.com/protik77/VASP-error-fix#error-scalapack-routine-zpotrf-ztrtri-failed
    https://2www.vasp.at/forum/viewtopic.php?p=21120

    Apparently this is caused by a non invertible hamiltonian.

    Usually happens when atoms are too close
    Suggested solution to change ALGO or NBANDS if not to change the
    hamiltonian to be solved
    """

    def match(self, job):
        return super().match(job) and match_in_error_log(
            PartialLine("LAPACK: Routine ZPOTRF ZTRTRI failed!"), job
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        algo = old_job.input.incar.get("ALGO", "Fast")
        mapa = {
            "Normal": "Fast",
            "Fast": "VeryFast",
            "VeryFast": "Normal",
        }
        new_job.input.incar["ALGO"] = mapa.get(algo, "Fast")


class VaspZpotrfTool(VaspTool):
    """
    Lifted from custodian.
    """

    def match(self, job):
        return super().match(job) and match_in_error_log(
            PartialLine("LAPACK: Routine ZPOTRF failed!"), job
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        new_job.input.incar["ISYM"] = 0
        new_job.input.incar["POTIM"] = old_job.input.incar.get("POTIM", 0.5) / 2
        new_job._restart_file_list = []
        new_job._restart_file_dict = {}

    applicable_status = ("aborted",)


class VaspEddavTool(VaspTool):
    """
    Lifted from custodian.
    """

    def match(self, job):
        return (
            super().match(job)
            and job.input.incar["ALGO"] != "All"
            and match_in_error_log(
                PartialLine("Error EDDDAV: Call to ZHEGV failed."), job
            )
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        new_job.input.incar["ALGO"] = "All"

    applicable_status = ("aborted",)


class VaspMinimizeStepsTool(VaspTool):
    """
    Ionic Minimization didn't converge.

    For simplicity, just restart with more steps instead of continuing.
    """

    def __init__(self, factor=2, **kwargs):
        super().__init__(**kwargs)
        self._factor = factor

    def match(self, job):
        return (
            super().match(job)
            and job.input.incar["IBRION"] != -1
            and job.input.incar["NSW"] == len(job["output/generic/dft/scf_energy_free"])
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        new_job.input.incar["NSW"] = int(
            old_job.input.incar.get("NSW", 100) * self._factor
        )
        new_job.input.incar["EDIFF"] = 1e-6
        new_job._restart_file_list = []
        new_job._restart_file_dict = {}

    applicable_status = ("not_converged",)


class VaspElectronicConvergenceTool(VaspTool):
    """ """


class VaspTooManyKpointsIsym(VaspTool):
    """
    Occurs when too many k-points are requested.

    Apparently there's a limit of 20k unique k-points
    https://www.error.wiki/VERY_BAD_NEWS!_internal_error_in_subroutine_IBZKPT

    If symmetry is off, try to turn it on.
    """

    def match(self, job):
        return (
            super().match(job)
            and match_in_error_log(
                [
                    PartialLine("VERY BAD NEWS! internal error in subroutine IBZKPT"),
                    PartialLine("NKPT>NKDIM"),
                ],
                job,
            )
            and job.input.incar["ISYM"] == 0
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        new_job.input.incar["ISYM"] = 1

    applicable_status = ("aborted",)


class VaspTooManyKpointsTruncate(VaspTool):
    """
    Occurs when too many k-points are requested.

    Apparently there's a limit of 20k unique k-points
    https://www.error.wiki/VERY_BAD_NEWS!_internal_error_in_subroutine_IBZKPT

    Simply lower the requested k-points below the acceptable limit.

    .. warning::
        This might change the results of the calculation!
    """

    def match(self, job):
        return super().match(job) and match_in_error_log(
            [
                PartialLine("VERY BAD NEWS! internal error in subroutine IBZKPT"),
                PartialLine("NKPT>NKDIM"),
            ],
            job,
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        kpoints = [min(45, int(k)) for k in old_job.input.kpoints[3].split()]
        new_job.set_kpoints(kpoints)


class VaspSetupPrimitiveCellTool(VaspTool):
    """
    Vasp recommends "changing" SYMPREC or refining POSCAR.

    I assume this means increasing SYMPREC, i.e. to larger values.
    """

    def match(self, job):
        return super().match(
            job
        ) and " internal error in VASP: SETUP_PRIMITIVE_CELL, S_NUM not divisible by NPCELL" in get_job_error_log(
            job
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        symprec = old_job.input.incar.get("SYMPREC", 1e-5)
        new_job.input.incar["SYMPREC"] = symprec * 10

    applicable_status = ("aborted",)


class VaspMemoryErrorTool(VaspTool):
    """
    Random crashes without any other indication are usually because memory ran
    out.  Increase the number of cores to have more nodes/memory available.
    """

    def __init__(self, factor=2, max_cores=160, **kwargs):
        super().__init__(**kwargs)
        self._factor = factor
        self._max_cores = max_cores

    def match(self, job):
        # coredump = 'Image              PC                Routine Line Source \n' in job['error.out']
        # return malloc or (forrtl and coredump)
        too_many_cores = job.server.cores >= self._max_cores
        return (
            super().match(job)
            and not too_many_cores
            and match_in_error_log(
                [
                    "forrtl: error (78): process killed (SIGTERM)",
                    "malloc(): corrupted top size",
                    PartialLine("Out of memory (unable to allocate a 'MPI_Info')"),
                ],
                job,
            )
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        if old_job.server.cores < self._max_cores:
            new_cores = old_job.server.cores * self._factor
        else:
            new_cores = old_job.server.cores
        new_job.server.cores = new_cores
        if new_cores >= 40:
            new_job.input.incar["NCORE"] = 20
        elif new_cores >= 20:
            new_job.input.incar["NCORE"] = 10

    applicable_status = ("aborted",)


class VaspEddrmmTool(VaspTool):
    def match(self, job):
        return super().match(job) and match_in_error_log(
            PartialLine("WARNING in EDDRMM: call to ZHEGV failed, returncode ="), job
        )

    def fix_inplace(self, job: GenericJob, handyman) -> bool:
        job.set_eddrmm_handling(status="ignore")
        job.status.collect = True
        job.run()
        return True

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        new_job.input.incar["ALGO"] = "Normal"
        try:
            new_job.restart_file_list.append(old_job.get_workdir_file("CHGCAR"))
            new_job.input.incar["ICHARG"] = 1
        except FileNotFoundError:
            # run didn't include CHGCAR file
            pass

    applicable_status = ("warning",)


class VaspSgrconTool(VaspTool):
    """
    custodian recommends changing to a gamma centered mesh.
    """

    def match(self, job):
        return (
            super().match(job)
            and job.input.kpoints[2] == "Monkhorst_Pack"
            and match_in_error_log(
                PartialLine("VERY BAD NEWS! internal error in subroutine SGRCON"), job
            )
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        new_job.set_kpoints(old_job.kpoint_mesh, scheme="GC")
        return new_job

    applicable_status = ("aborted",)


# class VaspLongCellAmin(VaspTool):
# def match(self, job):
#     return any([
#         "One of the lattice vectors is very long (>50 A), but AMIN is rather" in l
#             for l in job['OUTCAR']
#     ])

# def fix(self, old_job, new_job):
#     # vasp recommends 0.01 in the message, if that doesn't work let's try
#     # with smaller again
#     amin = old_job.input.incar.get("AMIN", 0.02)
#     new_job.input.incar['AMIN'] = amin / 2


class MurnaghanTool(RepairTool, abc.ABC):
    hamilton = "Murnaghan"


class MurnaghanFinishedChildrenTool(MurnaghanTool):

    def match(self, job):
        return (job.child_project.job_table().status == "finished").all()

    @wraps(RepairTool.fix_inplace)
    def fix_inplace(self, job, handyman):
        job.status.collect = True
        job.run()
        return True

    def fix(self, old_job, new_job):
        assert False, "Shouldn't happen!"

    applicable_status = ("aborted", "collect")


class MurnaghanAllowAbortedChildrenTool(MurnaghanTool):
    """
    Retroactively allow some children of Murnaghans to fail, by changing the
    input and collecting again.
    """

    applicable_status = ("aborted",)

    def __init__(self, allow_aborted: Union[int, float] = 0.1):
        """
        Args:
            allow_aborted (int, float): number of children that are allowed to fail; if given as a float in (0,1), use
                                        as percentage of children
        """
        self._allow_aborted = allow_aborted

    def match(self, job):
        status = job.child_project.get_jobs_status()
        # As long as child jobs are still running, do not do anything
        if status.get("running", 0) > 0 or status.get("submitted", 0) > 0:
            return False
        allow_aborted = self._allow_aborted
        if 0 < self._allow_aborted < 1:
            allow_aborted *= status.sum()
        return 0 < status.get("aborted", 0) < allow_aborted

    @wraps(RepairTool.fix_inplace)
    def fix_inplace(self, job, handyman):
        allow_aborted = self._allow_aborted
        if 0 < self._allow_aborted < 1:
            allow_aborted *= len(job.child_project.job_table(recursive=False))
        job.input["allow_aborted"] = allow_aborted
        job.status.collect = True
        job.run()
        return True

    def fix(self, old_job, new_job):
        assert False, "Shouldn't happen!"


class SphinxTool(AtomisticRepairTool, abc.ABC):

    hamilton = "Sphinx"


class SphinxSymmetryOffTool(RepairTool):

    def match(self, job):
        return (
            super().match(job)
            and "Symmetry inconsistency error: symmetries are no group\n"
            in job["sphinx.log"]
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        new_job.input._force_load()
        new_job.input.sphinx.structure.create_group("symmetry")
        return new_job


### Classes below are experimental
class VaspSymprecTool(VaspTool):

    def match(self, job):
        return super().match(job) and match_in_error_log(
            [
                " inverse of rotation matrix was not found (increase SYMPREC) 5",
                " POSMAP internal error: symmetry equivalent atom not found,",
            ],
            job,
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        symprec = old_job.input.incar.get("SYMPREC", 1e-5)
        new_job.input.incar["SYMPREC"] = 10 * symprec


class VaspRhosygSymprecTool(VaspTool):

    def match(self, job):
        return super().match(job) and match_in_error_log(
            " RHOSYG internal error: stars are not distinct, try to increase SYMPREC to e.g. ",
            job,
        )

    def fix(self, old_job, new_job):
        super().fix(old_job, new_job)
        new_job.input.incar["SYMPREC"] = 1e-4


DEFAULT_SHED = [
    TimeoutTool(2),
    MurnaghanFinishedChildrenTool(),
    VaspDisableIsymTool(),
    VaspSgrconTool(),
    VaspSubspaceTool(),
    VaspZbrentTool(),
    VaspZpotrfTool(),
    VaspEddavTool(),
    VaspSetupPrimitiveCellTool(),
    VaspTooManyKpointsIsym(),
    VaspMemoryErrorTool(max_cores=320),
    VaspNbandsTool(1.5),
    VaspMinimizeStepsTool(2),
    VaspEddrmmTool(),
]
