import abc
from collections import namedtuple
import os.path
from typing import List
from typing import Optional
from sqlalchemy import ForeignKey, String, Integer, Column, create_engine
from sqlalchemy.exc import MultipleResultsFound, NoResultFound
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import pandas as pd

DatabaseEntry = namedtuple(
    "DatabaseEntry", ["name", "username", "project", "status", "jobtype"]
)

Base = declarative_base()


class Project(Base):
    __tablename__ = "project_table"

    id = Column(Integer, primary_key=True)
    location = Column(String(250))
    # too stupid to get the bydirectional thing going, whatevs...
    # jobs = relationship("Job", back_populates="project")
    # jobs = relationship("Job", backref="project")


# FIXME: Can be many-to-many later
class JobStatus(Base):
    __tablename__ = "job_status_table"

    id = Column(Integer, primary_key=True)
    status = Column(String(250))


# FIXME: Can be many-to-many later
class JobType(Base):
    __tablename__ = "job_type_table"

    id = Column(Integer, primary_key=True)
    type = Column(String(250))


class Job(Base):
    __tablename__ = "job_table"

    id = Column(Integer, primary_key=True)
    username = Column(String(250))
    name = Column(String(250))

    jobtype_id = Column(Integer, ForeignKey("job_type_table.id"))
    project_id = Column(Integer, ForeignKey("project_table.id"))
    status_id = Column(Integer, ForeignKey("job_status_table.id"))
    # project = relationship("Project", back_populates="jobs")


# TODO: this will be pyiron_base.IsDatabase
class GenericDatabase(abc.ABC):
    """
    Defines the abstract database interface used by all databases.
    """

    @abc.abstractmethod
    def add_item(self, entry: DatabaseEntry) -> int:
        pass

    @abc.abstractmethod
    def get_item(self, job_id: int) -> DatabaseEntry:
        """
        Return database entry of the specified job.

        Args:
            job_id (int): id of the job

        Returns:
            :class:`.DatabaseEntry`: database entry with the given id

        Raises:
            ValueError: if no job with the given id exists
        """
        pass

    @abc.abstractmethod
    def get_item_id(self, job_name: str, project_id: int) -> Optional[int]:
        pass

    @abc.abstractmethod
    def get_project_id(self, location: str) -> Optional[int]:
        pass

    @abc.abstractmethod
    def update_status(self, job_id, status):
        pass

    @abc.abstractmethod
    def remove_item(self, job_id: int) -> DatabaseEntry:
        pass

    @abc.abstractmethod
    def job_table(self) -> pd.DataFrame:
        pass


class TinyDB(GenericDatabase):
    """
    Minimal database implementation and "reference".  Exists mostly to allow easy testing without messing with
    DatabaseAccess.
    """

    def __init__(self, path, echo=False):
        self._path = path
        self._echo = echo
        kwargs = {}
        if path in (":memory:", ""):
            # this allows to access the same DB from the different threads in one process
            # it's necessary for an in memory database, otherwise all threads see different dbs
            kwargs["poolclass"] = StaticPool
            kwargs["connect_args"] = {"check_same_thread": False}
        self._engine = create_engine(
            f"sqlite:///{self._path}", echo=self._echo, **kwargs
        )
        Base.metadata.create_all(self.engine)
        Base.metadata.reflect(self.engine, extend_existing=True)

    @property
    def engine(self):
        return self._engine

    def add_item(self, entry: DatabaseEntry) -> int:
        with Session(self.engine) as session:
            project = (
                session.query(Project)
                .where(Project.location == entry.project)
                .one_or_none()
            )
            if project is None:
                project = Project(location=entry.project)
                session.add(project)
            jobtype = (
                session.query(JobType)
                .where(JobType.type == entry.jobtype)
                .one_or_none()
            )
            if jobtype is None:
                jobtype = JobType(type=entry.jobtype)
                session.add(jobtype)
            status = JobStatus(status=entry.status)
            session.add(status)
            session.flush()
            job = Job(
                name=entry.name,
                username=entry.username,
                project_id=project.id,
                status_id=status.id,
                jobtype_id=jobtype.id,
            )
            session.add(job)
            session.flush()
            job_id = job.id
            session.commit()
        return job_id

    def update_status(self, job_id, status):
        with Session(self.engine) as session:
            try:
                s = (
                    session.query(JobStatus)
                    .select_from(Job)
                    .where(Job.id == job_id, JobStatus.id == Job.status_id)
                    .one()
                )
                s.status = status
                session.commit()
            except Exception as e:
                raise ValueError(f"job_id {job_id} doesn't exist: {e}")

    def _row_to_entry(self, job_data):
        return DatabaseEntry(
            name=job_data.name,
            project=job_data.location,
            username=job_data.username,
            status=job_data.status,
            jobtype=job_data.type,
        )

    def get_item(self, job_id: int) -> DatabaseEntry:
        """
        Return database entry of the specified job.

        Args:
            job_id (int): id of the job

        Returns:
            :class:`.DatabaseEntry`: database entry with the given id

        Raises:
            ValueError: if no job with the given id exists
        """
        try:
            with Session(self.engine) as session:
                job_data = (
                    session.query(
                        Job.__table__, Project.location, JobStatus.status, JobType.type
                    )
                    .select_from(Job)
                    .where(Job.id == job_id)
                    .join(Project, Job.project_id == Project.id)
                    .join(JobStatus, Job.status_id == JobStatus.id)
                    .join(JobType, Job.jobtype_id == JobType.id)
                    .one()
                )
                return self._row_to_entry(job_data)
        except NoResultFound:
            raise ValueError(f"No job with id {job_id} found!") from None

    def get_item_id(self, job_name: str, project_id: int) -> Optional[int]:
        with Session(self.engine) as session:
            try:
                return (
                    session.query(Job.id)
                    .where(
                        Job.name == job_name,
                        Job.project_id == project_id,
                    )
                    .one()
                    .id
                )
            except (MultipleResultsFound, NoResultFound):
                return None

    def get_project_id(self, location: str) -> Optional[int]:
        with Session(self.engine) as session:
            try:
                return (
                    session.query(Project.id)
                    .where(Project.location == location)
                    .one()
                    .id
                )
            # FIXME: MultipleResultsFound should be reraised because it indicates a broken database
            except (MultipleResultsFound, NoResultFound):
                return None

    def remove_item(self, job_id: int) -> DatabaseEntry:
        # FIXME: probably a bit inefficient, because it makes two connections to the DB
        entry = self.get_item(job_id)
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            session.delete(job)
            session.commit()
        return entry

    def job_table(self) -> pd.DataFrame:
        with Session(self.engine) as session:
            query = (
                session.query(
                    Job.__table__, Project.location, JobStatus.status, JobType.type
                )
                .select_from(Job)
                .join(Project, Job.project_id == Project.id)
                .join(JobStatus, Job.status_id == JobStatus.id)
                .join(JobType, Job.jobtype_id == JobType.id)
            )
            return pd.DataFrame([r._asdict() for r in query.all()])
