import abc
from collections import namedtuple
import os.path
from typing import List
from typing import Optional
from sqlalchemy import \
    create_engine, \
    Table, Column, \
    ForeignKey, UniqueConstraint, \
    Integer, BigInteger, String, TIMESTAMP, \
    select
from sqlalchemy.exc import MultipleResultsFound, NoResultFound
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import json

import pandas as pd


def get_sqlalchemy_class_doc(tablename, classname, **kwargs):
    """
    produces the docstrings from the inserted arguments for sqlalchemy classes
    :param tablename: the name of the table, seen in the DB
    :param classname: the name of the python class, for sqlalchemy objects
    :param kwargs: the columns/parameters of the table/class in the DB/sqlalchemy
    :return: the docstring of the class
    """
    ...  # siehe Project

    col_discr = ""
    col_xmpls = ""
    for key in kwargs:
        col_discr += f'\t-{key}: {kwargs[key][0]}\n'
        if not "id" in key:
            col_xmpls += f'\t\t\t{key}="{kwargs[key][1]}"\n'

    micro_description = f'\n\tsqlalchemy class to construct "{tablename}" table'
    implementation = f"\n\n\timplementation:\n{col_discr}"
    example = "\n\tto insert a dataset:"
    example += "\n\twith Session(engine) as session:"
    example += f'\n\t\txmpl = {classname}(\n{col_xmpls}\t\t)'
    example += "\n\t\tsession.add(xmpl)\n\t\tsession.commit()"

    return micro_description + implementation + example

DatabaseEntry = namedtuple(
    "DatabaseEntry", [
        "job", "subjob", "servers", "cores", "timestart", "timestop", "totalworloadtime",
        "user",
        "project",
        "jobstatus",
        "jobtype", "typeversion",
        "submitting_host",
        "queue"
    ]
)

Base = declarative_base()


class Project(Base):
    """
    (what is it? micro summary)
    sqlalchemy class to construct 'project' table

    (details
    - implementation
      - columns
    - examples
    - explanations to class attributes)
    """

    # the name of the projects table in the DB
    __tablename__ = "projects"

    __doc__ = get_sqlalchemy_class_doc(
        tablename=__tablename__,
        classname=__name__,
        project_id=["primary key, automatic"],
        project=["string, the project name", "/testproject"]
    )

    # the columns of projects table
    project_id = Column(Integer, primary_key=True)
    project = Column(String(255))

    # specifications
    # the combination of project has to be unique
    __table_args__ = (UniqueConstraint('project', name='_project_info'),)

    # relationships:
    # the 1:n relation from one project and multiple jobs
    jobs = relationship(
        "Job",                       # this relation points to Job class
        back_populates="project",               # relation name in job table
        cascade="all, delete-orphan"            # related jobs should be deleted, by de-associating
    )

    # representation string for project objects
    def __repr__(self):
        return f"Project(project_id={self.project_id!r}, project={self.project!r})"


class User(Base):
    # the name of the users table in the DB
    __tablename__ = "users"

    __doc__ = get_sqlalchemy_class_doc(
        tablename=__tablename__,
        classname=__name__,
        user_id=["primary key, automatic"],
        username=["string, the username", "testuser"]
    )

    # the columns of user table
    user_id = Column(Integer, primary_key=True)
    username = Column(String(20), nullable=False)

    # specifications
    # the username has to be unique
    __table_args__ = (UniqueConstraint('username', name='_usersname'),)

    # relationships:
    # the 1:n relation between the users and the jobs
    jobs = relationship(
        "Job",                          # this relation points to Job class
        back_populates="user",                  # relation name in job table
        cascade="all, delete-orphan"            # related jobs should be deleted, by de-associating
    )

    # representation string for user objects
    def __repr__(self):
        return f"User(user_id={self.user_id!r}, username={self.username!r})"


class JobStatus(Base):
    # the name of the status table in the DB
    __tablename__ = "status"

    __doc__ = get_sqlalchemy_class_doc(
        tablename=__tablename__,
        classname=__name__,
        status_id=["primary key, automatic"],
        status_name=["string, the status of jobs", "teststatus"]
    )

    # the columns of status table
    status_id = Column(Integer, primary_key=True)
    status_name = Column(String(15), nullable=False)

    # specifications
    # the name of a status has to be unique
    __table_args__ = (UniqueConstraint('status_name', name='_status_name'),)

    # relationships:
    # the 1:n relation between the status and the jobs
    jobs = relationship(
        "Job",                          # this relation points to Job class
        back_populates="jobstatus"               # relation name in job table
    )

    # representation string for status objects
    def __repr__(self):
        return f"JobStatus(status_id={self.status_id!r}, status_name={self.status_name!r})"


class JobType(Base):
    # the name of the jobtype table in the DB
    __tablename__ = "jobtype"

    __doc__ = get_sqlalchemy_class_doc(
        tablename=__tablename__,
        classname=__name__,
        jobtype_id=["primary key, automatic"],
        jobtype_name=["string, the name of the jobtype", "testjobtype"],
        typeversion=["string, the version of jobtype", "0.1.0"]
    )

    # the columns of jobtype table
    jobtype_id = Column(Integer, primary_key=True)
    jobtype_name = Column(String(25), nullable=False)
    typeversion = Column(String(35))

    # speciifications
    # the combination of the jobtype and typeversion
    __table_args__ = (UniqueConstraint('jobtype_name', 'typeversion', name='_jobtype_info'),)

    # relationships:
    # the 1:n relation from one jobtype to multiple jobs
    jobs = relationship(
        "Job",                          # this relation points to Job class
        back_populates="jobtype"                # relation name in job table
    )

    # representation string for jobtype objects
    def __repr__(self):
        return f"JobType(jobtype_id={self.jobtype_id!r}, jobtype_name={self.jobtype_name!r}, typeversion={self.typeversion!r})"


class Queue(Base):
    # the name of the queues table in the DB
    __tablename__ = "queues"

    __doc__ = get_sqlalchemy_class_doc(
        tablename=__tablename__,
        classname=__name__,
        queue_id=["primary key, automatic"],
        queue_name=["string, the queue name", "testqueue"]
    )

    # the columns of queues table
    queue_id = Column(Integer, primary_key=True)
    queue_name = Column(String(20), nullable=False)

    # specifications
    # the name of the queue has to be unique
    __table_args__ = (UniqueConstraint('queue_name', name='_queue_information'),)

    # relations:
    # the 1:n relation from one queue to multiple jobs
    jobs = relationship(
        "Job",                      # this relation points to Job class
        back_populates="queue"             # relation name in job table
    )

    # representation string for queue objects
    def __repr__(self):
        return f"Queue(queue_id={self.queue_id!r}, queue_name={self.queue_name!r})"


class Host(Base):
    # the name of the hosts table in the DB
    __tablename__ = "host"

    __doc__ = get_sqlalchemy_class_doc(
        tablename=__tablename__,
        classname=__name__,
        host_id=["primary key, automatic"],
        hostname=["string, the hostname", "testhost"]
    )

    # the columns of host table
    host_id = Column(Integer, primary_key=True)
    host_name = Column(String(20), nullable=False)

    # specifications
    # the hostname for submitting hosts has to be unique
    __table_args__ = (UniqueConstraint('host_name', name='_host_information'),)

    # relationships:
    # the 1:n relation from one submitting host to multiple jobs
    jobs = relationship(
        "Job",                      # this relation points to Job class
        back_populates="submitting_host"    # relation name in job table
    )

    # representation string for host objects
    def __repr__(self):
        return f"Host(host_id={self.host_id!r}, host_name={self.host_name!r})"


class MetadataInfo(Base):
    # the name of the metadata table in the DB
    __tablename__ = "metadata_info"

    __doc__ = get_sqlalchemy_class_doc(
        tablename=__tablename__,
        classname=__name__,
        metadata_id=["primary key, automatic"],
        name=["string, the name of the metadata", "metadata_test"],
        data=["string, the metadata", "test_metadata"]
    )

    # the columns of metadata table
    metadata_id = Column(Integer, primary_key=True)
    name = Column(String(30), nullable=False)
    data = Column(String, nullable=False)

    # specifications
    # the combination of the name and information of metadata has to be unique
    __table_args__ = (UniqueConstraint("name", "data", name="_metadata_info"),)

    # relations:
    # the m:n relation from multiple metadata information and multiple jobs
    jobs = relationship(
        "Job",                  # relation points to Job class
        secondary="job_metadata",       # name of secondary relation table
        back_populates="metadata_infos"       # relation name in job table
    )

    # representation string for metadata objects
    def __repr__(self):
        return f"""MetadataInfo(metadata_id={self.metadata_id!r}, name={self.name!r}, data={self.data!r})"""

# table for secondary relation
# connects job table and metadata table
job_metadata_table = Table(
    "job_metadata",     # name of the relations table
    Base.metadata,
    # set the combination of job_id and metadata_id as primary key (unique)
    Column("job_id", ForeignKey("job.job_id"), primary_key=True),
    Column("metadata_id", ForeignKey("metadata_info.metadata_id"), primary_key=True)
)


class Job(Base):
    # the name of the jobs table in the DB
    __tablename__ = "job"

    __doc__ = get_sqlalchemy_class_doc(
        tablename=__tablename__,
        classname=__name__,
        job_id=["primary key, automatic"],
        job_name=["string, the job name", "'/testjob'"],
        subjob=["string, the path to the project", "'/home/my-user'"],
        cores=["integer, the number of used cores", "20"],
        timestart=["timestamp, time when the job started", "'12:00'"],
        timestop=["timestamp, time when the job was finished", "'00:00'"],
        totalworkloadtime=["integer, seconds the job needed to be finished", "43200"],
        project=["foreign key, refers to the related project", "testjob.project.append(testproject)"],
        user=["foreign key, refers to the related user", "testjob.users.append(testuser)"],
        status=["foreign key, refers to the related status", "testjob.status.append(teststatus)"],
        jobtype=["foreign key, refers to the related jobtype", "testjob.jobtype.append(test_jobtype)"],
        queues=["foreign key, refers to the related queue", "testjob.queues.append(testqueue)"],
        submitting_host=["foreign key, refers to the related host", "testjob.host.append(testhost)"]#,
        # master=["foreign key, refers to the related master job", "testjob.masters.append(testmaster_job)"],
        # parent=["foreign key, refers to the related parent job", "testjob.parent.append(testparent_job)"]
    )

    # the columns of the job table
    job_id = Column(Integer, primary_key=True)
    job_name = Column(String(50), nullable=False)
    subjob = Column(String(51), nullable=False)
    servers = Column(String)
    cores = Column(Integer)
    timestart = Column(TIMESTAMP)
    timestop = Column(TIMESTAMP)
    totalworkloadtime = Column(Integer)
    project_id = Column(ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    status_id = Column(ForeignKey("status.status_id"), nullable=False)
    jobtype_id = Column(ForeignKey("jobtype.jobtype_id", ondelete="CASCADE"))
    queue_id = Column(ForeignKey("queues.queue_id", ondelete="CASCADE"))
    submitting_host_id = Column(ForeignKey("host.host_id", ondelete="CASCADE"), nullable=False)
    # master_id = Column(BigInteger, ForeignKey("job.job_id"))
    # parent_id = Column(BigInteger, ForeignKey("job.job_id"))

    # specifications
    # the combination of job and username has to be unique
    __table_args__ = (UniqueConstraint("job_name", "user_id", "project_id", name="_job_unique_info"),)

    # relationships:
    # the n:1 relation from multiple jobs to one project
    project = relationship(
        "Project",              # this relation points to Project class
        back_populates="jobs"           # relation name in projects table
    )

    # the n:1 relation from multiple jobs to one user
    user = relationship(
        "User",                 # this relation points to User class
        back_populates="jobs"           # relation name in user table
    )

    # the n:1 relation from multiple jobs to one status
    jobstatus = relationship(
        "JobStatus",               # this relation points to JobStatus class
        back_populates="jobs"           # relation name in status table
    )

    # the n:1 relation from multiple jobs to one jobtype
    jobtype = relationship(
        "JobType",              # this relation points to JobType class
        back_populates="jobs"           # relation name in jobtype table
    )

    # the n:1 relation from multiple jobs to one queue
    queue = relationship(
        "Queue",                # this relation points to Queue class
        back_populates="jobs"           # relation name in queue table
    )

    # the n:1 relation from multiple jobs to one submitting_host
    submitting_host = relationship(
        "Host",                 # this relation points to Host class
        back_populates="jobs"           # relation name in host table
    )

    # the m:n relation from multiple jobs to multiple jobs
    metadata_infos = relationship(
        "MetadataInfo",             # class name, where this relation points to
        secondary="job_metadata",       # connection via job_metadata table
        back_populates="jobs",          # relation name, in metadata class
    )

    ##  ToDo: master and parent relationships overlay each other
    # # the m:n relation from multiple jobs to one superior master job
    # master = relationship(
    #     "Job",               # class name, where this relation points to
    #     remote_side=[job_id]            # primary key column, for self referencing table
    # )
    #
    # # the m:n relation from multiple jobs to one superior parent job
    # parent = relationship(
    #     "Job",               # class name, where this relation points to
    #     remote_side=[job_id]            # primary key column, for self referencing table
    # )

    # the representation string for job objects
    def __repr__(self):
        return f"""Job(job_id={self.job_id!r}, job_name={self.job_name!r},
        subjob={self.subjob!r},
        servers={self.servers!r}
        cores={self.cores!r},
        timestart={self.timestart!r},
        timestop={self.timestop!r},
        totalworkloadtime={self.totalworkloadtime!r},
        project_id={self.project_id!r},
        status_id={self.status_id!r},
        jobtype_id={self.jobtype_id!r},
        host_id={self.submitting_host_id!r},
        queue_id={self.queue_id!r},
        user_id={self.user_id!r})"""


# TODO: this will be pyiron_base.IsDatabase
class GenericDatabase(abc.ABC):
    """
    Defines the abstract database interface used by all databases.
    """

    @abc.abstractmethod
    def add_jobentry(self, entry: DatabaseEntry) -> int:
        pass

    @abc.abstractmethod
    def get_jobentry(self, job_id: int) -> DatabaseEntry:
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
    def get_jobentry_id(self, job_name: str, project_id: int) -> Optional[int]:
        pass

    @abc.abstractmethod
    def get_project_id(self, location: str) -> Optional[int]:
        pass

    @abc.abstractmethod
    def update_status(self, job_id, status):
        pass

    @abc.abstractmethod
    def remove_jobentry(self, job_id: int) -> DatabaseEntry:
        pass

    @abc.abstractmethod
    def remove_project(self, project_id: int):
        pass

    @abc.abstractmethod
    def job_table(self) -> pd.DataFrame:
        pass


class TinyDB(GenericDatabase):
    """
    Complete database implementation and "reference".  Exists to allow easy testing the new database interface without
    messing with DatabaseAccess.
    """

    def __init__(self, path, echo=False):

        self.job_essentials = {
            "job",
            "subjob",
            "project",
            "user",
            "jobstatus",
            "submitting_host"
        }

        self.functions_dict = {
            "job": self.get_job,
            "project": self.get_project,
            "user": self.get_user,
            "jobstatus": self.get_status,
            "jobtype": self.get_jobtype,
            "queue": self.get_queue,
            "submitting_host": self.get_host
        }

        self.sideinfos = {
            "typeversion",
            "subjob",
            "servers",
            "cores",
            "timestart",
            "timestop",
            "totalworkloadtime"
        }

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

    def add_jobentry(self, par_dict):
        # ToDo: check if self._view_mode is turned on?

        # check if all essential information are available
        for attribute in self.job_essentials:
            if attribute not in par_dict.keys():
                raise ValueError(
                    f"expected dictionary with minimal keys {self.job_essentials}, but got {par_dict.keys()}."
                )
            if attribute == "jobtype":
                if "typeversion" not in par_dict.keys():
                    raise KeyError('missing "typeversion" in job information')

        # ToDo: check for chemicalformula length?

        with Session(self.engine) as session:
            # get the job object from database or new initialized
            job = self.get_job(par_dict)

            # loop over the attributes in the given information item
            for attribute in par_dict.keys():
                # if the attribute has a related function, this will be called
                # and return the related object
                if attribute in self.functions_dict:

                    item = self.functions_dict[attribute](par_dict)
                    setattr(job, attribute, item)  # job.attribute = item
                    # job.__dict__[attribute] = item      # assign the related object to th job
                elif attribute in self.sideinfos:
                    continue
                else:
                    self.set_metadata(attribute, par_dict[attribute])
                    item = self.get_metadata(attribute, par_dict[attribute])
                    job.metadata_infos.append(item)

            session.add(job)
            session.commit()  # submit the dataset to database
        return job

# ToDo: check functionality
    def update_status(self, job_id, status):
        with Session(self.engine) as session:
            try:
                s = (
                    session.query(JobStatus)
                    .select_from(Job)
                    .where(Job.job_id == job_id, JobStatus.status_id == Job.status_id)
                    .one()
                )
                s.status = status
                session.commit()
            except Exception as e:
                raise ValueError(f"job_id {job_id} doesn't exist: {e}")

## ToDo: check functionality
    def _row_to_entry(self, job_data):
        return DatabaseEntry(
            job=job_data.job,
            subjob=job_data.subjob,
            servers=job_data.servers,
            cores=job_data.cores,
            timestart=job_data.timestart,
            timestop=job_data.timestop,
            totalworloadtime=job_data.totalworktime,
            project=job_data.project,
            user=job_data.user,
            jobstatus=job_data.jobstatus,
            jobtype=job_data.jobtype,
            typeversion=job_data.typeversion,
            submitting_host=job_data.submitting_host,
            queue=job_data.queue
        )

## ToDo: check functionality
    def get_jobentry(self, job_id: int) -> DatabaseEntry:
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
                        Job.__table__, Project.project, JobStatus.status_name, JobType.jobtype_name
                    )
                    .select_from(Job)
                    .join(Project)
                    .join(JobStatus)
                    .join(JobType)
                    .where(Job.job_id == job_id)
                    .one()
                )
                return self._row_to_entry(job_data)
        except NoResultFound:
            raise ValueError(f"No job with id {job_id} found!") from None

## ToDo: check functionality
    def get_jobentry_id(self, job_name: str, project_id: int) -> Optional[int]:
        with Session(self.engine) as session:
            try:
                return (
                    session.query(Job.id)
                    .where(
                        Job.job == job_name,
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
                    .where(Project.project == location)
                    .one()
                    .id
                )
            # FIXME: MultipleResultsFound should be reraised because it indicates a broken database
            except (MultipleResultsFound, NoResultFound):
                return None

## ToDo: check functionality
    def remove_jobentry(self, job_id: int) -> DatabaseEntry:
        with Session(self.engine) as session:
            # look for the job with the respective id
            entry = session.execute(
                select(Job).
                where(Job.job_id == job_id)
            ).scalar()
            if entry:
                # delete the found entry
                session.delete(entry)
                session.commit()
                return entry
            # ToDo: what should be returned, in case the job was not found?

## ToDo: check functionality
    def job_table(self) -> pd.DataFrame:
        with Session(self.engine) as session:
            query = (
                session.query(
                    Job.__table__, Project.project, JobStatus.status_name, JobType.jobtype_name
                )
                .select_from(Job)
                .join(Project)
                .join(JobStatus)
                .join(JobType)
            )
            return pd.DataFrame([r._asdict() for r in query.all()])

    def get_project(self, item_dict):
        """
        checks in database for project or creates new project object with the appropriate project data
        :param item_dict: dictionary with all job data, contains also the project
        :return: project object, with the respective project data
        """
        with Session(self.engine) as session:
            # check for available projects with this specifications in database
            check_project = (
                select(Project).
                where(Project.project == item_dict["project"])
            )
            project_list = session.execute(check_project).scalars().all()

        # if project exists: load the project object
        if project_list:
            project = project_list[0]
        # else: initialize it
        else:
            project = Project(
                project=item_dict["project"],
            )

        return project

    def get_user(self, item_dict):
        """
        checks in database for username or creates new user object with the respective username
        :param item_dict: dictionary with all job data, contains also, contains also the username
        :return: user object, with the respective username
        """
        with Session(self.engine) as session:
            # check for the user with this username
            check_user = (
                select(User).
                where(User.username == item_dict["user"])
            )
            user_list = session.execute(check_user).scalars().all()

        # if user exists: load the user
        if user_list:
            user = user_list[0]
        # else: initialize it
        else:
            user = User(
                username=item_dict["user"]
            )

        return user

    def get_status(self, item_dict):
        """
        checks in database for status with the appropriate data
        :param item_dict: dictionary with all job data, contains also the status
        :return: status object, with the respective status name
        """
        with Session(self.engine) as session:
            # check for available status information
            check_status = (
                select(JobStatus).
                where(JobStatus.status_name == item_dict["jobstatus"])
            )
            status_list = session.execute(check_status).scalars().all()
        # if status information not in status table: throw ValueError
        if not status_list:
            raise ValueError(f"status {item_dict['jobstatus']} is unknown and does not exist in the database.")
        # else: return the value found
        else:
            return status_list[0]

    def get_jobtype(self, item_dict):
        """
        checks in database for jobtype or creates new jobtype object with the respective data
        :param item_dict: dictionary with all job data, contains also the jobtype and typeversion
        :return: jobtype object, with the respective jobtype data
        """
        with Session(self.engine) as session:
            # check for available jobtype with this specifications in database
            check_jobtype = (
                select(JobType).
                where(JobType.jobtype_name == item_dict["jobtype"]).
                where(JobType.typeversion == item_dict["typeversion"])
            )
            jobtype_list = session.execute(check_jobtype).scalars().all()

        # if jobtype exists: load the jobtype
        if jobtype_list:
            jobtype = jobtype_list[0]
        # else: initialize it
        else:
            jobtype = JobType(
                jobtype_name=item_dict["jobtype"],
                typeversion=item_dict["typeversion"]
            )

        return jobtype

    def get_queue(self, item_dict):
        """
        checks in database for a dataset or creates new queue object with the appropriate data
        :param item_dict: dictionary with all job data, contains also the name of the queue
        :return: queue object, with the respective name of the queue
        """
        with Session(self.engine) as session:
            # check for available queue with this specifications in database
            check_queue = (
                select(Queue).
                where(Queue.queue_name == item_dict["queue"])
            )
            queue_list = session.execute(check_queue).scalars().all()

        # if queue exists: load the queue object
        if queue_list:
            que = queue_list[0]
        # else: initialize it
        else:
            que = Queue(
                queue_name=item_dict["queue"]
            )

        return que

    def get_host(self, item_dict):
        """
        checks in database for a dataset with the appropriate hostname or creates new host object
        :param item_dict: dictionary with all job data, contains also the hostname
        :return: host object, with the respective hostname
        """
        with Session(self.engine) as session:
            # check for available host with this hostname in database
            check_host = (
                select(Host).
                where(Host.host_name == item_dict["submitting_host"])
            )
            host_list = session.execute(check_host).scalars().all()

        # if host exists: load the host object
        if host_list:
            host = host_list[0]
        # else: initialize it
        else:
            host = Host(
                host_name=item_dict["submitting_host"]
            )

        return host

    def get_job(self, item_dict):
        """
        checks for a job in database or creates new job object with appropriate job data
        :param item_dict: dictionary with all job data
        :return: job object, with the respective job data
        """
        # a list of additional information of a job, which can be null, by initializing the job object
        additionals = ["servers", "cores", "timestart", "timestop", "totalworkloadtime"]

        with Session(self.engine) as session:
            # check for available jobs with this specifications in database
            check_job = (
                select(Job).
                join(Job.user).
                join(Job.project).
                where(Job.job_name == item_dict["job"]).
                where(Job.subjob == item_dict["subjob"]).
                where(User.username == item_dict["user"]).
                where(Project.project == item_dict["project"])
            )
            found_job = session.execute(check_job).scalars().all()

            # if job exists: load the job object
            if found_job:
                job = found_job[0]
            # else: initialize it
            else:
                job = Job(
                    job_name=item_dict["job"],
                    subjob=item_dict["subjob"],
                )
            # and for every attribute of job object that differs from the new information,
            # add the new information to the job object
            for attribute in additionals:
                if attribute in item_dict and getattr(job, attribute) != item_dict[attribute]:
                    setattr(job, attribute, item_dict[attribute])

            return job

    def set_metadata(self, metadata_name, metadata_info):
        """
        checks in database for metadata dataset or creates a new metadata object with appropriate data
        :param metadata_name: string with the information which kind of metadata it is
        :param metadata_info: any kind of data type, containing the metadata
        :return: None
        """
        with Session(self.engine) as session:
            # check for available metadata datasets with this specifications in database
            check_metadata = (
                select(MetadataInfo).
                where(MetadataInfo.name == metadata_name).
                where(MetadataInfo.data == json.dumps(metadata_info))
            )
            metadata_list = session.execute(check_metadata).all()

            # if metadata exists: load the metadata object
            if not metadata_list:
                metadata_obj = MetadataInfo(
                    name=metadata_name,
                    data=json.dumps(metadata_info)
                )
                # for m:n relation between jobs and metadata,
                # the metadata dataset must be available before the job is submitted to the database
                session.add(metadata_obj)
                session.commit()

    def get_metadata(self, metadata_name, metadata_info):
        """
        checks in database for metadata dataset or creates a new metadata object with appropriate data
        :param metadata_name: string with the information which kind of metadata it is
        :param metadata_info: any kind of data type, containing the metadata
        :return: metadata object, with respective metadata
        """
        with Session(self.engine) as session:
            # check for available metadata datasets with this specifications in database
            check_metadata = (
                select(MetadataInfo).
                where(MetadataInfo.name == metadata_name).
                where(MetadataInfo.data == json.dumps(metadata_info))
            )
            metadata_obj = session.execute(check_metadata).scalars().all()

        return metadata_obj[0]
