import boto3
from botocore.client import Config
import os
import posixpath
import importlib
import fnmatch
import json


class S3ioConnect:
    def __init__(self, config):
        """
            Establishes connection to a specific 'bucket' of a S3 type object store.

            Args:
                config (str/dict):  if str: path to a json configuration file with login credentials for the bucket.
                                    if dict: dictionary containing the login credentials.

            The configuration needs to provide the following information:
                {
                access_key : ""
                secret_key : ""
                endpoint : ""
                bucket : ""
                }
        """

        if isinstance(config, str):
            with open(config) as json_file:
                config = json.load(json_file)

        self.s3resource = boto3.resource('s3',
                                    config=Config(s3={'addressing_style': 'path'}),
                                    aws_access_key_id=config['access_key'],
                                    aws_secret_access_key=config['secret_key'],
                                    endpoint_url=config['endpoint']
                                    )
        self.bucket_name = config["bucket"]
        self.bucket = self.s3resource.Bucket(self.bucket_name)


class FileS3IO:
    def __init__(self, config, path='/'):
        """
            Establishes connection to a specific 'bucket' of a S3 type object store.

            Args:
                config (str/dict/:class:`S3IO_connect`): Provides access information for the S3 type object store:
                        str: path to a json configuration file with login credentials for the bucket.
                        dict: dictionary containing the login credentials.
                        S3IO_connect: Instantiated S3IO_connect class to access the S3 system.

                path (str): Initial group in the bucket which is opened.

            The configuration needs to provide the following information:
                {
                access_key : ""
                secret_key : ""
                endpoint : ""
                bucket : ""
                }
        """
        self.history = [path]
        if isinstance(config, S3ioConnect):
            self._s3io = config
        else:
            self._s3io = S3ioConnect(config=config)

        self._bucket = self._s3io.bucket
        self._s3_path = None
        self.s3_path = path

    @property
    def s3_path(self):
        """
        Get the path in the S3 object store starting from the root group - meaning this path starts with '/'

        Returns:
            str: S3 path
        """
        return self._s3_path

    @s3_path.setter
    def s3_path(self, path):
        """
        Set the path in the S3 object store starting from the root group

        Args:
            path (str): S3 path
        """
        if (path is None) or (path == ""):
            path = "/"
        self._s3_path = posixpath.normpath(path)
        if not posixpath.isabs(self._s3_path):
            self._s3_path = "/" + self._s3_path
        if not self._s3_path[-1] == '/':
            self._s3_path = path + '/'

    @property
    def _bucket_path(self):
        """
        The bucket object internally does not use a '/' to indicate the root group.

        Return:
            str: Internal path in the bucket.
        """
        return self._s3_path[1:]

    def print_bucket_info(self):
        """ Print name of the associated bucket. """
        print('Bucket name: {}'.format(self._bucket.name))

    def list_groups(self):
        """
        List directories/groups in the current group.

        Returns:
            list: list of directory names.
        """
        groups = []
        group_path_len = len(self._bucket_path.split('/')) - 1
        for obj in self._list_objects():
            rel_obj_path_spl = obj.key.split('/')[group_path_len:]
            if len(rel_obj_path_spl) > 1:
                if rel_obj_path_spl[0] not in groups:
                    groups.append(rel_obj_path_spl[0])
        return groups

    def list_nodes(self):
        """
        List of 'files' ( string not followed by '/' ) in the current group.

        Returns:
            list: list of file names.
        """
        nodes = []
        group_path_len = len(self._bucket_path.split('/')) - 1
        for obj in self._list_objects():
            rel_obj_path_spl = obj.key.split('/')[group_path_len:]
            if len(rel_obj_path_spl) == 1:
                nodes.append(rel_obj_path_spl[0])
        return nodes

    def list_all(self):
        """
        Combination of list_groups() and list_nodes() in one dictionary with the corresponding keys:
        - 'groups': Sub-folder/ -groups.
        - 'nodes': Files in the current group.

        Returns:
            dict: dictionary with all items in the group.
        """
        return {
            "groups": self.list_groups(),
            "nodes": self.list_nodes(),
        }

    def _to_abs_bucketpath(self, path):
        """Helper function to convert a given path to an absolute path inside the S3 bucket."""
        if path is None or "":
            path = self._bucket_path
        if posixpath.isabs(path):
            path = path[1:]
        else:
            path = self._bucket_path + path
        return path

    def is_dir(self, path):
        """
        Check if given path is a directory.

        Args:
            path (str): path to check.

        Returns:
            bool: True if path is a directory.
        """
        path = self._to_abs_bucketpath(path)
        if len(path) > 1 and path[-1] != '/':
            path = path + '/'
        for obj in self._list_all_files_of_bucket():
            if path == obj.key[:len(path)]:
                return True

    def is_file(self, path):
        """
        Check if given path is a file.

        Args:
            path (str): path to check.

        Returns:
            bool: True if path is a file.
        """
        l = []
        for obj in self._list_all_files_of_bucket():
            l.append('/' + obj.key)
        if path in l:
            return True
        if self._bucket_path + path in l:
            return True

    def open(self, group):
        """
        Opens the provided group (create group if not yet present).

        Args:
            group (str): group to open/create.
        """
        new = self.copy()
        new.s3_path = self.s3_path + group
        new.history.append(new.s3_path)
        return new

    def copy(self):
        """
        Copy the Python object which links to the S3 object store.

        Returns:
            FileS3IO: New FileS3io object pointing to the same S3 object store
        """
        new = FileS3IO(config=self._s3io, path=self.s3_path)
        return new

    def close(self):
        """   Close current group and open previous group.         """
        if len(self.history) > 1:
            del self.history[-1]
        elif len(self.history) == 1:
            self.history[0] = "/"
        else:
            print("Err: no history")
        self._s3_path = self.history[-1]

    def upload(self, files, metadata=None):
        """
        Uploads files into the current group of the S3 object store.

        Arguments:
            files (list) : List of filenames to upload
            metadata (dictionary): metadata of the files (Not nested, only "str" type)
        """
        if metadata is None:
            metadata = {}

        for file in files:
            [path, filename] = os.path.split(file)

            #def printBytes(x):
            #    print('{} {}/{} bytes'.format(filename, x, s))
            #s = os.path.getsize(file)
            # Upload file accepts extra_args: Dictionary with predefined keys. One key is Metadata
            self._bucket.upload_file(
                file,
                self._bucket_path + filename,
                {"Metadata": metadata}
            )

    def download(self, files, targetpath="."):
        """
        Download files from current group to local file system (current directory is default)

        Arguments:
            files (list): List of filenames in the S3 object store.
            targetpath (str): Path in the local data system, to which the files should be downloaded.
        """
        if not os.path.exists(targetpath):
            os.mkdir(targetpath)
        for f in files:
            filepath = os.path.join(targetpath, f.split("/")[-1])
            print(filepath)
            self._bucket.download_file(self._bucket_path + f, filepath)

    def get_metadata(self, file):
        """
        Returns the metadata of a file.

        Args:
            file (str): path to a file of the bucket.
        Returns:
             dict: metadata field associated with the file.
        """
        file = self._to_abs_bucketpath(file)
        return self._bucket.Object(file).metadata

    def _s3io_object(self, file):
        """
        Returns an object with access to the S3 object store which can be downloaded via .get()

        Args:
            file (str): path to a file of the bucket.
        Returns:
        """
        file = self._to_abs_bucketpath(file)
        s3object = self._bucket.Object(file)
        return s3object

    def get(self, file):
        """
        Returns a s3.Object containing the requested file.

        Args:
            file(str): a path like string.
        Returns:
             Object containing a file.
        """
        file = self._to_abs_bucketpath(file)
        return self._bucket.Object(file).get()

    def put(self, data_obj, path=None, metadata=None):
        """
            Upload a data_obj to the current group/ the provided path.

            Args:
                data_obj(:class:`pyiron_contrib.generic.data.Data`): data object to upload the data from.
                path(str/None):
                metadata(dict/None): metadata to be used (has to be a dictionary of type {"string": "string, }).
                      Provided metadata overwrites the one possibly present in the data object.
        """
        if self.is_dir(path):
            path = self._to_abs_bucketpath(path)
        else:
            raise ValueError("No valid path specified!")
        if path[-1] != '/':
            path = path + '/'
        path = path + data_obj.filename

        data = data_obj.data()
        if metadata is None:
            metadata = data_obj.metadata
        if metadata is None:
            raise ValueError
        if not isinstance(path, str):
            raise ValueError
        if data is None:
            raise ValueError
        self._bucket.put_object(Key=path, Body=data, Metadata=metadata)


    def _list_objects(self):
        l = []
        for obj in self._bucket.objects.filter(Prefix=self._bucket_path):
            l.append(obj)
        return l

    def print_fileinfos(self):
        """
            Prints the filename, last modified date and size for all files in the current group,
            recursively including sub groups.
        """
        for obj in self._bucket.objects.filter(Prefix=self._bucket_path):
            print('/{} {} {} bytes'.format(obj.key, obj.last_modified, obj.size))

    def _list_all_files_of_bucket(self):
        l = []
        for obj in self._bucket.objects.all():
            l.append(obj)
        return l

    def glob(self, path):
        """
            Return a list of paths matching a pathname pattern.
            The pattern may contain simple shell-style wildcards a la fnmatch.

            Args:
                path(str): a path like string which may contain shell-style wildcards.

            Return:
                list: List of files matching the provided path.
        """
        path = self._to_abs_bucketpath(path)
        l = []
        for obj in self._bucket.objects.filter(Prefix=self._bucket_path):
            if fnmatch.fnmatchcase(obj.key, path):
                l.append(obj.key)
        return l

    @staticmethod
    def print_file_info(filelist):
        """
            Prints filename, last_modified, and size of each file in the provided list of file objects.

            Args:
                filelist (list): List containing objects from a bucket.
        """
        for obj in filelist:
            print('/{} {} {} bytes'.format(obj.key, obj.last_modified, obj.size))

    def remove_file(self, file):
        """
            Deletes the object associated with a file.

            Args:
                file (str/None): path like string to the file to be removed.
        """
        if not self.is_file(file):
            raise ValueError("{} is not a file.".format(file))
        file = self._to_abs_bucketpath(file)
        self._bucket.Object(file).delete()
        #self._remove_object(prefix=file, debug=debug)

    def remove_group(self, path=None, debug=False):
        """
            Deletes the current group with all it's content recursively.

            Args:
                path (str/None): group to be removed recursively.
                debug(bool): If True, additional information is printed.
        """
        if path is None:
            path = self._s3_path
        if not self.is_dir(path):
            raise ValueError("{} is not a group.".format(path))
        path = self._to_abs_bucketpath(path)
        self._remove_object(prefix=path, debug=debug)

    def _remove_object(self, prefix, debug=False):
        """
            Deletes all objects matching the provided prefix.

            Args:
                prefix(str): All objects with this prefix will be removed.
                debug(bool): If True, additional information is printed.
        """
        if debug:
            print('\nDeleting all objects with sample prefix {}/{}.'.format(self._bucket.name, prefix))
        delete_responses = self._bucket.objects.filter(Prefix=prefix).delete()
        if debug:
            for delete_response in delete_responses:
                for deleted in delete_response['Deleted']:
                    print('\t Deleted: {}'.format(deleted['Key']))

    def __enter__(self):
        """ Compatibility function for the with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Compatibility function for the with statement."""
        self.close()

    def __repr__(self):
        """
        Human readable string representation.

        Return:
            str: list all nodes and groups as string.
        """
        return str(self.list_all())

    def __getitem__(self, item):
        """
        Get/ read (meta) data from the S3 object store

        Args:
            item (str, slice): path to the data or key of the data object

        Returns:
            dict/s3.Object:  meta data or data object
        """
        if isinstance(item, slice):
            raise NotImplementedError("Implement if needed, e.g. for [:]")
        else:
            item_lst = item.split("/")
            if len(item_lst) == 1 and item_lst[0] != "..":
                if item == "":
                    return self
                if item in self.list_nodes():
                    return self._s3io_object(item)
                if item in self.list_groups():
                    return self.open(item)
                raise ValueError("Unknown item: {}".format(item))
            else:
                item_abs_lst = (
                    os.path.normpath(os.path.join(self.s3_path, item))
                        .replace("\\", "/")
                        .split("/")
                )
                s3_object = self.copy()
                s3_object.s3_path = "/".join(item_abs_lst[:-1])
                return s3_object[item_abs_lst[-1]]


"""
Some information about the bucket object:

o.bucket= self.bucket   has the following   options:
--------------------------------------------------------------------------------------------------------------
o.bucket.Acl(                         o.bucket.Website(                     o.bucket.multipart_uploads
o.bucket.Cors(                        o.bucket.copy(                        o.bucket.name
o.bucket.Lifecycle(                   o.bucket.create(                      o.bucket.object_versions
o.bucket.LifecycleConfiguration(      o.bucket.creation_date                o.bucket.objects
o.bucket.Logging(                     o.bucket.delete(                      o.bucket.put_object(
o.bucket.Notification(                o.bucket.delete_objects(              o.bucket.upload_file(
o.bucket.Object(                      o.bucket.download_file(               o.bucket.upload_fileobj(
o.bucket.Policy(                      o.bucket.download_fileobj(            o.bucket.wait_until_exists(
o.bucket.RequestPayment(              o.bucket.get_available_subresources(  o.bucket.wait_until_not_exists(
o.bucket.Tagging(                     o.bucket.load(
o.bucket.Versioning(                  o.bucket.meta

o.bucket.objects   has the following  options:
--------------------------------------------------------------------------------------------------------------
o.bucket.objects.all(        o.bucket.objects.filter(     o.bucket.objects.limit(      o.bucket.objects.pages(
o.bucket.objects.delete(     o.bucket.objects.iterator(   o.bucket.objects.page_size(

o.bucket.Object  is an object of the object store. It is identified by a key, i.e. the full path + file name in the bucket.
Actually, there is no such thing as directories inside the bucket. '/' is a valid character in filenames and we use this fact 
to separate files into directories. 
obj=o.bucket.Object(/path/to/file)  has the following option:
--------------------------------------------------------------------------------------------------------------
obj.Acl(                           obj.download_fileobj(              obj.put(
obj.Bucket(                        obj.e_tag                          obj.reload(
obj.MultipartUpload(               obj.expiration                     obj.replication_status
obj.Version(                       obj.expires                        obj.request_charged
obj.accept_ranges                  obj.get(                           obj.restore
obj.bucket_name                    obj.get_available_subresources(    obj.restore_object(
obj.cache_control                  obj.initiate_multipart_upload(     obj.server_side_encryption
obj.content_disposition            obj.key                            obj.sse_customer_algorithm
obj.content_encoding               obj.last_modified                  obj.sse_customer_key_md5
obj.content_language               obj.load(                          obj.ssekms_key_id
obj.content_length                 obj.meta                           obj.storage_class
obj.content_type                   obj.metadata                       obj.upload_file(
obj.copy(                          obj.missing_meta                   obj.upload_fileobj(
obj.copy_from(                     obj.object_lock_legal_hold_status  obj.version_id
obj.delete(                        obj.object_lock_mode               obj.wait_until_exists(
obj.delete_marker                  obj.object_lock_retain_until_date  obj.wait_until_not_exists(
obj.download_file(                 obj.parts_count                    obj.website_redirect_location

with obj.get() one gets the object.
with obj.download_file('Filename') one downloads the associated file to 'Filename'

getobj=o.bucket.Object(object_key).get() has the following options:
--------------------------------------------------------------------------------------------------------------
getobj.clear(       getobj.fromkeys(    getobj.items(       getobj.pop(         getobj.setdefault(  getobj.values(
getobj.copy(        getobj.get(         getobj.keys(        getobj.popitem(     getobj.update(

"""

class ProjectS3IO(FileS3IO):
    """Class connecting the S3IO with the Project class."""
    def __init__(self, project, config, path='/'):
        self._project = project.copy()
        super().__init__(config=config, path=path)
        self._project.data_backend = "S3"

    @property
    def base_name(self):
        """
        The absolute path to of the current pyiron project - absolute path on the file system, not including the S3
        path.

        Returns:
            str: current project path
        """
        return self._project.path

    @property
    def db(self):
        """
        Get connection to the SQL database

        Returns:
            DatabaseAccess: database conncetion
        """
        return self._project.db

    @property
    def path(self):
        """
        Absolute path of the S3 group starting from the system root - combination of the absolute system path plus the
        absolute path inside the S3 object store starting from the root group.

        Returns:
            str: absolute path
        """
        return os.path.join(self._project.path, self._bucket_path).replace("\\", "/")

    @property
    def project(self):
        """
        Get the project instance the ProjectS3io object is located in

        Returns:
            Project: pyiron project
        """
        return self._project

    @property
    def project_path(self):
        """
        the relative path of the current project / folder starting from the root path
        of the pyiron user directory

        Returns:
            str: relative path of the current project / folder
        """
        return self._project.project_path

    @property
    def root_path(self):
        """
        the pyiron user directory, defined in the .pyiron configuration

        Returns:
            str: pyiron user directory of the current project
        """
        return self._project.root_path

    @property
    def working_directory(self):
        """
        Get the working directory of the current ProjectS3io object. The working directory equals the path but it is
        represented by the filesystem:
            /absolute/path/to/the/project/path/inside/the/s3/store

        Returns:
            str: absolute path to the working directory
        """
        return self.path

    @property
    def sql_query(self):
        """
        Get the SQL query for the project

        Returns:
            str: SQL query
        """
        return self._project.sql_query

    @sql_query.setter
    def sql_query(self, new_query):
        """
        Set the SQL query for the project

        Args:
            new_query (str): SQL query
        """
        self._project.sql_query = new_query

    @property
    def user(self):
        """
        Get current unix/linux/windows user who is running pyiron

        Returns:
            str: username
        """
        return self._project.user

    @property
    def _filter(self):
        """
        Get project filter

        Returns:
            str: project filter
        """
        return self._project._filter

    @_filter.setter
    def _filter(self, new_filter):
        """
        Set project filter

        Args:
            new_filter (str): project filter
        """
        self._project._filter = new_filter

    @property
    def _inspect_mode(self):
        """
        Check if inspect mode is activated

        Returns:
            bool: [True/False]
        """
        return self._project._inspect_mode

    @_inspect_mode.setter
    def _inspect_mode(self, read_mode):
        """
        Activate or deactivate inspect mode

        Args:
            read_mode (bool): [True/False]
        """
        self._project._inspect_mode = read_mode

    def copy(self):
        """
        Copy the ProjectS3IO object - copying just the Python object but maintaining the same pyiron path

        Returns:
            ProjectS3IO: copy of the ProjectS3IO object
        """
        new_s3 = ProjectS3IO(
            project=self._project, config=self._s3io, path=self.s3_path
        )
        new_s3._filter = self._filter
        return new_s3

    def import_class(self, class_name):
        """
        Import given class from fully qualified name and return class object.

        Args:
            class_name (str): fully qualified name of a pyiron class

        Returns:
            type: class object of the given name
        """
        internal_class_name = class_name.split(".")[-1][:-2]
        if internal_class_name in self._project.job_type.job_class_dict:
            module_path = self._project.job_type.job_class_dict[internal_class_name]
        else:
            class_path = class_name.split()[-1].split(".")[:-1]
            class_path[0] = class_path[0][1:]
            module_path = '.'.join(class_path)
        return getattr(
            importlib.import_module(module_path),
            internal_class_name,
        )

    #def to_object(self, class_name=None, **qwargs):

    def get_job_id(self, job_specifier):
        """
        get the job_id for job named job_name in the local project path from database

        Args:
            job_specifier (str, int): name of the job or job ID

        Returns:
            int: job ID of the job
        """
        return self._project.get_job_id(job_specifier=job_specifier)

    def inspect(self, job_specifier):
        """
        Inspect an existing pyiron object - most commonly a job - from the database

        Args:
            job_specifier (str, int): name of the job or job ID

        Returns:
            JobCore: Access to the HDF5 object - not a GenericJob object - use load() instead.
        """
        return self._project.inspect(job_specifier=job_specifier)

    def load(self, job_specifier, convert_to_object=True):
        """
        Load an existing pyiron object - most commonly a job - from the database

        Args:
            job_specifier (str, int): name of the job or job ID
            convert_to_object (bool): convert the object to an pyiron object or only access the HDF5 file - default=True
                                      accessing only the HDF5 file is about an order of magnitude faster, but only
                                      provides limited functionality. Compare the GenericJob object to JobCore object.

        Returns:
            GenericJob, JobCore: Either the full GenericJob object or just a reduced JobCore object
        """
        return self._project.load(
            job_specifier=job_specifier, convert_to_object=convert_to_object
        )

    def load_from_jobpath(self, job_id=None, db_entry=None, convert_to_object=True):
        """
        Internal function to load an existing job either based on the job ID or based on the database entry dictionary.

        Args:
            job_id (int): Job ID - optional, but either the job_id or the db_entry is required.
            db_entry (dict): database entry dictionary - optional, but either the job_id or the db_entry is required.
            convert_to_object (bool): convert the object to an pyiron object or only access the HDF5 file - default=True
                                      accessing only the HDF5 file is about an order of magnitude faster, but only
                                      provides limited functionality. Compare the GenericJob object to JobCore object.

        Returns:
            GenericJob, JobCore: Either the full GenericJob object or just a reduced JobCore object
        """
        return self._project.load_from_jobpath(
            job_id=job_id, db_entry=db_entry, convert_to_object=convert_to_object
        )

    def remove_job(self, job_specifier, _unprotect=False):
        """
        Remove a single job from the project based on its job_specifier - see also remove_jobs()

        Args:
            job_specifier (str, int): name of the job or job ID
            _unprotect (bool): [True/False] delete the job without validating the dependencies to other jobs
                               - default=False
        """
        self._project.remove_job(job_specifier=job_specifier, _unprotect=_unprotect)

    def __getitem__(self, item):
        """Check if the result is within the S3 store or at the project level and return corresponding object."""
        if isinstance(item, slice):
            raise NotImplementedError("Implement if needed, e.g. for [:]")
        else:
            item_abs_path = (
                os.path.relpath(os.path.join(self.path, item), self.project.path)
                    .replace("\\", "/")
            )
            if item_abs_path.split('/')[0] != "..":
                return super().__getitem__(item)
            else:
                return self._project[item_abs_path]

