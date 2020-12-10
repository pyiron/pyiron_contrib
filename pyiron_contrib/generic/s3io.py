import boto3
from botocore.client import Config
import os
import fnmatch
import json


class FileS3IO(object):
    def __init__(self, config_file=None, config_json=None, group=''):
        """
            Establishes connection to a specific 'bucket' of a S3 type object store.

            Args:
                config_file (str/None):  path to a json configuration file with login credentials for the bucket.
                config_json (dict/None): directly providing credentials as dict; overwrites config_file.
                group (str): Initial group in the bucket which is opened.

            The configuration needs to provide the following information:
                {
                access_key : ""
                secret_key : ""
                endpoint : ""
                bucket : ""
                }
        """
        self.history = []
        if config_json is not None:
            config = config_json
        else:
            if config_file is None:
                raise TypeError("config_json or config_file have to be provided (both None)")
            with open(config_file) as json_file:
                config = json.load(json_file)

        s3resource = boto3.resource('s3',
            config=Config(s3={'addressing_style': 'path'}),
            aws_access_key_id=config['access_key'],
            aws_secret_access_key=config['secret_key'],
            endpoint_url=config['endpoint']
        )
        bucket_name = config['bucket']
        # Now, the bucket object
        self._bucket = s3resource.Bucket(bucket_name)
        self._group = ""
        self.open(group)

    @property
    def group(self):
        return self._group

    def print_bucket_info(self):
        """ Print name of the associated bucket. """
        print('Bucket name: {}'.format(self._bucket.name))

    def list_groups(self):
        """
        List 'directories' ( string followed by '/' ) in the current group.

        Returns:
            list: list of directory names.
        """
        groups = []
        group_path_len = len(self._group.split('/')) - 1
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
        group_path_len = len(self._group.split('/')) - 1
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

    def is_dir(self, path):
        """
        Check if given path is a directory.

        Args:
            path (str): path to check.

        Returns:
            bool: True if path is a directory.
        """
        if len(path) > 1 and path[-1] != '/':
            path = path+'/'
        for obj in self._list_all_files_of_bucket():
            if path in obj.key:
                if self._group+path in obj.key:
                    return True
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
            l.append(obj.key)
        if path in l:
            return True
        if self._group+path in l:
            return True

    def open(self, group):
        """
        Opens the provided group (create group if not yet present).

        Args:
            group (str): group to open/create.
        """
        if len(group) == 0:
            self._group = group
        elif group[-1] == '/':
            self._group = group
        else:
            self._group = group + '/'
        self.history.append(self._group)

    def close(self):
        """   Close current group and open previous group.         """
        if len(self.history) > 1:
            del self.history[-1]
        elif len(self.history) == 1:
            self.history[0] = ""
        else:
            print("Err: no history")
        self._group = self.history[-1]

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
                self._group + filename,
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
            self._bucket.download_file(self._group + f, filepath)

    def get_metadata(self, file, abspath=False):
        """
        Returns the metadata of a file.

        Args:
            file:
        Returns:
             dict: metadata field associated with file.
        """
        if abspath:
            return self._bucket.Object(file).metadata
        else:
            return self._bucket.Object(self._group + file).metadata

    def get(self, file, abspath=False):
        """
        Returns a data object containing the requested file.

        Args:
            file(str): a path like string.
            abspath(bool): If True, the path is treated as absolute path in the S3 system.
        Returns:
             Object containing a file.
        """
        if abspath:
            return self._bucket.Object(file).get()
        else:
            return self._bucket.Object(self._group + file).get()

    def put(self, data_obj, metadata=None):
        """
            Upload a data_obj to the current group/ the provided path.

            Args:
                data_obj(:class:`pyiron_contrib.generic.data.Data`): data object to upload the data from.
                metadata(dict): metadata to be used (has to be a dictionary of type {"string": "string, }).
                      Provided metadata overwrites the one possibly present in the data object.
        """
        path = self._group + data_obj.filename
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
        for obj in self._bucket.objects.filter(Prefix=self._group):
            l.append(obj)
        return l

    def print_fileinfos(self):
        """
            Prints the filename, last modified date and size for all files in the current group,
            recursively including sub groups.
        """
        for obj in self._bucket.objects.filter(Prefix=self._group):
            print('{} {} {} bytes'.format(obj.key, obj.last_modified, obj.size))

    def _list_all_files_of_bucket(self):
        l = []
        for obj in self._bucket.objects.all():
            l.append(obj)
        return l

    def glob(self, path, relpath=False):
        """
            Return a list of paths matching a pathname pattern.
            The pattern may contain simple shell-style wildcards a la fnmatch.

            Args:
                path(str): a path like string which may contain shell-style wildcards.
                relpath(bool): If False, the path is treated as absolute path in the S3 system.

            Return:
                list: List of files matching the provided path.
        """
        if relpath and len(self._group) > 0:
            path = self._group + path
        l = []
        for obj in self._bucket.objects.filter(Prefix=self._group):
            if fnmatch.fnmatchcase(obj.key, path):
                l.append(obj.key)
        return l

    def print_file_info(self, filelist):
        """
            Prints filename, last_modified, and size of each file in the provided list of files
        """
        for obj in filelist:
            print('{} {} {} bytes'.format(obj.key, obj.last_modified, obj.size))

    def remove_file(self, file, abspath=False):
        """
            Deletes the object associated with a file.

            Args:
                file (str/None): path like string to the file to be removed.
                abspath(bool): If True, treat the path as absolute.
        """
        if not abspath:
            file = self._group + file
        if not self.is_file(file):
            raise ValueError("{} is not a file.".format(file))
        self._bucket.Object(file).delete()
        #self._remove_object(prefix=file, debug=debug)

    def remove_group(self, group=None, debug=False):
        """
            Deletes the current group with all it's content recursively.

            Args:
                group (str/None): group to be removed recursively.
                debug(bool): If True, additional information is printed.
        """
        if group is None:
            group = self._group
        if not self.is_dir(group):
            raise ValueError("{} is not a group.".format(group))
        self._remove_object(prefix=group, debug=debug)

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
