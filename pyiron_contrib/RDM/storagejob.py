import os
import posixpath
import shutil

from pyiron_base import GenericJob, DataContainer
from pyiron_base.storage.filedata import FileData
from pyiron_contrib.generic.s3io import FileS3IO


class StorageType:
    """
    The StorageType has to be instantiated with one of the available storage types.

    Attributes:
       storage_type(str): type of the storage, one of _available_storage_types
       local(bool): storage_type == 'local'
       s3(bool): storage_type == 's3'
       read_only (bool): read only StorageType cannot be changed

    """

    _available_storage_types = ["local", "s3"]

    def __init__(self, storage_type=None):
        """What type of storage is used

        Parameters:
            storage_type(str): one of ['local', 's3']
        """
        self._storage_type = None
        self._read_only = False
        self.storage_type = storage_type

    @property
    def read_only(self):
        """
        bool: if set, raise warning when attempts are made to modify the container
        """
        return self._read_only

    @read_only.setter
    def read_only(self, val):
        # can't mark a read-only list as writeable
        if self._read_only and not val:
            self._read_only_error()
        else:
            self._read_only = bool(val)

    @classmethod
    def _read_only_error(cls):
        raise RuntimeError(
            "The storage type cannot be changed when it is already in use."
        )

    @property
    def local(self):
        return self.storage_type == "local"

    @property
    def s3(self):
        return self.storage_type == "s3"

    @property
    def storage_type(self):
        return self._storage_type

    @storage_type.setter
    def storage_type(self, storage_type):
        if self.read_only:
            self._read_only_error()
        if storage_type not in self._available_storage_types:
            raise ValueError(
                f"Expected 'storage_type' to be one of {self._available_storage_types} "
                f"but got {storage_type}."
            )
        self._storage_type = storage_type


class StorageJob(GenericJob):
    """Job to store files associated with meta data either locally, or at a remote data service like s3."""

    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.server.run_mode.interactive = True
        self._stored_files = DataContainer(table_name="stored_files")
        self._storage_type = StorageType(storage_type="local")
        self._storage_type_store = DataContainer(table_name="storage_type")
        self._external_storage = None

    def _before_generic_remove_child(self):
        if self._storage_type.s3 and self._external_storage.is_dir(self.path):
            self._external_storage.remove_group(path=self.path)

    def check_setup(self):
        """
        This function is called in self._run_if_new() if self.server.run_mode.queue
        i.e. self.server.run_mode = queue if one sets a queue
        afterwards self.save() and self.run() is called again -> self.run_if_created  followed by (in this case always)
        self.run_if_scheduler
        run_if_scheduler is the function called to sent the job to the cluster, however, it already sends the
        stuff to the queue which is obviously not intended if this is used as ssh storage system.
        TODO:
        - overwrite run_if_scheduler to not trigger a job to be run on the queing system.
        - make some file_handler class to allow for 'store/receive/delete file' with a uniform interface for
            - local
            - s3
            - ssh
        """
        raise NotImplementedError(
            "Storing files remotely via ssh is not yet supported."
        )

    @property
    def storage_type(self):
        return self._storage_type.storage_type

    def use_s3_storage(self, config=None, bucket_name=None, _only_warn=False):
        external_storage = FileS3IO(
            config=config, path=self.path, bucket_name=bucket_name
        )
        if len(external_storage.list_nodes() + external_storage.list_groups()) > 0:
            if not _only_warn:
                raise ValueError("Storage location not empty.")
            else:
                self._logger.warning("Storage NOT empty - Danger of data loss!")
        self._external_storage = external_storage
        self._storage_type.storage_type = "s3"
        self._storage_type_store.create_group("s3_config")
        self._storage_type_store.s3_config.config = config
        self._storage_type_store.s3_config.bucket_info = (
            self._external_storage.connection_info
        )
        self._storage_type_store.s3_config.bucket_name = bucket_name

    def _list_groups(self):
        return []

    def _list_nodes(self):
        return self.files_stored

    def use_local_storage(self):
        self._storage_type.storage_type = "local"
        if "s3_config" in self._storage_type_store.keys():
            del self._storage_type_store.s3_config
        self._external_storage = None

    @property
    def files_stored(self):
        return [key for key in self._stored_files.keys()]

    def validate_ready_to_run(self):
        if self._storage_type.local:
            self._create_working_directory()
            self._storage_type.read_only = True
        elif self._storage_type.s3:
            self._storage_type.read_only = True
        else:
            raise NotImplementedError("No available Storage found.")

    def add_files(self, filenames, metadata=None, overwrite=False):
        """Add files to the storage

        Parameters:
            filenames(str/list of str): Files to save in the Storage; each file is stored under the same filename,
                beware name clashes if retrieved from different sources.
            metadata(dict/list of dict/None): metadata to attach to the files:
                if dict: apply this metadata dict to all files
                if list: has to be of same length as filenames, one metadata per file
                if None: Do not attach metadata (not recommended)
            overwrite(bool): if true to overwrite already present files
        """
        if self.status.initialized:
            self.run()
        if isinstance(filenames, str):
            filenames = [filenames]
        if isinstance(metadata, list) and len(metadata) != len(filenames):
            raise ValueError(
                f"Length of filenames and metadata have to match, "
                f"but got len(filenames)={len(filenames)} and len(metadata)={len(metadata)}."
            )

        files = [os.path.basename(file) for file in filenames]
        if len(set(files)) != len(files):
            raise ValueError(f"Resulting filenames {files} have duplicates.")

        for i, file in enumerate(filenames):
            if not overwrite and os.path.basename(file) in self._stored_files.keys():
                self._logger.warning(
                    f"{file} not copied, since already present and 'overwrite'=False."
                )
                continue
            _metadata = metadata[i] if isinstance(metadata, list) else metadata
            self._store_file(file, _metadata)

        self.run()

    def _store_file(self, file, metadata):
        try:
            if self._storage_type.local:
                shutil.copy(file, self.working_directory)
            elif self._storage_type.s3:
                self._external_storage.upload_file(file, metadata)
        except Exception as e:
            raise IOError(f"Storing {file} failed") from e
        else:
            self._stored_files[os.path.basename(file)] = metadata

    def _remove_file(self, file):
        try:
            if self._storage_type.local:
                os.remove(os.path.join(self.working_directory, file))
            elif self._storage_type.s3:
                self._external_storage.remove_file(file)
        except Exception as e:
            raise IOError(f"Removing {file} failed") from e
        else:
            del self._stored_files[file]

    def remove_files(self, filenames, dryrun=True, raise_error=True, silent=False):
        """Remove files in the storage

        Parameters:
            filenames(str/list of str):     Files to remove from the Storage
            dryrun(bool):                   If true, only report the files to be removed (default=True)
            raise_error(bool):              If true, missing files raise a FileNotFoundError (nothing gets removed),
                                            else remove all files found.
            silent(bool):                   If true, do not print the status message.
        """
        if isinstance(filenames, str):
            filenames = [filenames]

        files_not_found = [file for file in filenames if file not in self._stored_files]
        if len(files_not_found) > 0 and raise_error:
            raise FileNotFoundError(
                f"Files {files_not_found} not found, abort removing files. You may specify "
                f"raise_error=False to suppress this error."
            )

        files_to_remove = [file for file in filenames if file in self._stored_files]
        if dryrun and not silent:
            print(f"dryrun: to remove {files_to_remove} specify dryrun=False")
            return

        for file in files_to_remove:
            self._remove_file(file)

        if not silent:
            print(f"removed {files_to_remove}")
        self.run()

    def run_static(self):
        raise NotImplementedError("There is no static run mode for a Storage job.")

    def collect_output(self):
        pass

    def run_if_interactive(self):
        self.status.running = True
        self.to_hdf()

    def interactive_close(self):
        self.to_hdf()

    def interactive_fetch(self):
        pass

    def interactive_flush(self, path="generic", include_last_step=True):
        pass

    def run_if_refresh(self):
        pass

    def _run_if_busy(self):
        pass

    def write_input(self):
        pass

    def reconnect_s3_storage(self, config=None, bucket_name=None):
        config = config or self._storage_type_store.s3_config.config
        bucket_name = bucket_name or self._storage_type_store.s3_config.bucket_name
        try:
            external_storage = FileS3IO(
                config=config, path=self.path, bucket_name=bucket_name
            )
        except Exception as e:
            raise RuntimeError("Could not restore connection to the S3 storage.") from e
        if (
            self._storage_type_store.s3_config.bucket_info
            == external_storage.connection_info
        ):
            self._external_storage = external_storage
        else:
            raise RuntimeError(
                f"New and saved s3 storage do not match! Got {external_storage.connection_info}"
                f" but expected {self._storage_type_store.s3_config.connection_info}."
            )

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf, group_name)
        self._stored_files.from_hdf(hdf=self._hdf5)
        self._storage_type_store.from_hdf(hdf=self._hdf5)
        self._storage_type = StorageType(self._storage_type_store.storage_type)
        self._storage_type.read_only = self._storage_type_store.fixed_storage_type
        if self._storage_type.s3 and self.status != "initialized":
            self.reconnect_s3_storage()

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf, group_name)
        if self._storage_type.s3:
            self._hdf5["REQUIRE_FULL_OBJ_FOR_RM"] = True
        self._stored_files.to_hdf(hdf=self._hdf5)
        self._storage_type_store.storage_type = self.storage_type
        self._storage_type_store.fixed_storage_type = self._storage_type.read_only
        self._storage_type_store.to_hdf(hdf=self._hdf5)

    def __getitem__(self, item):
        # copied from super().__getitem__ changing the output of returning a file
        """
        Get/read data from the HDF5 file, child jobs or access log files.

        If the job is :method:`~.decompress`ed, item can also be a file name to
        access the raw output file of that name of the job.  See available file
        with :method:`~.list_files()`.

        `item` is first looked up in this jobs HDF5 file, then in the HDF5 files of any child jobs and finally it is
        matched against any files in the job directory as described above.

        Args:
            item (str, slice): path to the data or key of the data object

        Returns:
            dict, list, float, int, None: data or data object; if nothing is found None is returned
        """
        try:
            return self._hdf5[item]
        except ValueError:
            pass

        name_lst = item.split("/")
        item_obj = name_lst[0]

        # Here this function is changed to return a FileData object.
        if self._storage_type.local:
            if name_lst[0] in self.list_files():
                file_name = posixpath.join(
                    self.working_directory, "{}".format(item_obj)
                )
                return FileData(file=file_name, metadata=self._stored_files[item])
        elif self._storage_type.s3:
            if name_lst[0] in self._external_storage.list_nodes():
                return self._external_storage[name_lst[0]]
        return None
