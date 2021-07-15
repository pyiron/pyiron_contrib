import os
import posixpath
import shutil

from pyiron_base import GenericJob, DataContainer
from pyiron_base.generic.filedata import FileData


class StorageType:
    _available_storage_types = ['local', 's3']

    def __init__(self, storage_type):
        """What type of storage is used

        Parameters:
            storage_type(str): one of ['local', 's3']
        """
        self._storage_type = None
        self._read_only = False
        if storage_type in self._available_storage_types:
            self.storage_type = storage_type
        else:
            raise ValueError(f"Expected one of {self._available_storage_types} but got {storage_type}")

    @property
    def storage_type(self):
        return self._storage_type

    def set_read_only(self):
        self._read_only = True

    @storage_type.setter
    def storage_type(self, storage_type):
        if self._read_only:
            raise RuntimeError("The storage type cannot be changed when it is already in use.")
        for storage in self._available_storage_types:
            setattr(self, storage, False)
        setattr(self, storage_type, True)


class StorageJob(GenericJob):

    def __init__(self, project, name):
        super().__init__(project, name)
        self.server.run_mode.interactive = True
        self._stored_files = DataContainer(table_name='stored_files')
        self._storage_type = StorageType('local')

    @property
    def storage_type(self):
        return self._storage_type.storage_type

    @storage_type.setter
    def storage_type(self, storage_type):
        self._storage_type.storage_type = storage_type

    def validate_ready_to_run(self):
        if self._storage_type.local:
            self._create_working_directory()
            self._storage_type.set_read_only()
        else:
            raise NotImplementedError("Currently, only a local Storage is implemented.")

    def add_files(self, filenames, metadata=None, overwrite=False):
        """Add files to the storage

        Parameters:
            filenames(str/list of str): Files to save in the Storage
            metadata(dict/list of dict/None): metadata to attach to the files:
                if dict: apply this metadata dict to all files
                if list: has to be of same length as filenames, one metadata per file
                if None: Do not attach metadata (not recommended)
            overwrite(bool): if true to overwrite already present files
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        if isinstance(metadata, list) and len(metadata) != len(filenames):
            raise ValueError(f"Length of filenames and metadata have to match, "
                             f"but got len(filenames)={len(filenames)} and len(metadata)={len(metadata)}.")

        for i, file in enumerate(filenames):
            if not overwrite and file in self._stored_files.keys():
                print(f"WARNING: {file} not copied, since already present and 'overwrite'=False.")
                continue
            self._store_file(file)
            self._stored_files[file] = metadata[i] if isinstance(metadata, list) else metadata

        self.run()

    def _store_file(self, file):
        if self._storage_type.local:
            shutil.copy(file, self.working_directory)

    def _remove_file(self, file):
        if self._storage_type.local:
            os.remove(os.path.join(self.working_directory, file))

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
            raise FileNotFoundError(f"Files {files_not_found} not found, abort removing files. You may specify "
                                    f"raise_error=False to suppress this error.")

        files_to_remove = [file for file in filenames if file in self._stored_files]
        if dryrun and not silent:
            print(f"dryrun: to remove {files_to_remove} specify dryrun=False")
            return

        for file in files_to_remove:
            del self._stored_files[file]
            self._remove_file(file)

        if not silent:
            print(f"removed {files_to_remove}")
        self.run()

    def run_static(self):
        raise NotImplementedError('There is no static run mode for a Storage job.')

    def collect_output(self):
        pass

    def append(self, job):
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

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(hdf, group_name)
        self._stored_files.from_hdf(hdf=self._hdf5, group_name=group_name)

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf, group_name)
        self._stored_files.to_hdf(hdf=self._hdf5, group_name=group_name)

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
        if item_obj in self._list_ext_childs():
            child = self._hdf5[self._name + "_hdf5/" + item_obj]
            print("job get: ", self._name + "_jobs")
            if len(name_lst) == 1:
                return child
            else:
                return child["/".join(name_lst[1:])]

        if name_lst[0] in self.list_files():
            file_name = posixpath.join(self.working_directory, "{}".format(item_obj))
            # Here this funtion is changed to return a FileData object.
            return FileData(file=file_name, metadata=self._stored_files[item])
        return None
