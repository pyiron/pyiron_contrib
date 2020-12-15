from pyiron_base import InputList
from pyiron import Project as ProjectCore
from pyiron_contrib.project.file_browser import FileBrowser

class Project(ProjectCore):

    """ Basically a wrapper of Project from pyiron_base to extend for metadata. """
    def __init__(self, path="", user=None, sql_query=None, default_working_directory=False):
        super().__init__(path=path,
                         user=user,
                         sql_query=sql_query,
                         default_working_directory=default_working_directory
                         )
        self._project_info = InputList(table_name="projectinfo")
        self._metadata = InputList(table_name="metadata")
        self._filebrowser = None
        self.hdf5 = self.create_hdf(self.path, self.base_name + "_projectdata")
        self.load_metadata()
        self._load_projectinfo()
    __init__.__doc__ = ProjectCore.__init__.__doc__

    def open_file_browser(self, Vbox=None):
        """
        Provides a file browser to inspect the local data system.

        Args:
             Vbox (:class:`ipywidgets.Vbox` / None): Vbox in which the file browser is displayed.
                                            If None, a new Vbox is provided.
        """
        if self._filebrowser is None:
            self._filebrowser = FileBrowser(project=self, Vbox=Vbox, fix_storage_sys=True, hdf_as_dirs=True)
        return self._filebrowser.gui()

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = InputList(metadata, table_name="metadata")

    def save_metadata(self):
        self._metadata.to_hdf(self.hdf5, group_name=None)

    def load_metadata(self):
        try:
            self._metadata.from_hdf(self.hdf5, group_name=None)
        except ValueError:
            pass

    @property
    def project_info(self):
        return self._project_info

    @project_info.setter
    def project_info(self, project_info):
        self._project_info = InputList(project_info, table_name="projectinfo")

    def _save_projectinfo(self):
        self._project_info.to_hdf(self.hdf5, group_name=None)

    def _load_projectinfo(self):
        try:
            self._project_info.from_hdf(self.hdf5, group_name=None)
        except ValueError:
            pass

    def copy(self):
        """
        Copy the project object - copying just the Python object but maintaining the same pyiron path

        Returns:
            Project: copy of the project object
        """
        new = Project(path=self.path, user=self.user, sql_query=self.sql_query)
        return new

    def open(self, rel_path, history=True):
        new = super().open(rel_path, history=history)
        new.hdf5 = new.create_hdf(new.path, new.base_name + "_projectdata")
        new._metadata = InputList(table_name="metadata")
        new._project_info = InputList(table_name="projectinfo")
        new.load_metadata()
        new._load_projectinfo()
        return new
