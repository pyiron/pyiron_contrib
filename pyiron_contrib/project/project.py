from pyiron_base import InputList, ImportAlarm
from pyiron import Project as ProjectCore
import posixpath
try:
    from pyiron_contrib.project.project_browser import ProjectBrowser
    import_alarm = ImportAlarm()
except ImportError:
    import_alarm = ImportAlarm(
        "The dependencies of the project's browser are not met (ipywidgets, IPython)."
    )


class Project(ProjectCore):

    """ Basically a wrapper of Project from pyiron_atomistic to extend functionality. """
    @import_alarm
    def __init__(self, path="", user=None, sql_query=None, default_working_directory=False):
        super().__init__(path=path,
                         user=user,
                         sql_query=sql_query,
                         default_working_directory=default_working_directory
                         )
        self._project_info = InputList(table_name="projectinfo")
        self._metadata = InputList(table_name="metadata")
        self._project_browser = None
        self.hdf5 = self.create_hdf(self.path, self.base_name + "_projectdata")
        self.from_hdf()
    __init__.__doc__ = ProjectCore.__init__.__doc__

    @property
    def browser(self, Vbox=None, show_files=False):
        """
        Provides a file browser to inspect the local data system.

        Args:
             Vbox (:class:`ipywidgets.Vbox` / None): Vbox in which the file browser is displayed.
                                            If None, a new Vbox is provided.
        """
        if self._project_browser is None:
            self._project_browser = ProjectBrowser(project=self,
                                                   show_files=show_files,
                                                   Vbox=Vbox)
        else:
            self._project_browser.update(Vbox=Vbox, show_files=show_files)
        return self._project_browser.gui()

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = InputList(metadata, table_name="metadata")

    @property
    def project_info(self):
        return self._project_info

    @project_info.setter
    def project_info(self, project_info):
        self._project_info = InputList(project_info, table_name="projectinfo")

    def to_hdf(self, hdf=None, group_name=None):
        """Store meta data and info of the project in the project hdf5 file."""
        if hdf is None:
            hdf = self.hdf5
        self._metadata.to_hdf(hdf, group_name=None)
        self._project_info.to_hdf(hdf, group_name=None)

    def from_hdf(self, hdf=None, group_name=None):
        """Load meta data and info of the project from the project hdf5 file."""
        if hdf is None:
            hdf = self.hdf5
        try:
            self._metadata.from_hdf(hdf, group_name=None)
            self._project_info.from_hdf(hdf, group_name=None)
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
        new.from_hdf()
        return new

    def display_item(self, item, outwidget=None):
        from pyiron_contrib.generic.display_item import DisplayItem
        if item in self.list_files() and item not in self.list_files(extension="h5"):
            DisplayItem(self.path+item, outwidget)
        else:
            DisplayItem(self.__getitem__(item), outwidget)

