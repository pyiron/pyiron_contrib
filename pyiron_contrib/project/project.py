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
        self._project_browser = None
    __init__.__doc__ = ProjectCore.__init__.__doc__

    def open_browser(self, Vbox=None, show_files=False):
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

    def list_nodes(self, recursive=False):
        """
        List nodes/ jobs/ pyiron objects inside the project

        Args:
            recursive (bool): search subprojects [True/False] - default=False

        Returns:
            list: list of nodes/ jobs/ pyiron objects inside the project
        """
        if "nodes" not in self._filter:
            return []
        nodes = self.get_jobs(recursive=recursive, columns=["job"])["job"]
        return nodes

    def copy(self):
        """
        Copy the project object - copying just the Python object but maintaining the same pyiron path

        Returns:
            Project: copy of the project object
        """
        new = Project(path=self.path, user=self.user, sql_query=self.sql_query)
        return new

    def display_item(self, item, outwidget=None):
        from pyiron_contrib.generic.display_item import DisplayItem
        if item in self.list_files() and item not in self.list_files(extension="h5"):
            return DisplayItem(self.path+item, outwidget).display()
        else:
            return DisplayItem(self.__getitem__(item), outwidget).display()

