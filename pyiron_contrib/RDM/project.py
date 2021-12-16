import posixpath

from pyiron_atomistics import Project as ProjectBase
import os

from pyiron_base import ProjectHDFio
from pyiron_contrib.generic.filedata import FileData


class Project(ProjectBase):

    def _get_item_helper(self, item, convert_to_object=True):
        """
        Internal helper function to get item from project

        Args:
            item (str, int): key
            convert_to_object (bool): convert the object to an pyiron object or only access the HDF5 file - default=True
                                      accessing only the HDF5 file is about an order of magnitude faster, but only
                                      provides limited functionality. Compare the GenericJob object to JobCore object.

        Returns:
            Project, GenericJob, JobCore, dict, list, float: basically any kind of item inside the project.
        """
        if item == "..":
            return self.parent_group
        if item in self.list_nodes():
            if self._inspect_mode or not convert_to_object:
                return self.inspect(item)
            return self.load(item)
        if item in self.list_files(extension="h5"):
            file_name = posixpath.join(self.path, "{}.h5".format(item))
            return ProjectHDFio(project=self, file_name=file_name)
        if item in self.list_files():
            file_name = posixpath.join(self.path, "{}".format(item))
            return FileData(file_name)
        if item in self.list_dirs():
            with self.open(item) as new_item:
                return new_item.copy()
        if item in os.listdir(self.path) and os.path.isdir(os.path.join(self.path, item)):
            return self.open(item)
        raise ValueError("Unknown item: {}".format(item))

    def get_pr_browser(self):
        from pyiron_gui.project.project_browser import ThreeWindowHasGroupsBrowserWithOutput
        return ThreeWindowHasGroupsBrowserWithOutput(self)
