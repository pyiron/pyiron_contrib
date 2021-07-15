from pyiron_base import Project


class RDMProject(Project):
    """ Basically an easy wrapper of a generic Project to extend for metadata. """

    @property
    def metadata(self):
        try:
            return self.data.metadata
        except AttributeError:
            self.data.metadata = None
        return self.data.metadata

    @metadata.setter
    def metadata(self, metadata):
        self.data.metadata = metadata

    def save_metadata(self):
        self.data.write()

    @property
    def project_info(self):
        try:
            return self.data.project_info
        except AttributeError:
            self.data.project_info = None
        return self.data.project_info

    @project_info.setter
    def project_info(self, project_info):
        self.data.project_info = project_info

    def save_projectinfo(self):
        self.data.write()

