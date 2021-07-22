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

    def list_resources(self):
        """
        Return a list of names of all configured resources (aka StorageJobs).

        Returns:
            list of str: names of nodes which are StorageJobs
        """
        table = self.job_table(recursive=False)
        bool_vec = table["hamilton"] == "StorageJob"
        return table[bool_vec]["job"].tolist()
