# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
import os.path
import io

from pyiron_base import ImportAlarm
from pyiron_base.interfaces.has_groups import HasGroups
from pyiron_contrib.generic.filedata import FileDataTemplate, load_file
from typing import Union, List

try:
    import coscine
    import_alarm = ImportAlarm()
except ImportError:
    import_alarm = ImportAlarm("Connecting to CoScInE requires the coscine package.")
    import_alarm.warn_if_failed()


class CoscineMetadata(coscine.resource.MetadataForm):
    """Add a proper representation to the coscine version"""
    def __init__(self, meta_data_form:coscine.resource.MetadataForm):
        super().__init__(
            profile=meta_data_form.profile,
            lang=meta_data_form._lang,
            vocabulary=meta_data_form._vocabulary,
            entries=meta_data_form._entries
        )

    def __repr__(self):
        return self.__str__()


class CoscineFileData(FileDataTemplate):
    def __init__(self, coscine_object: coscine.Object):
        self._coscine_object = coscine_object
        self._data = None
        self._filename = coscine_object.name
        self.filetype = self._get_filetype_from_filename(self._filename)

    @property
    def data(self):
        if self._data is None:
            self._data = self._coscine_object.content()
        return load_file(io.BytesIO(self._data), filetype=self.filetype)

    def download(self, path="./"):
        self._coscine_object.download(path=path)

    @property
    def metadata(self):
        return CoscineMetadata(self._coscine_object.MetadataForm())


class CoscineResource(HasGroups):
    def __init__(self, resource: coscine.Resource):
        self._resource = resource

    def _list_nodes(self):
        return [obj.name for obj in self._resource.objects()]

    def _list_groups(self):
        # Right now a coscine resource is flat.
        return []

    def remove_file(self, item):
        file_obj = self._get_one_file_obj(item, error_msg=f"Multiple matches found for item = {item}, aborting remove!")
        if file_obj is not None:
            file_obj.delete()

    def upload(self, file, metadata: coscine.resource.MetadataForm, filename=None):
        filename = filename or os.path.basename(file)
        _meta_data = metadata.generate()
        self._resource.upload(filename, file, _meta_data)

    def _get_one_file_obj(self, item, error_msg) -> Union[coscine.Object, None]:
        if item not in self.list_nodes():
            return None

        objects = self._resource.objects(Name=item)
        if len(objects) == 1:
            return objects[0]
        elif len(objects) > 1:
            raise ValueError(error_msg)
        else:
            raise RuntimeError(f"This should be impossible. Please contact the developers.")

    def __getitem__(self, item):
        if item in self._list_nodes():
            return CoscineFileData(
                self._get_one_file_obj(
                    item,
                    f"Multiple matches for item = {item}, use `get_items(name={item})` to receive all matching files."
                    f" See get_items docstring for more filtering options."
                )
            )
        raise ValueError(f"Unknown item {item}")

    def get_items(self, name=None, **kwargs) -> List[CoscineFileData]:
        """Use the coscine filtering options to receive all matching files.

        Args:
            name(str/None): The name of the file as provided by `list_nodes`. None is default.
            **kwargs: Additional keyword arguments as known to coscine. The most important of these is
                'Name' for which this method has the 'name' alias. Any file _not_ matching the value for the
                keyword will be excluded.
                In coscine-0.5.1 the following list of kwargs is available:
                ['Name', 'Path', 'Size', 'Kind', 'Modified', 'Created', 'Provider', 'IsFolder', 'IsFile', 'Action']
        Raises:
            ValueError: if 'name' and 'Name' are both given and do not match.
        """
        _name = kwargs.pop('Name', None)
        if name != _name:
            raise ValueError(f"name='{name}' and Name='{_name}' both provided and not matching.")

        kwargs['Name'] = name or _name
        return [CoscineFileData(obj) for obj in self._resource.objects(**kwargs)]


class CoscineProject(HasGroups):
    def __init__(self, project: Union[coscine.Project, coscine.Client, str]):
        """
        project(coscine.project/coscine.client/str):
        """
        self._project = None
        if isinstance(project, str) and os.path.isfile(project):
            with open(project) as f:
                token = f.read()
            self._client = self._connect_client(token)
        elif isinstance(project, str):
            self._client = self._connect_client(project)
        elif hasattr(project, 'client'):
            self._project = project
            self._client = project.client
        else:
            self._client = project

        if self._client.verbose:
            print("Silenced client!")
            self._client.verbose = False
        self._path = None

    @staticmethod
    def _connect_client(token):
        client = coscine.Client(token, verbose=False)
        try:
            client.projects()
        except coscine.CoscineException as e:
            raise ValueError("Error connecting to CoScInE with provided token.") from e
        return client

    @property
    def verbose(self):
        return self._client.verbose

    @verbose.setter
    def verbose(self, val):
        self._client.verbose = val

    def _list_groups(self):
        if self._project is None:
            return [pr.name for pr in self._client.projects()]
        else:
            return [pr.name for pr in self._project.subprojects()]

    def _list_nodes(self):
        if self._project is None:
            return []
        else:
            return [res.name for res in self._project.resources()]

    def __getitem__(self, key):
        if key in self.list_nodes():  # This implies project is not None
            return CoscineResource(self._project.resource(key))
        return self.get_group(key)

    def get_node(self, key):
        if key in self.list_nodes():
            return self._project.resource(key)
        else:
            return KeyError(key)

    def get_group(self, key):
        if key in self.list_groups() and self._project is not None:
            return self.__class__(self._project.subprojects(displayName=key)[0])
        elif key in self.list_groups():
            return self.__class__(self._client.project(key))
        else:
            raise KeyError(key)
