# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.
from dateutil import parser
import os.path
import io
import warnings

import pandas as pd

from pyiron_base import ImportAlarm
from pyiron_base.interfaces.has_groups import HasGroups
from pyiron_contrib.generic.filedata import StorageInterface
from pyiron_contrib.generic.filedata import (
    FileDataTemplate,
    load_file,
    MetaDataTemplate,
)
from typing import Union, List

try:
    import coscine

    import_alarm = ImportAlarm()
except ImportError:
    import_alarm = ImportAlarm("Connecting to CoScInE requires the coscine package.")
    import_alarm.warn_if_failed()


class CoscineMetadata(coscine.resource.MetadataForm, MetaDataTemplate):
    """Add a proper representation to the coscine version"""

    def __init__(self, meta_data_form: coscine.resource.MetadataForm):
        super().__init__(client=meta_data_form.client, graph=meta_data_form.profile)

    def to_dict(self):
        result = {}
        for key, value in self.items():
            if len(str(value)) > 0:
                result[key] = value
        return result

    def __repr__(self):
        return self.__str__()


class CoscineFileData(FileDataTemplate):
    def __init__(self, coscine_object: coscine.FileObject):
        super().__init__()
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

    def _get_metadata(self):
        form = CoscineMetadata(self._coscine_object.form())
        return form

    def _set_metadata(self, metadata):
        raise AttributeError("can't set attribute")


def _list_filter(list_to_filter, **kwargs):
    """Return entry in list matching all attributes."""

    def filter_func(item):
        for key, value in kwargs:
            if getattr(item, key) != value:
                return False
        return True

    return list(filter(filter_func, list_to_filter))


class CoscineResource(StorageInterface):
    def __init__(self, resource: Union[coscine.Resource, dict, "CoscineResource"]):
        """Giving access to a CoScInE Resource to receive or upload files

        Args:
            resource (coscine.Resource/dict): Either directly provide a coscine.Resource or a dictionary with
                token, project_id, and resource_id to directly connect to the respective resource.
        """
        if isinstance(resource, coscine.Resource):
            self._resource = resource
        elif isinstance(resource, dict):
            client = coscine.Client(resource["token"])
            pr = _list_filter(
                client.projects(toplevel=False), id=resource["project_id"]
            )[0]
            self._resource = _list_filter(pr.resources(), id=resource["resource_id"])[0]
        elif isinstance(resource, self.__class__):
            self._resource = resource._resource
        else:
            raise ValueError(
                f"Unknown resource type {type(resource)}, supported types !"
            )

    def _list_nodes(self):
        return [obj.name for obj in self._resource.objects()]

    def _list_groups(self):
        # Right now a coscine resource is flat.
        return []

    def is_dir(self, item):
        return False

    def is_file(self, item):
        if item in self.list_nodes():
            return True
        else:
            return False

    def remove_file(self, item):
        file_obj = self._get_one_file_obj(
            item,
            error_msg=f"Multiple matches found for item = {item}, aborting remove!",
        )
        if file_obj is not None:
            file_obj.delete()

    def upload_file(
        self, file, metadata: coscine.resource.MetadataForm = None, filename=None
    ):
        _meta_data = self.validate_metadata(metadata)
        filename = filename or os.path.basename(file)
        self._resource.upload(filename, file, _meta_data)

    def _get_one_file_obj(self, item, error_msg) -> Union[coscine.FileObject, None]:
        if item not in self.list_nodes():
            return None

        objects = self._resource.objects(Name=item)
        if len(objects) == 1:
            return objects[0]
        elif len(objects) > 1:
            raise ValueError(error_msg)
        else:
            raise RuntimeError(
                f"This should be impossible. Please contact the developers."
            )

    def __getitem__(self, item):
        if item in self._list_nodes():
            return CoscineFileData(
                self._get_one_file_obj(
                    item,
                    f"Multiple matches for item = {item}, use `get_items(name={item})` to receive all matching files."
                    f" See get_items docstring for more filtering options.",
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
        _name = kwargs.pop("Name", None)
        if name != _name:
            raise ValueError(
                f"name='{name}' and Name='{_name}' both provided and not matching."
            )

        kwargs["Name"] = name or _name
        return [
            CoscineFileData(obj)
            for obj in _list_filter(self._resource.objects(), **kwargs)
        ]

    @property
    def metadata_template(self) -> CoscineMetadata:
        return CoscineMetadata(self._resource.MetadataForm())

    @property
    def requires_metadata(self):
        return True

    def _get_form_from_dict(self, metadata_dict):
        form = self.metadata_template
        form.clear()

        # Try to update form by the dictionary:
        try:
            form.fill(metadata_dict)
        except KeyError as e:
            store_err = KeyError(f"Meta data template has no key '{e.args[0]}'")
        else:
            return form, None

        # Assume we already got a dictionary from form.generate()
        # A form has the method parse(data) which is intended to be used on a coscine.Object.metadata which is the same
        # as {'file_handle_with_file_name': from.generate()} of the form used to upload; thus:
        form.parse({"dummy_file_name": metadata_dict})
        return form, store_err

    def parse_metadata(self, metadata):
        generated_metadata = self.validate_metadata(metadata, raise_error=False)
        if generated_metadata is None:
            raise ValueError("metadata is not valid")
        form, _ = self._get_form_from_dict(generated_metadata)
        return {key: val.raw() for key, val in form.items()}

    def validate_metadata(self, metadata, raise_error=True):
        encountered_error = None
        if metadata is None and raise_error:
            raise ValueError(
                "Coscine resources require meta data to upload a file! Use metadata_template to "
                "get the correct meta data form and provide a completed form here."
            )
        elif metadata is None:
            return None
        elif isinstance(metadata, coscine.resource.MetadataForm):
            metadata_form = self.metadata_template
            metadata_form.clear()
            metadata_form.fill({k: v for k, v in metadata.items()})
        else:
            metadata_form, encountered_error = self._get_form_from_dict(metadata)

        try:
            result_meta_data = metadata_form.generate()
        except ValueError as e:
            if raise_error and encountered_error is not None:
                raise ValueError(
                    f"The provided meta data is not valid; might be related to previous error "
                    f"{encountered_error}"
                ) from e
            elif raise_error:
                raise ValueError(f"The provided meta data is not valid") from e
            else:
                return None
        else:
            return result_meta_data


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
        elif hasattr(project, "client"):
            self._project = project
            self._client = project.client
        else:
            self._client = project

        if self._client.settings.verbose:
            print("Silenced client!")
            self._client.settings.verbose = False
        self._path = None

    @staticmethod
    def _connect_client(token):
        client = coscine.Client(token)
        try:
            client.projects()
        except (PermissionError, RuntimeError, ConnectionError) as e:
            raise ValueError("Error connecting to CoScInE with provided token.") from e
        return client

    @property
    def read_only(self):
        return self._client.settings.read_only

    @read_only.setter
    def read_only(self, val: bool):
        self._client.settings.read_only = val

    @property
    def verbose(self):
        return self._client.settings.verbose

    @verbose.setter
    def verbose(self, val):
        self._client.settings.verbose = val

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
            try:
                return self.__class__(self._project.subproject(display_name=key))
            except IndexError:
                warnings.warn("More than one project matches - returning first match!")
                for _pr in self._project.subprojects():
                    if _pr.display_name == key:
                        return self.__class__(_pr)
        elif key in self.list_groups():
            return self.__class__(self._client.project(key))
        else:
            raise KeyError(key)

    def create_group(
        self,
        project_name,
        display_name=None,
        project_description=None,
        principal_investigators=None,
        project_start=None,
        project_end=None,
        discipline=None,
        participating_organizations=None,
        project_keywords=None,
        metadata_visibility=None,
        grant_id=None,
    ):
        if project_name in self.list_all():
            raise ValueError("The name is already in this project!")
        update_dict = {
            "Project Name": project_name,
            "Display Name": display_name,
            "Project Description": project_description,
            "Principal Investigators": principal_investigators,
            "Project Start": project_start,
            "Project End": project_end,
            "Discipline": discipline,
            "Participating Organizations": participating_organizations,
            "Project Keywords": project_keywords,
            "Metadata Visibility": metadata_visibility,
            "Grant ID": grant_id,
        }
        for key, val in update_dict.items():
            if val is None:
                del update_dict[key]

        if 'Display Name' not in update_dict:
            update_dict['Display Name'] = project_name

        for key in ['Project Start', 'Project End']:
            if isinstance(update_dict[key], str):
                update_dict[key] = parser.parse(update_dict[key])

        form = self._client.project_form() if self._project is None else self._project.form()
        form.fill(update_dict)
        form.generate()
        if self._project is None:
            self._client.create_project(form)
        else:
            self._project.create_subproject(form)
