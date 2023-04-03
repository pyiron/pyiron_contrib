from pyiron_base import Toolkit

from pyiron_contrib.generic.coscineIo import CoscineProject
from pyiron_contrib.generic.s3io import FileS3IO


class StorageInterfaceFactory(Toolkit):

    def __init__(self, project):
        super().__init__(project)
        self._storage_interface = {}

    def create_s3_interface(self, *args, **kwargs):
        return FileS3IO(*args, **kwargs)

    def attach_s3(self, s3_interface: FileS3IO):
        self._project.data.create_group('StorageInterface')
        self._project.data.StorageInterface['s3'] = {
            's3path': s3_interface.s3_path,
            'bucket': s3_interface.bucket_info
        }

    @property
    def s3(self):
        if 's3' in self._storage_interface:
            return self._storage_interface['s3']
        elif 's3' in self._project.data.StorageInterface:
            pass
            # ToDo: Connect S3
        else:
            raise AttributeError('S3 access not available for this project.')

    # ToDo: same vor cosine
    def coscine(self, *args, **kwargs):
        return CoscineProject(*args, **kwargs)