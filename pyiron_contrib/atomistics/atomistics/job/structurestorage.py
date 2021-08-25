from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage as StructureStorageBase
from pyiron_base import deprecate

class StructureStorage(StructureStorageBase):
    @deprecate("import from pyiron_atomistics.atomistics.structure.structurestorage instead")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
