import warnings


try:
    from pyiron import Project
except:
    warnings.warn("pyiron module not found, importing Project from pyiron_base")
    from pyiron_base import Project

from pyiron_contrib.generic.storage_interface_toolkit import StorageInterfaceFactory

Project.register_tools("storage_interface", StorageInterfaceFactory)
