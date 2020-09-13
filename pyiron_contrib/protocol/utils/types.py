# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_contrib.protocol.utils.misc import getLogger
from abc import ABCMeta
import sys

"""
Utility function to inject subclasses of pyiron_contrib.protocol.generic.Protocol as JobTypes into pyiron
"""

__author__ = "Dominik Gehringer, Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "December 10, 2019"


# we have to use ABC as parent metatype otherwise we will run into a metaclass conflict however for Protocol that is
#exactly what we want since one is not allowed to instantiate a protocol
class PyironJobTypeRegistryMetaType(ABCMeta):
    """
    Metatype which keeps track of all its subclasses
    """

    __registry = {}
    __wrapped = {}

    @staticmethod
    def inject_dynamic_types():
        """
        Sets all the dynimically created classes as attributes to the current module object, however with the original
        name. Updates pyiron JobTypeChoice's JOB_CLASS_DICT
        """

        # fetch the current module name
        module_name = PyironJobTypeRegistryMetaType.__module__

        if module_name not in sys.modules:
            # throw a info message if the module is no loaded yes -> could possibly happen but is very unlikely
            getLogger('PyironJobTypeRegistryMetaType').info('Skipping not module "%s" not loaded yet. Please call PyironJobTypeRegistryMetaType.inject_dynamic_types manually')
        else:
            # get the module object
            module = sys.modules[module_name]
            updates = {}
            # register all dynamically created subclasses of (cls, GenericJob) as attributes
            for (origin_module, wrapped_name), tp in PyironJobTypeRegistryMetaType.__wrapped.items():
                if not hasattr(module, wrapped_name):
                    setattr(module, wrapped_name, tp)
                    # information needed for pyiron.base.job.jobtype.JOB_CLASS_DICT
                    updates[wrapped_name] = module_name

            # inject it into pyiron
            # adding it to JOB_CLASS_DICT also enables it for autocomplete
            from pyiron_base import JOB_CLASS_DICT
            # make sure that it affects all instances of pyiron.base.job.jobtype.JobTypeChoice
            JOB_CLASS_DICT.update(updates)

    def __init__(cls, name, bases, nmspc):
        """
        type initialize for type of this metaclass
        Args:
            name: (str) class name
            bases: (tuple of type) the base classes
            nmspc: (dict) the attribute of the tpye
        """

        super(PyironJobTypeRegistryMetaType, cls).__init__(name, bases, nmspc)
        # that's a little bit ugly but we want to exclude this types and do not want to wrap them
        if name in ('PyironJobTypeRegistry', 'Protocol', 'PyironJobTypeRegistryMetaType'):
            return
        # convenience function to obtain the full class qualifier
        fullname = lambda : '%s.%s' %( cls.__module__, cls.__name__)

        class_name = cls.__name__
        class_module = cls.__module__
        key = (class_module, class_name)
        # construct the new class name
        new_class_name = class_name
        # we place a `__artificial__` attribute in the type information to indicate that it is a dynamically created type
        # when calling `type(new_class_name, new_bases, new_spec)` we construct a subclass of cls, thus it is a
        # recsursive call to this method, we have to avoid that
        # otherwise we wrap the wrapper again and agin
        if '__artificial__' not in nmspc:
            from pyiron_base import GenericJob
            new_bases = (cls, GenericJob)
            # edit the metainformation
            new_spec = nmspc.copy()
            new_spec['__qualname__'] = new_class_name
            new_spec['__artificial__'] = True
            new_spec['__module__'] = PyironJobTypeRegistryMetaType.__module__
            # this bad guy caused some troubles
            if '__classcell__' in new_spec:
                del new_spec['__classcell__']
            try:
                new_type = type(new_class_name, new_bases, new_spec)
            except TypeError as e:
                # if creating the type goes wrong we are really screwed up
                getLogger(fullname()).exception(fullname(), exc_info=e)
                raise
            # register everything nicely
            PyironJobTypeRegistryMetaType.__registry[key] = cls
            PyironJobTypeRegistryMetaType.__wrapped[key] = new_type
            PyironJobTypeRegistryMetaType.wrapped = PyironJobTypeRegistryMetaType.__wrapped

            # try to inject it -> in case of dynamically created subclasses
            PyironJobTypeRegistryMetaType.inject_dynamic_types()


class PyironJobTypeRegistry(metaclass=PyironJobTypeRegistryMetaType):
    """
    Convenience class as it is ABC in the abc module
    """
    pass
