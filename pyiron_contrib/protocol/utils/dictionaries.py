# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import re
import numpy as np
from collections import OrderedDict
from pyiron_contrib.protocol.utils.pointer import Pointer
from pyiron_contrib.protocol.utils.misc import  LoggerMixin, fullname
from pyiron_atomistics.atomistics.structure.atoms import Atoms

"""
Classes to setup input and output dataflows for protocols
"""

__author__ = "Dominik Gehringer, Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "December 10, 2019"

# define a regex to find integer values
integer_regex = re.compile(r'[-+]?([1-9]\d*|0)')


TIMELINE_DICT_KEY_FORMAT = 't_{time}'
GENERIC_LIST_INDEX_FORMAT = 'i_{index}'


class IODictionary(dict, LoggerMixin):
    """
    A dictionary class representing the parameters of a Command class. The dictionary holds a path which is recipe
    that can be resolved at runtime to obtain underlying values. A dictionary instance can hold multiple instances of
    IODictionary as value items which can be resolved into the real values when desired.
    """

    # those members are not accessible
    _protected_members = [
        'resolve',
        'to_hdf',
        'from_hdf'
    ]

    def __init__(self, **kwargs):
        super(IODictionary, self).__init__(**kwargs)

    def __getattr__(self, item):
        if item in IODictionary._protected_members:
            return object.__getattribute__(self, item)
        return self.__getitem__(item)

    def __getitem__(self, item):
        if item in self.keys():
            value = super(IODictionary, self).__getitem__(item)
            if isinstance(value, Pointer):
                return ~value
            elif isinstance(value, list):  # TODO: Allow other containers than list
                cls = type(value)
                return cls([element if not isinstance(element, Pointer) else ~element for element in value])
            else:
                return value
        return super(IODictionary, self).__getitem__(item)

    def __setattr__(self, key, value):
        super(IODictionary, self).__setitem__(key, value)

    def resolve(self):
        """
        Even though printing the dictionary, or asking for a particular item resolves paths fine, somehow using the **
        unpacking syntax fails to resolve pointers. This is to cover that issue since I couldn't find anything on
        google how to modify the ** behaviour.
        """
        resolved = {}
        for key in self.keys():
            resolved[key] = self.__getitem__(key)
        return resolved

    @staticmethod
    def _try_save_key(k, v, hdf, exclude=(dict, tuple, list)):
        """
        Tries to save a simple value

        Args:
            k: (str) key name of the HDF entry
            v: (obj) value
            hdf: the hdf server

        Returns: (bool) wether saving the key was successful or not

        """
        # try to call to_hdf first
        if hasattr(v, 'to_hdf'):
            v.to_hdf(hdf, group_name=k)
            result = True
        else:
            if exclude is not None:
                if isinstance(v, exclude):
                    return False
            try:
                # try to do it the easy way
                hdf[k] = v
                result = True
            except TypeError:
                result = False
        return result

    def _generic_to_hdf(self, value, hdf, group_name=None):
        """
        Saves also dictionaries and lists to hdf
        Args:
            value: (obj) to object to save
            hdf: the hdf server
            group_name: (str) the group name where to store it
        """
        if isinstance(value, dict):
            # if we deal with a dictionary we have to open a new group anyway
            with hdf.open(group_name) as server:
                # store class metadata
                server['TYPE'] = str(type(value))
                server['FULLNAME'] = fullname(value)
                for k, v in value.items():
                    # try to save it
                    if not isinstance(k, str):
                        # it is possible that the keys are not strings, thus we have to enforce this
                        self.logger.warning('Key "%s" is not a string, it will be converted to %s' % (k, str(k)))
                        k = str(k)
                    # try it the easy way first (either call v.to_hdf or directly save it
                    if self._try_save_key(k, v, server):
                        pass  # everything was successful
                    else:
                        # well pyiron did not manage lets -> more complex object
                        self._generic_to_hdf(v, server, group_name=k)
        elif isinstance(value, (list, tuple)):
            # check if all do have the same type -> then we can make a numpy array out of it
            if len(value) == 0:
                pass  # there is nothing to do, no data to store
            else:
                first_type = type(value[0])
                same = all([type(v) == first_type for v in value])
                # if all items are of the same type and it is simple
                if same and issubclass(first_type, (float, complex, int, np.ndarray)):
                    # that is trivial we do have an array
                    if issubclass(first_type, np.ndarray):
                        # we do not want dtype=object, thus we do make this distinction
                        hdf[group_name] = np.array(value)
                    else:
                        hdf[group_name] = np.array(value, dtype=first_type)
                else:
                    with hdf.open(group_name) as server:
                        # again write the metadata
                        server['TYPE'] = str(type(value))
                        server['FULLNAME'] = fullname(value)
                        for i, v in enumerate(value):
                            index_key = GENERIC_LIST_INDEX_FORMAT.format(index=i)
                            if self._try_save_key(index_key, v, server):
                                pass  # everything was successful
                            else:
                                self._generic_to_hdf(v, server, group_name=index_key)
        else:
            # so this one is the primitive item case
            # lets check if it has a to_hdf method
            try:
                value.to_hdf(hdf, group_name=group_name)
            except AttributeError:
                # Ok there is no to_hdf method however lets try it again
                try:
                    hdf[group_name] = value
                except:
                    # now we have no clue any more, we have to raise this error
                    raise

    def _generic_from_hdf(self, hdf, group_name=None):
        """
        Loads dicts, lists and tuples as well as their subclasses from an hdf file

        Args:
            hdf: the hdf server
            group_name: (str) the group name

        Returns: (obj) the object to return
        """

        # handle special types at first
        # try a simple load
        if 'TYPE' not in hdf[group_name].list_nodes():
            return hdf[group_name]
        elif hdf[group_name]['TYPE'] == str(IODictionary):
            iodict = IODictionary()
            iodict.from_hdf(hdf, group_name)
            return iodict
        elif hdf[group_name]['TYPE'] == str(Atoms):
            struct = Atoms()
            struct.from_hdf(hdf, group_name)
            return struct
        # FULLNAME will only be present if _generic_to_hdf wrote the underlying object
        elif 'FULLNAME' in hdf[group_name].keys():
            with hdf.open(group_name) as server:
                from pydoc import locate
                # convert the class qualifier to a type
                cls_ = locate(server['FULLNAME'])
                # handle a dictionary
                if issubclass(cls_, dict):
                    result = {}
                    # nodes are primitive objects -> that is easy
                    for k in server.list_nodes():
                        # skip the special nodes
                        if k in ('TYPE', 'FULLNAME'):
                            continue
                        result[k] = server[k]

                    for k in server.list_groups():
                        # groups are more difficult, since they're other objects -> give it a try
                        result[k] = self._generic_from_hdf(server, group_name=k)

                    # create the instance -> we have to assume a constructor of type cls_(**kwargs) for that
                    # NOTE: if the default constructor is not available this code will break
                    result = cls_(result)
                    return result
                elif issubclass(cls_, (list, tuple)):
                    result = []
                    # we have to keep track of the indices -> str.__cmp__ != int.__cmp__ we cannot assume an order
                    indices = []

                    for k in server.list_nodes():
                        if k in ('TYPE', 'FULLNAME'):
                            continue
                        # nodes are trivial
                        index = int(k.replace('i_', ''))
                        result.append(server[k])
                        indices.append(index)
                        # TODO: Since Atoms object appear as a node we might have to call it here too

                    for k in server.list_groups():
                        # we do have the recursive call here
                        index = int(k.replace('i_', ''))
                        result.append(self._generic_from_hdf(server, group_name=k))
                        indices.append(index)

                    # sort it, with the keys as indices
                    result = sorted(enumerate(result), key=lambda t: indices[t[0]])
                    # create the instance, and get rid of the instances
                    result = cls_([val for idx, val in result])
                    return result
                else:
                    raise ImportError('Could not locate type(%s)' % server['FULLNAME'])
        else:
            raise TypeError('I do not know how to deserialize type(%s)' % hdf[group_name])

    def to_hdf(self, hdf, group_name=None):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server['TYPE'] = str(type(self))

            for key in list(self.keys()):
                # default value is not to save any property
                try:
                    value = getattr(self, key)
                    try:
                        value.to_hdf(hdf5_server, group_name=key)
                    except AttributeError:
                        self._generic_to_hdf(value, hdf5_server, group_name=key)
                except KeyError:
                    # to_hdf will get called *before* protocols have run, so the pointers in these dictionaries
                    # won't be able to resolve. For now just let it not resolve and don't save it.
                    continue
                except (RuntimeError, OSError, TypeError):
                    # if a "key" is initialized with a primitive value and the and the graph was already saved
                    # it might happen that the "key" already exists hdf5_server[key] but is of wrong HDF5 type
                    # e.g dataset instead of group. Thus the underlying library will raise an runtime error.
                    # The current workaround now is to try to delete the dataset and rewrite it
                    # TODO: Change to `del hdf5_server[key]` once pyiron.base.generic.hdfio is fixed
                    import posixpath
                    # hdf5_server.h5_path is relative
                    del hdf5_server[posixpath.join(hdf5_server.h5_path, key)]
                    # now we try again
                    try:
                        value.to_hdf(hdf5_server, group_name=key)
                    except AttributeError:
                        self._generic_to_hdf(value, hdf5_server, group_name=key)
                    except Exception:
                        raise

    def from_hdf(self, hdf, group_name):
        with hdf.open(group_name) as hdf5_server:
            for key in hdf5_server.list_nodes():
                if key in ('TYPE', 'FULLNAME'):
                    continue
                # Nodes are leaves, so just save them directly
                # structures will be listed as nodes
                try:
                    setattr(self, key, hdf5_server[key])
                except Exception as e:
                    self.logger.exception('Failed to load "{}"'.format(key), exc_info=e)
                    setattr(self, key, self._generic_from_hdf(hdf5_server, group_name=key))

            for key in hdf5_server.list_groups():
                # Groups are more complex data types with their own depth
                # For now we only treat other IODicts and Atoms (i.e. structures) explicitly.
                setattr(self, key, self._generic_from_hdf(hdf5_server, group_name=key))


class InputDictionary(IODictionary):
    """
    An ``IODictionary`` which is instantiated with a child dictionary to store default values. If a requested item
    can't be found in this dictionary, a default value is sought.
    """

    def __init__(self):
        super(InputDictionary, self).__init__()
        self.default = IODictionary()

    def __getitem__(self, item):
        try:
            return super(InputDictionary, self).__getitem__(item)
        except (KeyError, IndexError):
            return self.default.__getitem__(item)

    def __getattr__(self, item):
        if item == 'default':
            return object.__getattribute__(self, item)
        else:
            return super(InputDictionary, self).__getattr__(item)

    def __setattr__(self, key, value):
        if key == 'default':
            object.__setattr__(self, key, value)
        else:
            super(InputDictionary, self).__setattr__(key, value)

    def keys(self):
        both_keys = set(super(InputDictionary, self).keys()).union(self.default.keys())
        for k in both_keys:
            yield k

    def values(self):
        for k in self.keys():
            yield self[k]

    def items(self):
        # Make sure the dictionary resolve pointers
        for k in self.keys():
            # It resolves the, since we call __getitem__
            yield k, self[k]

    def __iter__(self):
        # Make sure all keys get into the ** unpacking also those from the default dictionary
        return self.keys().__iter__()


class TimelineDict(LoggerMixin, OrderedDict):
    """
        Dictionary which acts as timeline
    """

    def _parse_key(self, k):

        # leftover should contain only a number, try to parse it
        integer_matches = integer_regex.findall(k)
        if len(integer_matches) > 1:
            self.logger.warning('More than one integer was found. I\'ll take the first one')
            integer_matches = [integer_matches[0]]

        try:
            result = int(integer_matches[0])
        except:
            raise KeyError(k)
        else:
            return result

    def keys(self):
        for k in super(TimelineDict, self).keys():
            yield TIMELINE_DICT_KEY_FORMAT.format(time=k)

    def items(self):
        for k, v in zip(self.keys(), super(TimelineDict, self).values()):
            yield k, v

    def _super_getitem(self, k):
        return super(TimelineDict, self).__getitem__(k)

    def _super_keys(self):
        return super(TimelineDict, self).keys()

    @property
    def timeline(self):
        return np.array(list(self._super_keys()))

    @property
    def data(self):
        return np.array(list(self.values()))

    @property
    def array(self):
        return np.array([
            list(self._super_keys()),
            list(self.values())
        ])

    def _check_key_type(self, key):
        if isinstance(key, str):
            time = self._parse_key(key)
        elif isinstance(key, int):
            time = key
        elif isinstance(key, float):
            self.logger.warning('Floating points number are not allowed here. They will be converted to an integer')
            time = int(key)
        else:
            raise TypeError('Only strings of format "%s", integers and floats are allowed as keys'.format(
                TIMELINE_DICT_KEY_FORMAT))
        return time

    def __setitem__(self, key, value):
        super(TimelineDict, self).__setitem__(self._check_key_type(key), value)

    def __getitem__(self, item):
        return super(TimelineDict, self).__getitem__(self._check_key_type(item))
