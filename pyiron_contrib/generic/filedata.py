import json
import os
import numpy as np
import pandas

from pyiron_base import ImportAlarm, FileHDFio

_has_imported = {}
import_alarm = ImportAlarm()
_not_imported = ''
try:
    from PIL import Image
    _has_imported['PIL'] = True
except ImportError:
    _has_imported['PIL'] = False
try:
    from IPython.core.display import display
    _has_imported['IPython'] = True
except ImportError:
    _has_imported['IPython'] = False
if all(_has_imported.values()):
    pass
else:
    for k, j in _has_imported.items():
        if j and len(_not_imported) > 0:
            _not_imported += ', '
        if j:
            _not_imported += k
    import_alarm = ImportAlarm(
        "Reduced functionality, since " + _not_imported + " could not be imported."
    )




class LoadFile:

    """ Class to load a file into an appropriate object. """

    def __call__(self, file):
        return self.load_file(file)

    def load_file(self, file):
        """
            Load the file and return an appropriate object containing the data.

            Args:
                file (str): path to the file to be displayed.
        """
        _, filetype = os.path.splitext(file)
        if filetype.lower() in ['.h5', '.hdf']:
            return FileHDFio(file_name=file)
        if filetype.lower() in ['.json']:
            return self._load_json(file)
        elif filetype.lower() in ['.txt']:
            return self._load_txt(file)
        elif filetype.lower() in ['.csv']:
            return self._load_csv(file)
        elif _has_imported['PIL'] and filetype.lower() in Image.registered_extensions():
            return self._load_img(file)
        else:
            return self._load_default(file)

    @staticmethod
    def _load_txt(file):
        with open(file) as f:
            return f.readlines()

    @staticmethod
    def _load_json(file):
        with open(file) as f:
            return json.load(f)

    @staticmethod
    def _load_csv(file):
        return pandas.read_csv(file)

    @staticmethod
    def _load_img(file):
        return Image.open(file)

    def _load_default(self, file):
        try:
            return self._load_txt(file)
        except:
            return file


class FileData:

    """ FileData stores an instance of a data file, e.g. a single Image from a measurement """
    def __init__(self, source=None, data=None, filename=None, metadata=None, filetype=None):
        """
            FileData class to store data and associated metadata.

            Args:
                source (str/None): path to the data file
                data (object/None): object containing data
                filename (str/None): filename associated with the data object, Not used if source is given!
                metadata (dict/InputList): Dictionary of metadata associated with the data
                filetype (str): File extension associated with the type data,
                                If provided this overwrites the assumption based on the extension of the filename.
        """
        self.file_loader = LoadFile()
        if (source is None) and (data is None):
            raise ValueError("No data given")
        if data is not None:
            self._data = data
        self.measurement = None
        if source is not None:
            self.filename = os.path.split(source)[1]
            self.source = source
        elif filename is None:
            raise ValueError("No filename given")
        else:
            self.filename = filename
        if (filetype is None) and (self.filename is not None):
            filetype = os.path.splitext(self.filename)[1]
            if len(filetype[1:]) == 0:
                self.filetype = None
            else:
                self.filetype = filetype[1:]
        else:
            self.filetype = filetype
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
        self._hasdata = True if self._data is not None else False

    @property
    def data(self):
        if self._hasdata:
            return self._data
        else:
            return self.file_loader(self.source)

    def data_as_np_array(self):
        """
        Try to return the data converted to a numpy array. Returns None if failed.
        """
        try:
            return np.asarray(self.data)
        except:
            return None



class DisplayItem:
    """Class to display an item in an appropriate fashion."""
    def __init__(self, item=None, outwidget=None):
        """
            Class to display different files in a notebook.

                Args:
                item (object/None): item to be displayed.
                outwidget (:class:`ipywidgets.Output` widget): Will be used to display the file.
        """
        self.output = outwidget
        self.item = item
        self._FileDisplay = LoadFile()
        if item is not None and outwidget is not None:
            self._display()

    def display(self, item=None, outwidget=None):
        if item is not None:
            self.item = item
        if outwidget is not None:
            self.output = outwidget
        return self._display()

    def _display(self):
        if isinstance(self.item, str):
            if os.path.isfile(self.item):
                obj = self._display_file()
            else:
                obj = self._display_obj()
        elif str(type(self.item)) == "<class 'boto3.resources.factory.s3.Object'>":
            obj = self._display_s3_metadata()
        else:
            obj = self._display_obj()
        if self.output is None or not _has_imported['IPython']:
            return obj
        else:
            with self.output:
                display(obj)

    def _display_s3_metadata(self):
        metadata_str  = "Metadata:\n"
        metadata_str += "------------------------\n"
        for key, value in self.item.metadata.items():
            metadata_str += key + ': ' + value +'\n'
        return metadata_str

    def _display_file(self):
        return self._FileDisplay(self.item)

    def _display_obj(self):
        return self.item