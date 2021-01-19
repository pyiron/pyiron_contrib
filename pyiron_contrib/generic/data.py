import io
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


class Data:

    """ Data stores an instance of a data file, e.g. a single Image from a measurement """
    def __init__(self, source=None, data=None, filename=None, metadata=None, filetype=None, storedata=False):
        """
            Data class to store data and associated metadata.

            Args:
                source (str/None): path to the data file
                data (object/None): object containing data
                filename (str/None): filename associated with the data object, Not used if source is given!
                metadata (dict/InputList): Dictionary of metadata associated with the data
                filetype (str): File extension associated with the type data,
                                If provided this overwrites the assumption based on the extension of the filename.
                storedata (bool): If True, data is read from source (as bytes) and stored.
        """
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
        if storedata and data is None:
            self._data = self._read_source()
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
        self.hasdata = True if self._data is not None else False

    def _read_source(self):
        with open(self.source, "rb") as f:
            content = f.read()
        return content

    @property
    def data(self):
        if self.hasdata:
            return self._data
        else:
            return self._read_source()

    def data_as_np_array(self):
        """
        Returns the data converted to a numpy array if conversion is known for the given filetype.
        Otherwise returns None.
        """
        if self.filetype.upper() in ["TIF", "TIFF"]:
            return np.array(Image.open(io.BytesIO(self.data)))
        return None

    def __repr__(self):
        """ Providing the filename of the associated data """
        return "pyiron-Data containing " + self.filename




class DisplayFile:

    """ Class to display a file in the given outwidget. """
    def __init__(self, file, outwidget):
        """
            Class to display different files in a notebook.

            Args:
                file (str/None): path to the file to be displayed.
        """
        self.file = file

    def display_file(self, file):
        """
            Display the file in the outwidget,

            Args:
                file (str): path to the file to be displayed.
        """
        self.file = file
        return self._display_file()

    def _display_file(self):
        _, filetype = os.path.splitext(self.file)
        if filetype.lower() in ['.h5', '.hdf']:
            return FileHDFio(file_name=self.file)
        if filetype.lower() in ['.json']:
            return self._display_json()
        elif filetype.lower() in ['.txt']:
            return self._display_txt()
        elif filetype.lower() in ['.csv']:
            return self._display_csv()
        elif _has_imported['PIL'] and filetype.lower() in Image.registered_extensions():
            return self._display_img()
        else:
            return self._display_default()

    def _display_txt(self):
        with open(self.file) as f:
            return f.readlines()

    def _display_json(self):
        return json.load(self.file)

    def _display_csv(self):
        return pandas.read_csv(self.file)

    def _display_img(self):
        return Image.open(self.file)

    def _display_default(self):
        try:
            return self._display_txt()
        except:
            return self.file


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
        self._FileDisplay = DisplayFile(file=None, outwidget=self.output).display_file
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