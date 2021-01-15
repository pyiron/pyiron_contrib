import os

import pandas
import json
from pyiron_base import FileHDFio, ImportAlarm
_has_imported = {}
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
    import_alarm = ImportAlarm()
else:
    _not_imported = ''
    for k, j in _has_imported.items():
        if j and len(_not_imported) > 0:
            _not_imported += ', '
        if j:
            _not_imported += k
    import_alarm = ImportAlarm(
        "Reduced functionality, since " + _not_imported + " could not be imported."
                               )


class DisplayFile:

    """ Class to display a file in the given outwidget. """
    def __init__(self, file, outwidget):
        """
            Class to display different files in a notebook.

            Args:
                file (str/None): path to the file to be displayed.
                outwidget (:class:`ipywidgets.Output` widget): Will be used to display the file.
        """
        self.file = file

    def display_file(self, file):
        """
            Display the file in the outwidget,

            Args:
                file (str): path to the file to be displayed.
                outwidget (:class:`ipywidgets.Output` widget / None): New output widget to be used to display the file.
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
