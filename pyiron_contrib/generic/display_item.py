import os

import pandas
from IPython import display as IPyDisplay
from IPython.core.display import display
from matplotlib import pylab as plt
from skimage import io

class DisplayFile:

    """ Class to display a file in the given outwidget. """
    def __init__(self, file, outwidget):
        """
            Class to display different files in a notebook.

            Args:
                file (str/None): path to the file to be displayed.
                outwidget (:class:`ipywidgets.Output` widget): Will be used to display the file.
        """
        self.fig = None
        self.ax = None
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
        if filetype.lower() in ['.tif', '.tiff']:
            return self._display_tiff()
        elif filetype.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
            return self._display_img()
        elif filetype.lower() in ['.txt']:
            return self._display_txt()
        elif filetype.lower() in ['.csv']:
            return self._display_csv()
        else:
            return self._display_default()

    def _display_tiff(self):
        #plt.ioff()
        data = io.imread(self.file)
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.clear()
        self.ax.imshow(data)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        return self.fig

    def _display_txt(self):
        with open(self.file) as f:
            return f.read()

    def _display_csv(self):
        return pandas.read_csv(self.file)

    def _display_img(self):
        return IPyDisplay.Image(self.file)

    def _display_default(self):
        try:
            with open(self.file) as f:
                return f.readlines()
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
        if self.output is None:
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

