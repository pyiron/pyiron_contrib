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
        self.output = outwidget
        self.fig = None
        self.ax = None
        self.file = file
        if file is not None:
            self._display_file()

    def display_file(self, file, outwidget=None):
        """
            Display the file in the outwidget,

            Args:
                file (str): path to the file to be displayed.
                outwidget (:class:`ipywidgets.Output` widget / None): New output widget to be used to display the file.
        """
        if outwidget is not None:
            self.output = outwidget
        self.file = file
        self._display_file()

    def _display_file(self):
        _, filetype = os.path.splitext(self.file)
        if filetype.lower() in ['.tif', '.tiff']:
            self._display_tiff()
        elif filetype.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
            self._display_img()
        elif filetype.lower() in ['.txt']:
            self._display_txt()
        elif filetype.lower() in ['.csv']:
            self._display_csv()
        else:
            self._display_default()

    def _display_tiff(self):
        plt.ioff()
        data = io.imread(self.file)
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax.clear()
        self.ax.imshow(data)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        with self.output:
            display(self.fig)

    def _display_txt(self):
        with self.output:
            with open(self.file) as f:
                print(f.read(), end='')

    def _display_csv(self):
        with self.output:
            display(pandas.read_csv(self.file))

    def _display_img(self):
        with self.output:
            display(IPyDisplay.Image(self.file))

    def _display_default(self):
        with self.output:
            with open(self.file) as f:
                print(f.readlines())


class DisplayMetadata:

    """ Class to display metadata of a file in the given outwidget. """
    def __init__(self, metadata, outwidget):
        """
            Display the metadata in the outwidget.

            Args:
                metadata (dict/None): Metadata to be displayed.
                outwidget (:class:`ipywidgets.Output` widget): New output widget to be used to display the metadata.
        """
        self.output = outwidget
        self.metadata = metadata
        if metadata is not None:
            self._display_metadata()

    def display(self, metadata, outwidget=None):
        """
            Display the metadata in the outwidget

            Args:
                metadata (dict): Metadata to be displayed.
                outwidget (:class:`ipywidgets.Output` widget / None): New output widget to be used to display the metadata.
        """
        self.metadata = metadata
        if outwidget is not None:
            self.output = outwidget
        self._display_metadata()

    def _display_metadata(self):
        with self.output:
            print("Metadata:")
            print("------------------------")
            for key, value in self.metadata.items():
                print(key + ': ' + value)


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
        if outwidget is None:
            try:
                import ipywidgets as widgets
                self.output = widgets.Output()
                display(self.output)
            except ImportError:
                self.output = None
        self._FileDisplay = DisplayFile(file=None, outwidget=self.output).display_file
        self._MetadataDisplay = DisplayMetadata(metadata=None, outwidget=self.output).display
        if item is not None:
            self._display()

    def display(self, item=None, outwidget=None):
        if item is not None:
            self.item = item
        if outwidget is not None:
            self.output = outwidget
        self._display()

    def _display(self):
        if self.output is None:
            return self._fallback_display()
        self.output.clear_output()
        if isinstance(self.item, str):
            if os.path.isfile(self.item):
                self._display_file()
            else:
                self._display_obj()
            return
        if str(type(self.item)) == "<class 'boto3.resources.factory.s3.Object'>":
            self._display_s3_metadata()
        else:
            self._display_obj()

    def _display_s3_metadata(self):
        self._MetadataDisplay(self.item.metadata, self.output)

    def _display_file(self):
        self._FileDisplay(self.item, self.output)

    def _display_obj(self):
        with self.output:
            display(self.item)

    def _fallback_display(self):
        print('fallback')
        if isinstance(self.item, str):
            try:
                with open(self.item) as f:
                    print(f.readlines())
            except FileNotFoundError:
                print(self.item)
        else:
            print(self.item)
