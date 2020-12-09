import io
import os

import numpy as np
from PIL import Image


class Data:
    """
    Data stores an instance of a data file, e.g. a single Image from a measurement
    """
    def __init__(self, source=None, data=None, filename=None, metadata=None, filetype=None, storedata=False):
        if (source is None) and (data is None):
            raise ValueError("No data given")
        if data is None:
            self.hasdata = False
        else:
            self.hasdata = True
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
        returns the data converted to a numpy array if conversion is known for the given filetype.
        Otherwise returns None.
        """
        if self.filetype.upper() in ["TIF", "TIFF"]:
            return np.array(Image.open(io.BytesIO(self.data)))
        return None

    def __repr__(self):
        return "pyiron-Data containing " + self.filename