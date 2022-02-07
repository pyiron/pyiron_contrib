import xml.etree.ElementTree as ET
import sys
import os
import threading
from xml.dom import minidom
import time
import select
import ctypes


def write_pretty_xml(elem, filename):
    """
    The files directly written with xml.etree.ElemenTrees contain no newlines and indents.
    This is a helper function to write them in a human readable way.

    Args:
        elem (ElementTree): xml.etree.ElementTree ElementTree object to be written
        filename (string): name for the xml file
    """
    ugly = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(ugly)
    reparsed = reparsed.toprettyxml(indent="    ")
    with open(filename, "w") as f:
        f.write(reparsed)
    return


"""
This was supposed to catch the output of atomicrex when running it using its python interface.
It never worked completely but maybe this will be useful for some other code

class OutputCatcher():
    libc = ctypes.CDLL(None)
    c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

    def __init__(self, filename, threaded=False):
        sys.stdout.flush()
        self.filename = filename
        #self.pipe_in, self.pipe_out = os.pipe()
        ## Maybe use tempfile instead?
        self.file_out = os.open(self.filename, os.O_WRONLY|os.O_TRUNC|os.O_CREAT)
        self.threaded = threaded
        self.libc.fflush(self.c_stdout)

    def __enter__(self):
        self.stdout_old = os.dup(1)
        self.stdout_new = os.dup(1)
        os.dup2(self.file_out, 1)
        sys.stdout = os.fdopen(self.stdout_new, 'w')
        if self.threaded:
            self.worker = threading.Thread(target=self.read_output)
            self.worker.start()
            time.sleep(0.01)
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.close()
        if self.threaded:
            self.worker.join()
        else:
            self.read_output()
        self.libc.fflush(self.c_stdout)
        os.dup2(self.stdout_old, 1)
        os.close(self.stdout_old)
        os.close(self.file_out)

    def read_output(self):
        r, _, _ = select.select([self.stdout_new], [], [], 0)
        while bool(r):
            char = os.read(self.stdout_new, 1024)
            self.file_out.write(char)
            r, _, _ = select.select([self.stdout_new], [], [], 0)
"""
