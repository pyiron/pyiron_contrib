import xml.etree.ElementTree as ET
import sys
import os
from xml.dom import minidom

def write_pretty_xml(elem, filename):
    """
    The files directly written with xml.etree.ElemenTrees contain no newlines and indents. 
    This is a helper function to write them in a human readable way.

    Args:
        elem (ElementTree): xml.etree.ElementTree ElementTree object to be written
        filename (string): name for the xml file
    """    
    ugly = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(ugly)
    reparsed = reparsed.toprettyxml(indent="    ")
    with open(filename, "w") as f:
        f.write(reparsed)
    return

class OutputCatcher():
    def __init__(self, filename):
        self.filename = filename
        sys.stdout.flush()
        self.file_out = os.open(self.filename, os.O_WRONLY|os.O_TRUNC|os.O_CREAT)

    def __enter__(self):
        self.stdout_old = os.dup(1)
        self.stdout_new = os.dup(1)
        os.dup2(self.file_out, 1)
        sys.stdout = os.fdopen(self.stdout_new, 'w')
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        os.dup2(self.stdout_old, 1)
        os.close(self.stdout_old)
        os.close(self.file_out)