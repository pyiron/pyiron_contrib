import xml.etree.ElementTree as ET
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