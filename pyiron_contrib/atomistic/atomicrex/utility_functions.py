import xml.etree.ElementTree as ET
from xml.dom import minidom

def write_pretty_xml(elem, filename):
    ugly = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(ugly)
    reparsed = reparsed.toprettyxml(indent="    ")
    with open(filename, "w") as f:
        f.write(reparsed)
    return 