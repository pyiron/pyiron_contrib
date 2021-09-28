import xml.etree.ElementTree as ET

from pyiron_base import DataContainer

class Constraint:
    __version__ = "0.1.0"
    __hdf_version__ = "0.1.0"
    def __init__(self, identifier=None, dependent_dof=None, expression=None) -> None:
        self._storage = DataContainer(table_name="_storage")
        self.identifier = identifier
        self.dependent_dof = dependent_dof
        self.expression = expression

    @property
    def identifier(self):
        return self._storage["identifier"]

    @identifier.setter
    def identifier(self, identifier):
        self._storage["identifier"] = identifier

    @property
    def dependent_dof(self):
        return self._storage["dependent-dof"]

    @dependent_dof.setter
    def dependent_dof(self, dependent_dof):
        self._storage["dependent_dof"] = dependent_dof

    @property
    def expression(self):
        return self._storage["expression"]

    @property
    def comment(self):
        return self._storage["comment"]
    
    @expression.setter
    def expression(self, expression):
        self._storage["expression"] = expression

    def to_hdf(self, hdf, group_name=None):
        with hdf.open(group_name) as h:
            self._type_to_hdf(hdf=h)
            self._storage.to_hdf(hdf=h)

    def from_hdf(self, hdf, group_name=None):
        self._storage.from_hdf(hdf=hdf)

    def _repr_json_(self):
        return self._storage._repr_json_()
    
    def _type_to_hdf(self, hdf):
        """
        Internal helper function to save type and version in hdf root

        Args:
            hdf (ProjectHDFio): HDF5 group object
        """
        hdf["NAME"] = self.__class__.__name__
        hdf["TYPE"] = str(type(self))
        hdf["VERSION"] = self.__version__
        hdf["HDF_VERSION"] = self.__hdf_version__

    
    def _to_xml_element(self):
        root = ET.Element("constraint")
        root.set("id", self.identifier)
        root.set("dependent-dof", self.dependent_dof)
        expr = ET.SubElement(root, "expression")
        expr.text = self.expression
        return root

class ParameterConstraints:
    __version__ = "0.1.0"
    __hdf_version__ = "0.1.0"
    def __init__(self) -> None:
        self._storage = DataContainer(table_name="_storage")

    def add_constraint(self, identifier, dependent_dof, expression):
        self._storage[identifier] = Constraint(identifier=identifier, dependent_dof=dependent_dof, expression=expression)

    def to_hdf(self, hdf, group_name="parameter_constraints"):
        with hdf.open(group_name) as h:
            self._type_to_hdf(hdf=h)
            self._storage.to_hdf(hdf=h)

    def _type_to_hdf(self, hdf):
        """
        Internal helper function to save type and version in hdf root

        Args:
            hdf (ProjectHDFio): HDF5 group object
        """
        hdf["NAME"] = self.__class__.__name__
        hdf["TYPE"] = str(type(self))
        hdf["VERSION"] = self.__version__
        hdf["HDF_VERSION"] = self.__hdf_version__

    def from_hdf(self, hdf, group_name="parameter_constraints"):
        with hdf.open(group_name) as h:
            self._storage.from_hdf(hdf=h)

    def _repr_json_(self):
        return self._storage._repr_json_()

    def _to_xml_element(self):
        root = ET.Element("parameter-constraints")
        for c in self._storage.values():
            root.append(c._to_xml_element())
        return root

