import xml.etree.ElementTree as ET

from pyiron_base import DataContainer


class Constraint(DataContainer):
    def __init__(self, identifier=None, dependent_dof=None, expression=None) -> None:
        super().__init__(table_name=f"constraint_{identifier}")
        self.identifier = identifier
        self.dependent_dof = dependent_dof
        self.expression = expression

    def _to_xml_element(self):
        root = ET.Element("constraint")
        root.set("id", self.identifier)
        root.set("dependent-dof", self.dependent_dof)
        expr = ET.SubElement(root, "expression")
        expr.text = self.expression
        return root


class ParameterConstraints(DataContainer):
    def __init__(self, table_name="parameter_constraints") -> None:
        super().__init__(table_name=table_name)

    def add_constraint(self, identifier, dependent_dof, expression):
        self[identifier] = Constraint(
            identifier=identifier, dependent_dof=dependent_dof, expression=expression
        )

    def _to_xml_element(self):
        root = ET.Element("parameter-constraints")
        for c in self.values():
            root.append(c._to_xml_element())
        return root
