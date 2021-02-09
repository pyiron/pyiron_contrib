import xml.etree.ElementTree as ET

from pyiron_base import InputList


class ARFitProperty(InputList):
    def __init__(
            self,
            prop=None,
            target_value=None,
            fit=None,
            relax=None,
            relative_weight=None,
            residual_style=None,
            output=None,
            tolerance=None,
            min_val=None,
            max_val=None,
            *args,
            **kwargs
    ):
        super().__init__(table_name="fit_property", *args, **kwargs)
        self._prop = None
        self._residual_style = None
        if prop is not None:
            self.prop = prop
        if residual_style is not None:
            self.residual_style = residual_style
        self.target_value = target_value
        self.fit = fit
        self.relax = relax
        self.relative_weight = relative_weight
        self.tolerance = tolerance
        self.min_val = min_val
        self.max_val = max_val
        self.final_value = None


    @property
    def prop(self):
        return self._prop

    @prop.setter
    def prop(self, prop):
        fittable_properties = [
            "atomic-energy",
            "atomic-forces",
            "bulk-modulus",
            "pressure",
        ]
        #cij_list = [f"c{i}{j}" for i, j in range(1,7) if j>=i]
        #fittable_properties.extend(cij_list)
        if prop in fittable_properties:
            self._prop = prop
        else:
            raise ValueError(f"prop should be one of {fittable_properties}")

    @property
    def residual_style(self):
        return self._residual_style

    @residual_style.setter
    def residual_style(self, residual_style):
        res_styles = ["squared", "squared-relative", "absolute-diff"]
        if residual_style in res_styles:
            self._residual_style = residual_style
        else:
            raise ValueError(f"residual style has to be one of {res_styles}")

    def _is_scalar(self):
        if self.prop != "atomic-forces":
            return True
        else:
            return False

    def to_xml_element(self):
        xml = ET.Element(f"{self.prop}")
        xml.set("fit", f"{self.fit}".lower())
        #xml.set("relax", f"{self.relax}".lower())
        xml.set("relative-weight", f"{self.relative_weight}")
        if self._is_scalar():
            xml.set("target", f"{self.target_value}")
            if self.tolerance is not None:
                xml.set("tolerance", f"{self.tolerance}")
            if self.min_val is not None:
                xml.set("min", f"{self.min_val}")
            if self.max_val is not None:
                xml.set("min", f"{self.max_val}")
        return xml

    @staticmethod
    def _parse_final_value(line):
        if line.startswith("atomic-forces avg/max"):
            return "atomic-forces", None
        else:
            line = line.split()
            return line[0].rstrip(":"), float(line[1])



class ARFitPropertyList(InputList):
    def __init__(self, *args, **kwargs):
        super().__init__(table_name="fit_property", *args, **kwargs)

    def add_FitProperty(
            self,
            prop,
            target_value,
            fit=True,
            relax=False,
            relative_weight=1,
            residual_style="squared-relative",
            output=True,
            tolerance=None,
            min_val=None,
            max_val=None,
    ):
        self[prop] = ARFitProperty(
            prop = prop,
            target_value = target_value,
            fit = fit,
            relax = relax,
            relative_weight = relative_weight,
            residual_style = residual_style,
            output = output,
            tolerance = tolerance,
            min_val = min_val,
            max_val = max_val,
        )

    def to_xml_element(self):
        properties = ET.Element("properties")
        for p in self.values():
            properties.append(p.to_xml_element)