import xml.etree.ElementTree as ET

from pyiron_base import InputList


class ARFitProperty(InputList):
    """
    Class to describe properties that can be fitted using atomicrex.
    Property and target value have to be given,
    the other parameters control details of the fitting procedure.
    For more information what they do visit the atomicrex documentation.
    
    Direct interaction with this class shouldn't be necessary.
    To conveniently create ARFitProperty objects use the
    ARFitPropertList add_fit_property method.
    """    
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
        """Only allow properties that can be fitted using atomicrex

        Args:
            prop (string): name of the property

        Raises:
            ValueError: Given property can not be fitted using atomicrex.
        """    
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
        """Residual styles available in atomicrex

        Args:
            residual_style (string): [description]

        Raises:
            ValueError: [description]
        """        
        res_styles = ["squared", "squared-relative", "absolute-diff"]
        if residual_style in res_styles:
            self._residual_style = residual_style
        else:
            raise ValueError(f"residual style has to be one of {res_styles}")

    def _is_scalar(self):
        """
        Internal helper function, scalar properties are provided in the structure xml
        and vector properties (atomic forces) in the structure file
        Returns:
            [bool]: True if scalar, False if vector property.
        """        
        if self.prop != "atomic-forces":
            return True
        else:
            return False

    def to_xml_element(self):
        xml = ET.Element(f"{self.prop}")
        xml.set("fit", f"{self.fit}".lower())
        #xml.set("relax", f"{self.relax}".lower())
        xml.set("relative-weight", f"{self.relative_weight}")
        if self.tolerance is not None:
            xml.set("tolerance", f"{self.tolerance}")
        
        if self._is_scalar():
            xml.set("target", f"{self.target_value}")
            if self.min_val is not None:
                xml.set("min", f"{self.min_val}")
            if self.max_val is not None:
                xml.set("min", f"{self.max_val}")
            xml.set("residual-style", f"{self.residual_style}")
        else:
            if self.residual_style == "squared-relative":
                raise ValueError("Squared-relative residual style is not implemented for forces in atomicrex")
            if self.min_val is not None or self.max_val is not None:
                raise ValueError("Min and Max val can only be given for scalar properties")
            else:
                xml.set("residual-style", f"{self.residual_style}")
        return xml

    @staticmethod
    def _parse_final_value(line):
        """
        Parses the final values of properties used in the fitting process from
        atomicrex output.

        Args:
            line (string): string from atomicrex output containing final value of some property

        Returns:
            [(string, float)]: property, final value
        """        
        if line.startswith("atomic-forces avg/max"):
            return "atomic-forces", None
        else:
            line = line.split()
            return line[0].rstrip(":"), float(line[1])



class ARFitPropertyList(InputList):
    """
    InputList of ARFitProperties that additionally provides utility functions
    that allow convenient addition of fit properties to a structure.
    Also provides internal functionality.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(table_name="fit_property", *args, **kwargs)

    def add_FitProperty(
            self,
            prop,
            target_value,
            fit=True,
            relax=False,
            relative_weight=1,
            residual_style="squared",
            output=True,
            tolerance=None,
            min_val=None,
            max_val=None,
    ):
        """
        Adds a fittable property to the fit properties
        of an atomicrex structure.
        Default values should be ok for most cases,
        but it is strongly recommended to check the purpose
        of each argument in the atomicrex documentation.

        Args:
            prop (string): property to be fitted
            target_value (float):
            fit (bool, optional): Include the property in the fit procedure. Defaults to True.
            relax (bool, optional): Calculate before or after structure relaxation. Defaults to False.
            relative_weight (int, optional): Weight for the objective function. Defaults to 1.
            residual_style (str, optional): See atomicrex documentation. Defaults to "squared-relative".
            output (bool, optional): Determines if the value is written to output.
            Could cause parsing problems if False. Defaults to True.
            tolerance (float, optional): See atomicrex documentation. Defaults to None.
            min_val ([type], optional): [description]. Defaults to None.
            max_val ([type], optional): [description]. Defaults to None.
        """    
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
        """Internal helper function converting the list into an atomicrex xml element.
        """        
        properties = ET.Element("properties")
        for p in self.values():
            properties.append(p.to_xml_element)