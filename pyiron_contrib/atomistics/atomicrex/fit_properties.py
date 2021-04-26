import xml.etree.ElementTree as ET

import numpy as np

from pyiron_base import DataContainer


class ARFitProperty(DataContainer):
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
            output_all=None,
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
        self.output_all = output_all

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

        if self.fit:
            xml.set("fit", "true")
        else:
            xml.set("fit", "false")
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
            if self.output_all:
                xml.set("output-all", "true")
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


class ARFitPropertyList(DataContainer):
    """
    DataContainer of ARFitProperties that additionally provides utility functions
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
            output_all=True,
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
            min_val (float, optional): Only scalar properties, if relaxation enabled. Defaults to None.
            max_val (float, optional): Only scalar properties, if relaxation enabled. Defaults to None.
            output_all (bool, optional): Only vector properties. Determines if full vector is written to output. Defaults to True.

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
            output_all=output_all,
        )

    def to_xml_element(self):
        """Internal helper function converting the list into an atomicrex xml element.
        """        
        properties = ET.Element("properties")
        for p in self.values():
            properties.append(p.to_xml_element)


Residual_Styles = [
"squared",
"squared-relative",
"absolute-diff"
]


class FlattenedARProperty:
    """
    Class to read and write scalar properties of a structure, f.e. the energy.
    """    
    def __init__(self, num_structures, prop):
        self.num_structures = num_structures
        self.prop = prop
        self._init_arrays()
    

    def _init_arrays(self):
        self.target_value = np.full(self.num_structures, np.nan)
        self.fit = np.full(self.num_structures, fill_value=False, dtype=bool)
        self.relative_weight = np.ones(self.num_structures) # default 1
        # resiudal styles are "squared", "squared-relative" and "absolute-diff"
        # Translate to 0, 1 and 2
        self.residual_style = np.zeros(self.num_structures, dtype=np.uint8) # default 0
        self.relax = np.full(self.num_structures, fill_value=False, dtype=bool) # default False
        self.tolerance = np.full(self.num_structures, fill_value=np.nan)
        self.min_val = np.full(self.num_structures, fill_value=np.nan)
        self.max_val = np.full(self.num_structures, fill_value=np.nan)
        self.final_value = np.full(self.num_structures, fill_value=np.nan)
        self.output = np.full(self.num_structures, fill_value=False, dtype=bool)

    def to_hdf(self, hdf, group_name=None):
        if group_name is None:
            group_name = f"{self.property}"
        with hdf.open(group_name) as h:
            h["target_value"] = self.target_value
            h["fit"] = self.fit
            h["relative_weight"] = self.relative_weight
            h["residual_style"] = self.residual_style
            h["relax"] = self.relax
            h["tolerance"] = self.tolerance
            h["min_val"] = self.min_val
            h["max_val"] = self.max_val
            h["final_value"] = self.final_value
            h["output"] = self.output

    #def _type_to_hdf(self, hdf):
    #    """
    #    Internal helper function to save type and version in hdf root#
    #
    #    Args:
    #        hdf (ProjectHDFio): HDF5 group object
    #    """
    #    hdf["TYPE"] = str(type(self))

    def from_hdf(self, hdf, group_name=None):
        if group_name is None:
            group_name = f"{self.prop}"
        with hdf.open(group_name) as h:
            self.target_value = h["target_value"]
            self.fit = h["fit"]
            self.relative_weight = h["relative_weight"]
            self.residual_style = h["residual_style"]
            self.relax = h["relax"]
            self.tolerance = h["tolerance"]
            self.min_val = h["min_val"]
            self.max_val = h["max_val"]
            self.output = h["output"]
            self.final_value = h["final_value"]

    def to_xml_element(self, index):
        xml = ET.Element(f"{self.prop}")
        if self.output[index]:
            xml.set("output", "true")
        if self.fit[index]:
            xml.set("fit", "true")
            xml.set("target", f"{self.target_value[index]}")
            #xml.set("relax", f"{self.relax}".lower())
            xml.set("relative-weight", f"{self.relative_weight[index]}")
            xml.set("residual-style", f"{Residual_Styles[self.residual_style[index]]}")
            if not np.isnan(self.tolerance[index]):
                xml.set("tolerance", f"{self.tolerance[index]}")
            if not np.isnan(self.min_val[index]):
                xml.set("min", f"{self.min_val[index]}")
            if not np.isnan(self.max_val[index]):
                xml.set("min", f"{self.max_val[index]}")
        return xml


class FlattenedARVectorProperty:
    """
    Like AR property, but for vector properties, i.e. forces
    """
    def __init__(self, num_structures, num_atoms, prop):
        self.num_structures = num_structures
        self.num_atoms = num_atoms
        self.prop = prop
        self._init_arrays()

    
    def _init_arrays(self):
        self.target_value = np.full((self.num_atoms, 3), np.nan)
        self.fit = np.full(self.num_structures, fill_value=False, dtype=bool)
        self.relative_weight = np.ones(self.num_structures) # default 1
        # resiudal styles are "squared", "squared-relative" and "absolute-diff"
        # Translate to 0, 1 and 2
        self.residual_style = np.zeros(self.num_structures, dtype=np.uint8) # default 0
        self.relax = np.full(self.num_structures, fill_value=False, dtype=bool) # default False
        self.tolerance = np.full(self.num_structures, fill_value=np.nan)
        self.final_value = np.full((self.num_atoms, 3), fill_value=np.nan)
        self.output = np.full(self.num_structures, fill_value=False, dtype=bool)

    def to_hdf(self, hdf, group_name=None):
        if group_name is None:
            group_name = f"{self.prop}"
        with hdf.open(group_name) as h:
            h["target_value"] = self.target_value
            h["fit"] = self.fit
            h["relative_weight"] = self.relative_weight
            h["residual_style"] = self.residual_style
            h["relax"] = self.relax
            h["tolerance"] = self.tolerance
            h["final_value"] = self.final_value
            h["output"] = self.output

    def from_hdf(self, hdf, group_name=None):
        if group_name is None:
            group_name = f"{self.prop}"
        with hdf.open(group_name) as h:
            self.target_value = h["target_value"]
            self.fit = h["fit"]
            self.relative_weight = h["relative_weight"]
            self.residual_style = h["residual_style"]
            self.relax = h["relax"]
            self.tolerance = h["tolerance"]
            self.final_value = h["final_value"]
            self.output = h["output"]
    
#    def _type_to_hdf(self, hdf):
#        """
#        Internal helper function to save type and version in hdf root
#
#        Args:
#            hdf (ProjectHDFio): HDF5 group object
#        """
#        hdf["TYPE"] = str(type(self))
    
    def to_xml_element(self, index):
        xml = ET.Element(f"{self.prop}")
        if self.output[index]:
            xml.set("output-all", "true")
        if self.fit[index]:
            xml.set("output-all", "true")
            xml.set("fit", "true")
            #xml.set("relax", f"{self.relax}".lower())
            xml.set("relative-weight", f"{self.relative_weight[index]}")
            xml.set("residual-style", f"{Residual_Styles[self.residual_style[index]]}")
            if not np.isnan(self.tolerance[index]):
                xml.set("tolerance", f"{self.tolerance[index]}")
        return xml

