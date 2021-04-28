import posixpath
import xml.etree.ElementTree as ET
import re
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyiron_base import PyironFactory, DataContainer
from pyiron_contrib.atomistics.atomicrex.function_factory import FunctionFactory, FunctionParameterList
from pyiron_contrib.atomistics.atomicrex.utility_functions import write_pretty_xml

class ARPotFactory(PyironFactory):
    """
    Factory class providing convenient acces to fittable potential types.
    TODO: add other potential types supported by atomicrex.
    """
    @staticmethod
    def eam_potential(identifier="EAM", export_file="output.eam.fs", rho_range_factor=2.0, resolution=10000, species=["*","*"]):
        return EAMPotential(
            identifier=identifier,
            export_file=export_file,
            rho_range_factor=rho_range_factor,
            resolution=resolution,
            species=species,
        )

    @staticmethod
    def meam_potential(identifier="MEAM", export_file="meam.out", species=["*", "*"]):
        return MEAMPotential(
            identifier=identifier,
            export_file=export_file,
            species=species,
        )

    @staticmethod
    def lennard_jones_potential(sigma, epsilon, cutoff, species={"a": "*", "b": "*"}, identifier="LJ"):
        return LJPotential(sigma, epsilon, cutoff, species=species, identifier=identifier)

    @staticmethod
    def tersoff_potential(elements, param_file=None):
        return TersoffPotential(elements=elements, param_file=param_file)

    @staticmethod
    def abop_potential():
        pass


class AbstractPotential(DataContainer):
    """Potentials should inherit from this class for hdf5 storage.
    """    
    def __init__(self, init=None, table_name="potential"):
        super().__init__(init, table_name=table_name)
    
    def copy_final_to_initial_params(self):
        raise NotImplementedError("Should be implemented in the subclass.")
    
    def write_xml_file(self, directory):
        """
        Write the potential.xml file for the atomicrex job.
        Has to be defined.
        Args:
            directory (str): working directory path
        """    
        raise NotImplementedError("Has to be implemented in the subclass")
    
    def _parse_final_parameters(self, lines):
        raise NotImplementedError("Has to be implemented in the subclass")

    def _potential_as_pd_df(self, job):
        """
        Internal function used to convert a fitted potential in a pandas datafram
        that can be used with lammps calculations.
        Since the formatting is different for every type of potential
        this has to be implemented in the child classes

        Args:
            job (pyiron_job): Takes the fit job as argument, to obtain f.e. the working directory.
        """ 
        raise NotImplementedError("Should be implemented in the subclass")
    
    def _plot_final_potential(self):
        raise NotImplementedError("Should be implemented in the subclass")


class TersoffPotential(AbstractPotential):
    def __init__(self, init=None, elements=None, param_file=None, identifier="tersoff"):
        super().__init__(init=init)
        if init is None:
            if elements is not None:
                if param_file is None:
                    self.param_file = f"{''.join(elem for elem in elements)}.tersoff"
                else:
                    self.param_file = param_file

                self.elements = elements
                self.tag_dict = _get_tag_dict(elements)
                self.parameters = TersoffParameters(self.tag_dict)
            self.identifier = identifier

    def write_xml_file(self, directory):
        self.parameters._write_tersoff_file(directory)

        tersoff = ET.Element("tersoff")
        tersoff.set("id", f"{self.identifier}")
        tersoff.set("species-a", "*")
        tersoff.set("species-b", "*")

        params = ET.SubElement(tersoff, "param-file")
        params.text = "input.tersoff"

        tersoff.append(self.parameters._to_xml_element())

        filename = posixpath.join(directory, "potential.xml")
        write_pretty_xml(tersoff, filename)
    
    
    def _parse_final_parameters(self, lines):
        for l in lines:
            tag, param, value = _parse_tersoff_line(l)
            self.parameters[tag][param].final_value = value


# Parameters in the order of tersoff file format
tersoff_file_params = [
    "e1",
    "e2",
    "e3",
    "m",
    "gamma",
    "lambda3",
    "c",
    "d",
    "theta0",
    "n",
    "beta",
    "lambda2",
    "B",
    "R",
    "D",
    "lambda1",
    "A"
]

class TersoffParameters(DataContainer):
    def __init__(self, tag_dict=None):
        super().__init__()
        if tag_dict is not None:
            self._setup_parameters(tag_dict)
        
    def _setup_parameters(self, tags):
        for tag, v in tags.items():
            self[tag] = FunctionParameterList()

            # Parameters are added in the same order as they appear in lammps/tersoff format
            # If this order is kept always? this allows to iterate through it in a for loop
            # in some sitauations, instead of looking up the keys
            # 3 body params m(not fittable), gamma, lambda3, c, d, theta0
            self[tag].add_parameter("m", start_val=None, fitable=False)

            self[tag].add_parameter("gamma", start_val=None, tag=tag)
            self[tag].add_parameter("lambda3", start_val=None, tag=tag)
            self[tag].add_parameter("c", start_val=None, tag=tag)
            self[tag].add_parameter("d", start_val=None, tag=tag)
            self[tag].add_parameter("theta0", start_val=None, tag=tag)

            # 2 body params powern(only abop fittable), beta, lambda1, lambda2, A, B
            self[tag].add_parameter("n", start_val=None, fitable=False)
            self[tag].add_parameter("beta", start_val=None, fitable=v, tag=tag)
            self[tag].add_parameter("lambda2", start_val=None, fitable=v, tag=tag)
            self[tag].add_parameter("B", start_val=None, fitable=v, tag=tag)           
            # 2 and 3 body params D and R
            self[tag].add_parameter("D", start_val=None, fitable=False)
            self[tag].add_parameter("R", start_val=None, fitable=False)
            # 2 body params again
            self[tag].add_parameter("lambda1", start_val=None, fitable=v, tag=tag)
            self[tag].add_parameter("A", start_val=None, fitable=v, tag=tag)

            
    def start_vals_from_file(self, file, tag_list=None):
        with open(file) as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                elif l.startswith("#"):
                    continue
                else:
                    l = l.split()
                    tag = "".join((l[0], l[1], l[2]))
                    ## Allow to filter the file for certain tags by setting a tag_list
                    if tag_list is not None:
                        if tag not in tag_list:
                            continue

                    for i in range(3, 17):
                        param = tersoff_file_params[i]
                        try:
                            self[tag][param].start_val = float(l[i])
                        except:
                            KeyError(
                                f"Could not assign start value of tag: {tag} to parameter: {param}" 
                            )
    
    def _write_tersoff_file(self, directory, filename="input.tersoff"):
        filepath = posixpath.join(directory, filename)
        with open(filepath, "w") as f:
            f.write("# Tersoff parameter file in lammps/tersoff format written via pyiron atomicrex interface.\n")
            f.write("# Necessary to fit tersoff and abop potentials.\n#\n")
            f.write(f"# {' '.join(tersoff_file_params)}\n#\n")
            tag_re = re.compile("[A-Z][a-z]?") 
            for tag in self.keys():
                for el in tag_re.findall(tag):
                    f.write(f"{el} ")
                for i in range(3, 17):
                    param = tersoff_file_params[i]
                    f.write(f"{self[tag][param].start_val} ")
                f.write("\n")
    
    @staticmethod
    def tags_from_elements(elements):
        """
        Helper function that returns a list of tags
        from a list of elements, f.e. for the start_vals_from_file function

        Args:
            elements (list[str]): list of elements .f.e ["Si", "C"]

        Returns:
            list[str]: list of tags f.e. ["SiSiSi", "SiSiC", ...]
        """        
        return _tag_list(elements)


    def _to_xml_element(self):
        fit_dof = ET.Element("fit-dof")
        for tag, parameters in self.items():
            for param in parameters.values():
                if param.fitable:
                    fit_dof.append(param._to_xml_element())
        return fit_dof



    #TODO: Probably a good idea to flatten this data and write to_hdf and from_hdf correspondingly
#    def _flatten(self):
#        pass
#
#    def to_hdf():
#        pass
#
#    def from_hdf():
#        pass


def tersoff_to_abop_params(tersoff_parameters):
    pass

def abop_to_tersoff_params(abop_parameters):
    pass


class LJPotential(AbstractPotential):
    """
    Lennard Jones potential. Lacking some functionality,
    mostly implemented for testing and demonstatration purposes.
    TODO: Check if this still works after alls changes
    TODO: Implement missing functionality.
    """    
    def __init__(self, init=None, sigma=None, epsilon=None, cutoff=None, species=None, identifier=None):
        super().__init__(init=init)
        if init is None:
            self.sigma = sigma
            self.epsilon = epsilon
            self.cutoff = cutoff
            self.species = species
            self.identifier = identifier

    def write_xml_file(self, directory):
        """
        Internal function to create xml element.
        """        
        lj = ET.Element("lennard-jones")
        lj.set("id", f"{self.identifier}")
        lj.set("species-a", self.species["a"])
        lj.set("species-b", self.species["b"])

        sigma = ET.SubElement(lj, "sigma")
        sigma.text = f"{self.sigma}"
        eps = ET.SubElement(lj, "epsilon")
        eps.text = f"{self.epsilon}"
        cutoff = ET.SubElement(lj, "cutoff")
        cutoff.text = f"{self.cutoff}"

        fit_dof = ET.SubElement(lj, "fit-dof")
        sigma = ET.SubElement(fit_dof, "sigma")
        epsilon = ET.SubElement(fit_dof, "epsilon")

        filename = posixpath.join(directory, "potential.xml")
        write_pretty_xml(lj, filename)


class EAMPotential(AbstractPotential):
    """
    Embedded-Atom-Method potential.
    Usage: Create using the potential factory class.
    Add functions defined using the function_factory
    to self.pair_interactions, self.electron_densities
    and self.embedding_energies in dictionary style,
    using the identifier of the function as key.
    Example:
    eam.pair_interactions["V"] = morse_function

    """    

    def __init__(
        self,
        init=None,
        identifier=None,
        export_file=None,
        rho_range_factor=None,
        resolution=None,
        species=None):
        
        super().__init__(init=init)
        if init is None:
            self.pair_interactions = DataContainer(table_name="pair_interactions")
            self.electron_densities = DataContainer(table_name="electron_densities")
            self.embedding_energies = DataContainer(table_name="embedding_energies")
            self.identifier = identifier
            self.export_file = export_file
            self.rho_range_factor = rho_range_factor
            self.resolution = resolution
            self.species = species
    
    def copy_final_to_initial_params(self):
        """
        Copies final values of function paramters to start values.
        This f.e. allows to continue global with local minimization.
        """        
        for functions in (self.pair_interactions, self.electron_densities, self.embedding_energies):
            for f in functions.values():
                for param in f.parameters.values():
                    param.copy_final_to_start_value()

    def _potential_as_pd_df(self, job):
        """
        Makes the tabulated eam potential written by atomicrex usable
        for pyiron lammps jobs.
        """
        if self.export_file is None:
            raise ValueError("export_file must be set to use the potential with lammps")

        species = [el for el in job.input.atom_types.keys()]
        species_str = ""
        for s in species:
            species_str += f"{s} "

        pot = pd.DataFrame({
            "Name": f"{self.identifier}",
            "Filename": [[f"{job.working_directory}/{self.export_file}"]],
            'Model': ['Custom'],
            "Species": [species],
            "Config": [[
                "pair_style eam/fs\n",
                f"pair_coeff * * {job.working_directory}/{self.export_file} {species_str}\n",
                ]]
        })    
        return pot

    def write_xml_file(self, directory):
        """
        Internal function to convert to an xml element
        """        
        eam = ET.Element("eam")
        eam.set("id", f"{self.identifier}")
        eam.set("species-a", f"{self.species[0]}")
        eam.set("species-b", f"{self.species[1]}")

        if self.export_file:
            export = ET.SubElement(eam, "export-eam-file")
            export.set("resolution", f"{self.resolution}")
            export.set("rho-range-factor", f"{self.rho_range_factor}")
            export.text = f"{self.export_file}"

        mapping = ET.SubElement(eam, "mapping")
        functions = ET.SubElement(eam, "functions")

        for pot in self.pair_interactions.values():
            pair_interaction = ET.SubElement(mapping, "pair-interaction")
            pair_interaction.set("species-a", f"{pot.species[0]}")
            pair_interaction.set("species-b", f"{pot.species[1]}")
            pair_interaction.set("function", f"{pot.identifier}")
            functions.append(pot._to_xml_element())

        for pot in self.electron_densities.values():
            electron_density = ET.SubElement(mapping, "electron-density")
            electron_density.set("species-a", f"{pot.species[0]}")
            electron_density.set("species-b", f"{pot.species[1]}")
            electron_density.set("function", f"{pot.identifier}")
            functions.append(pot._to_xml_element())

        for pot in self.embedding_energies.values():
            embedding_energy = ET.SubElement(mapping, "embedding-energy")
            embedding_energy.set("species", f"{pot.species[0]}")
            embedding_energy.set("function", f"{pot.identifier}")
            functions.append(pot._to_xml_element())

        filename = posixpath.join(directory, "potential.xml")
        write_pretty_xml(eam, filename)

    def _parse_final_parameters(self, lines):
        """
        Internal Function.
        Parse function parameters from atomicrex output.

        Args:
            lines (list[str]): atomicrex output lines

        Raises:
            KeyError: Raises if a parsed parameter can't be matched to a function.
        """        
        for l in lines:
            identifier, param, value = _parse_parameter_line(l)
            if identifier in self.pair_interactions:
                self.pair_interactions[identifier].parameters[param].final_value = value
            elif identifier in self.electron_densities:
                self.electron_densities[identifier].parameters[param].final_value = value
            elif identifier in self.embedding_energies:
                self.embedding_energies[identifier].parameters[param].final_value = value
            else:

                raise KeyError(
                    f"Can't find {identifier} in potential, probably something went wrong during parsing.\n"
                    "Fitting parameters of screening functions probably doesn't work right now"
                )


    def plot_final_potential(self, job, filename=None):
        """
        Plots the the fitted potential. Reads the necessary data from eam.fs file (funcfl format used by lammps),
        therefore does not work if no table is written. Can be used with disabled fitting to plot functions that
        can't be plotted directly like splines.

        Args:
            job (AtomicrexJob): Instance of the job to construct filename.
            filename (str, optional): If a filename is given this eam is plotted instead of the fitted one. Defaults to None.
        """        


        # Read the cotents of the eam file
        # this is copy pasted from one of my old scripts and could probably be reworked
        elements = {}
        
        if filename is None:
            filename = f"{job.working_directory}/{self.export_file}"

        with open(filename, "r") as f:
            # skip comment lines
            for _ in range(3):
                f.readline()
            
            element_list =  f.readline().split()
            for element in element_list[1:]:
                elements[element] = {}
            
            Nrho, drho, Nr, dr, cutoff = f.readline().split()
            Nrho = int(Nrho)
            drho = float(drho)
            Nr = int(Nr)
            dr = float(dr)
            cutoff = float(cutoff)

            rho_values = np.linspace(0,Nrho*drho, Nrho, endpoint=False)
            r_values = np.linspace(0, Nr*dr, Nr, endpoint=False)

            for element in elements:
                # skip a line with unnecessary information
                f.readline()
                elements[element]["F"] = np.fromfile(f, count=Nrho, sep=" ")
                for rho_element in elements:
                    elements[element]["rho_{}{}".format(element, rho_element)] = np.fromfile(f, count = Nr, sep=" ")

            # V_ij = V_ji so it is written only once in the file => avoid attempts to read it twice
            # with a list of elements where it has been read
            # TODO: Have another look how to do this checking
            V_written = []
            for element in elements:
                for V_element in elements:
                    elementV_element = "{}{}".format(element, V_element)
                    V_elementelement = "{}{}".format(V_element, element)
                    if not elementV_element in V_written and not V_elementelement in V_written:
                        elements[element]["V_{}".format(elementV_element)] = np.fromfile(f, count = Nr, sep=" ")
                        # The tabulated values are not V(r) but V(r) * r, so they are divided by r here,
                        # with exception of the first value to prevent division by 0.
                        elements[element]["V_{}".format(elementV_element)][1:] = elements[element]["V_{}".format(elementV_element)][1:] / r_values[1:]
                        V_written.append(elementV_element)
        
        # TODO: figure out how to index ax for multiple elements
        fig, ax = plt.subplots(nrows=3*len(elements), ncols=len(elements), figsize=(len(elements)*8, len(elements)*3*6), squeeze=False)
        for i, (el, pot_dict) in enumerate(elements.items()):
            for j, (pot, y) in enumerate(pot_dict.items()):
                if pot == "F":
                    xdata = rho_values
                    xlabel = "$\\rho $ [a.u.]"
                    k = 0
                elif "rho" in pot:
                    xdata = r_values
                    xlabel = "r [$\AA$]"
                    k = 1
                elif "V" in pot:
                    xdata = r_values[1:]
                    y = y[1:]
                    xlabel = "r [$\AA$]"
                    k = 2

                ax[i+k, 0].plot(xdata, y)
                ax[i+k, 0].set(ylim=(-5,5), title=f"{el} {pot}", xlabel=xlabel)
        return fig, ax


class MEAMPotential(AbstractPotential):
    def __init__(self, init=None, identifier=None, export_file=None, species=None):
        super().__init__(init=init)
        if init is None:
            self.pair_interactions = DataContainer(table_name="pair_interactions")
            self.electron_densities = DataContainer(table_name="electron_densities")
            self.embedding_energies = DataContainer(table_name="embedding_energies")
            self.f_functions = DataContainer(table_name="f_functions")
            self.g_functions = DataContainer(table_name="g_functions")
            self.identifier = identifier
            self.export_file = export_file
            self.species = species

    def copy_final_to_initial_params(self):
        """
        Copies final values of function paramters to start values.
        This f.e. allows to continue global with local minimization.
        """        
        for functions in (
            self.pair_interactions,
            self.electron_densities,
            self.embedding_energies,
            self.f_functions,
            self.g_functions
            ):
            for f in functions.values():
                for param in f.parameters.values():
                    param.copy_final_to_start_value()
    

    def write_xml_file(self, directory):
        """
        Internal function to convert to an xml element
        """        
        meam = ET.Element("meam")
        meam.set("id", f"{self.identifier}")
        meam.set("species-a", f"{self.species[0]}")
        meam.set("species-b", f"{self.species[1]}")

        if self.export_file:
            export = ET.SubElement(meam, "export-functions")
            export.text = f"{self.export_file}"

        mapping = ET.SubElement(meam, "mapping")
        functions = ET.SubElement(meam, "functions")

        for pot in self.pair_interactions.values():
            pair_interaction = ET.SubElement(mapping, "pair-interaction")
            pair_interaction.set("species-a", f"{pot.species[0]}")
            pair_interaction.set("species-b", f"{pot.species[1]}")
            pair_interaction.set("function", f"{pot.identifier}")
            functions.append(pot._to_xml_element())

        for pot in self.electron_densities.values():
            electron_density = ET.SubElement(mapping, "electron-density")
            electron_density.set("species-a", f"{pot.species[0]}")
            electron_density.set("species-b", f"{pot.species[1]}")
            electron_density.set("function", f"{pot.identifier}")
            functions.append(pot._to_xml_element())

        for pot in self.f_functions.values():
            f_function = ET.SubElement(mapping, "f-function")
            f_function.set("species-a", f"{pot.species[0]}")
            f_function.set("species-b", f"{pot.species[1]}")
            f_function.set("function", f"{pot.identifier}")
            functions.append(pot._to_xml_element())
        
        for pot in self.g_functions.values():
            g_function = ET.SubElement(mapping, "g-function")
            g_function.set("species-a", f"{pot.species[0]}")
            g_function.set("species-b", f"{pot.species[1]}")
            g_function.set("species-c", f"{pot.species[2]}")
            g_function.set("function", f"{pot.identifier}")
            functions.append(pot._to_xml_element())

        for pot in self.embedding_energies.values():
            embedding_energy = ET.SubElement(mapping, "embedding-energy")
            embedding_energy.set("species", f"{pot.species[0]}")
            embedding_energy.set("function", f"{pot.identifier}")
            functions.append(pot._to_xml_element())

        filename = posixpath.join(directory, "potential.xml")
        write_pretty_xml(meam, filename)


    def _parse_final_parameters(self, lines):
        """
        Internal Function.
        Parse function parameters from atomicrex output.

        Args:
            lines (list[str]): atomicrex output lines

        Raises:
            KeyError: Raises if a parsed parameter can't be matched to a function.
        """        
        for l in lines:
            identifier, param, value = _parse_parameter_line(l)
            if identifier in self.pair_interactions:
                self.pair_interactions[identifier].parameters[param].final_value = value
            elif identifier in self.electron_densities:
                self.electron_densities[identifier].parameters[param].final_value = value
            elif identifier in self.f_functions:
                self.f_functions[identifier].parameters[param].final_value = value
            elif identifier in self.g_functions:
                self.g_functions[identifier].parameters[param].final_value = value
            elif identifier in self.embedding_energies:
                self.embedding_energies[identifier].parameters[param].final_value = value
            else:
                raise KeyError(
                    f"Can't find {identifier} in potential, probably something went wrong during parsing.\n"
                    "Fitting parameters of screening functions probably doesn't work right now"
                )


def _parse_parameter_line(line):
    """
    Internal Function.
    Parses the function identifier, name and final value of a function parameter
    from an atomicrex output line looking like:
    EAM[EAM].V_CuCu[morse-B].delta: 0.0306585 [-0.5:0.5]
    EAM[EAM].CuCu_rho[spline].node[1].y: 2.10988 [1:3]

    TODO: Add parsing of polynomial parameters.
    Returns:
        [(str, str, float): [description]
    """        

    line = line.strip().split()
    value = float(line[1])
    info = line[0].split("].")
    identifier = info[1].split("[")[0]
    param = info[2]
    if param.startswith("node"):
        x = float(param.split("[")[1])
        param = f"node_{x}"
    else:
        param = param.rstrip(":")
    return identifier, param, value

def _tag_list(elements):
    """
    Internal helper function that returns a list of tags
    from a list of elements

    Args:
        elements (list[str]): list of elements .f.e ["Si", "C"]

    Returns:
        list[str]: list of tags f.e. ["SiSiSi", "SiSiC", ...]
    """
    return ["".join(el) for el in itertools.product(elements, repeat=3)]

def _get_tag_dict(elements):
    """
    Internal helpfer function for Tersoff and ABOP potentials.
    Uses a list of elements strings to return a dict of all tags and if these tags need 2 and 3 body parameters (True)
    or only 3 body parameters (False).
    F.e. ["Si", "C"] returns:
        {'SiSiSi': True,
        'SiSiC': False,
        'SiCSi': False,
        'SiCC': True,
        'CSiSi': True,
        'CSiC': False,
        'CCSi': False,
        'CCC': True}

    Args:
        list[str]: list of elements

    Returns:
        dict[dict]: elemen tags and 
    """
    tags = {}
    tag_list = list(itertools.product(elements, repeat=3))
    for l in tag_list:
        k = "".join(l)
        if l[1] == l[2]:
            tags[k] = True # True if 2 and 3 body
        else:
            tags[k] = False # False if only 3 body
    return tags

def _parse_tersoff_line(line):
    """
    Internal Function.
    Parses the tag, parameter and final value of a function parameter
    from an atomicrex output line looking like:

    tersoff[Tersoff].lambda1[SiSiSi]: 2.4799 [0:]
    
    Returns:
        [(str, str, float): tag, param, value
    """
    line = line.strip().split()
    value = float(line[1])
    info = line[0].split("].")[1]
    info = info.split("[")
    param = info[0]
    tag = info[1].rstrip("]:")
    return tag, param, value