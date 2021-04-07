import posixpath
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyiron_base import PyironFactory, DataContainer
from pyiron_contrib.atomistic.atomicrex.function_factory import FunctionFactory
from pyiron_contrib.atomistic.atomicrex.utility_functions import write_pretty_xml

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


class AbstractPotential(DataContainer):
    """Potentials should inherit from this class for hdf5 storage.
    """    
    def __init__(self, init=None, table_name="potential"):
        super().__init__(init, table_name=table_name)
    
    def copy_final_to_initial_params(self):
        raise NotImplementedError("Should be implemented in the subclass.")
    
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


class LJPotential(AbstractPotential):
    """
    Lennard Jones potential. Lacking some functionality,
    mostly implemented for testing and demonstatration purposes.
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

    def __init__(self, init=None, identifier=None, export_file=None, rho_range_factor=None, resolution=None, species=None):
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

    Defined as staticmethod because it can probably be reused for other potential types.

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