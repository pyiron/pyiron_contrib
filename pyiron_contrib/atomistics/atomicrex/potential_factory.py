import posixpath
import xml.etree.ElementTree as ET
import re
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyiron_base import PyironFactory, DataContainer
from pyiron_contrib.atomistics.atomicrex.function_factory import (
    FunctionFactory,
    FunctionParameterList,
)
from pyiron_contrib.atomistics.atomicrex.utility_functions import write_pretty_xml


class ARPotFactory(PyironFactory):
    """
    Factory class providing convenient acces to fittable potential types.
    TODO: add other potential types supported by atomicrex.
    """

    @staticmethod
    def eam_potential(
        identifier="EAM",
        export_file="output.eam.fs",
        rho_range_factor=2.0,
        resolution=10000,
        species=["*", "*"],
    ):
        return EAMPotential(
            identifier=identifier,
            export_file=export_file,
            rho_range_factor=rho_range_factor,
            resolution=resolution,
            species=species,
        )

    @staticmethod
    def adp_potential(
        identifier="ADP",
        export_file="output.adp.fs",
        rho_range_factor=2.0,
        resolution=10000,
        species=["*", "*"],
    ):
        return ADPotential(
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
    def lennard_jones_potential(
        sigma, epsilon, cutoff, species={"a": "*", "b": "*"}, identifier="LJ"
    ):
        return LJPotential(
            sigma, epsilon, cutoff, species=species, identifier=identifier
        )

    @staticmethod
    def tersoff_potential(elements, export_file=None):
        return TersoffPotential(elements=elements, export_file=export_file)

    @staticmethod
    def abop_potential(elements, export_file=None):
        return ABOPPotential(elements=elements, export_file=export_file)


class AbstractPotential(DataContainer):
    """Potentials should inherit from this class for hdf5 storage."""

    def __init__(self, init=None, table_name="potential"):
        super().__init__(init, table_name=table_name)

    # def copy_final_to_initial_params(self):
    #    raise NotImplementedError("Should be implemented in the subclass.")

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


class BOPAbstract(AbstractPotential):
    def __init__(self, init=None, elements=None, export_file=None, identifier=None):
        super().__init__(init=init)
        if init is None:
            self.identifier = identifier
            if elements is not None:
                if export_file is None:
                    self.export = f"{''.join(elem for elem in elements)}.tersoff"
                else:
                    self.export_file = export_file

                self._tag_dict = _get_tag_dict(elements)
                self.elements = elements
                if self.identifier == "tersoff":
                    self.parameters = TersoffParameters(self._tag_dict)
                elif self.identifier == "abop":
                    self.parameters = ABOPParameters(self._tag_dict)

    @property
    def param_file(self):
        return self.export_file

    def write_xml_file(self, directory):
        self.parameters._write_tersoff_file(directory)

        tersoff = ET.Element(f"{self.identifier}")
        tersoff.set("id", f"{self.identifier}")
        tersoff.set("species-a", "*")
        tersoff.set("species-b", "*")

        output = ET.SubElement(tersoff, "export-potential")
        output.text = f"{self.export_file}"

        params = ET.SubElement(tersoff, "param-file")
        params.text = "input.tersoff"

        tersoff.append(self.parameters._to_xml_element())

        filename = posixpath.join(directory, "potential.xml")
        write_pretty_xml(tersoff, filename)

    def _parse_final_parameters(self, lines):
        for l in lines:
            tag, param, value = _parse_tersoff_line(l)
            self.parameters[tag][param].final_value = value

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

        pot = pd.DataFrame(
            {
                "Name": f"{self.identifier}",
                "Filename": [[f"{job.working_directory}/{self.export_file}"]],
                "Model": ["Custom"],
                "Species": [species],
                "Config": [
                    [
                        "pair_style tersoff\n",
                        f"pair_coeff * * {job.working_directory}/{self.export_file} {species_str}\n",
                    ]
                ],
            }
        )
        return pot


class TersoffPotential(BOPAbstract):
    def __init__(self, init=None, elements=None, export_file=None):
        super().__init__(
            init=init, elements=elements, export_file=export_file, identifier="tersoff"
        )


class ABOPPotential(BOPAbstract):
    def __init__(self, init=None, elements=None, export_file=None):
        super().__init__(
            init=init, elements=elements, export_file=export_file, identifier="abop"
        )


# tag regular expression
tag_re = re.compile("[A-Z][a-z]?")
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
    "A",
]


# Transform tersoff to abop parameters
def get_r0(lam1, lam2, A, B):
    return 1 / (lam1 - lam2) * np.log(lam1 * A / (lam2 * B))


def get_D0(lam1, lam2, A, B):
    return (
        A
        * (lam1 / lam2 - 1)
        * np.exp(-lam1 / (lam1 - lam2) * np.log(lam1 * A / lam2 * B))
    )


def get_beta(lam1, lam2):
    return lam1 / np.sqrt(2 * lam1 / lam2)


def get_S(lam1, lam2):
    return lam1 / lam2


# abop to tersoff parameters
def get_lam1(beta, S):
    return beta * np.sqrt(2 * S)


def get_lam2(beta, S):
    return beta * np.sqrt(2 / S)


def get_A(beta, S, D0, r0):
    return D0 / (S - 1) * np.exp(beta * np.sqrt(2 * S) * r0)


def get_B(beta, S, D0, r0):
    return D0 * S / (S - 1) * np.exp(beta * np.sqrt(2 / S) * r0)


class BOPParameters(DataContainer):
    def __init__(self, tag_dict=None):
        super().__init__()
        if tag_dict is not None:
            self._setup_parameters(tag_dict)

    def _setup_parameters(self, tag_dict):
        raise NotImplementedError("Implement in subclass")

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


class TersoffParameters(BOPParameters):
    def __init__(self, tag_dict=None):
        super().__init__(tag_dict)

    def _setup_parameters(self, tag_dict):
        for tag, v in tag_dict.items():
            self[tag] = FunctionParameterList()

            # Parameters are added in the same order as they appear in lammps/tersoff format
            # If this order is kept always? this allows to iterate through it in a for loop
            # in some sitauations, instead of looking up the keys
            # 3 body params m(not fittable), gamma, lambda3, c, d, theta0
            self[tag].add_parameter("m", start_val=3, fitable=False)  # 3 or 1

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
            f.write(
                "# Tersoff parameter file in lammps/tersoff format written via pyiron atomicrex interface.\n"
            )
            f.write("# Necessary to fit tersoff and abop potentials.\n#\n")
            f.write(f"# {' '.join(tersoff_file_params)}\n#\n")
            for tag in self.keys():
                for el in tag_re.findall(tag):
                    f.write(f"{el} ")
                for i in range(3, 17):
                    param = tersoff_file_params[i]
                    f.write(f"{self[tag][param].start_val} ")
                f.write("\n")

    def _to_xml_element(self):
        fit_dof = ET.Element("fit-dof")
        for tag, parameters in self.items():
            for param in parameters.values():
                if param.fitable:
                    fit_dof.append(param._to_xml_element())
        return fit_dof


class ABOPParameters(BOPParameters):
    def __init__(self, tag_dict=None):
        super().__init__(tag_dict)

    def _setup_parameters(self, tag_dict):
        for tag, v in tag_dict.items():
            self[tag] = FunctionParameterList()

            # 2 Body parameters, fitable only if v (see https://lammps.sandia.gov/doc/pair_tersoff.html)
            self[tag].add_parameter("D0", start_val=None, fitable=v, tag=tag)
            self[tag].add_parameter("S", start_val=None, fitable=v, tag=tag)
            self[tag].add_parameter("r0", start_val=None, fitable=v, tag=tag)
            self[tag].add_parameter("beta", start_val=None, fitable=v, tag=tag)
            self[tag].add_parameter("beta2", start_val=None, fitable=v, tag=tag)
            # default enable False for n in 1/2n term and standard value 1
            self[tag].add_parameter(
                "powern", start_val=1, fitable=v, enabled=False, tag=tag
            )
            # 3 Body parameters
            self[tag].add_parameter("gamma", start_val=None, tag=tag)
            self[tag].add_parameter("c", start_val=None, tag=tag)
            self[tag].add_parameter("d", start_val=None, tag=tag)
            self[tag].add_parameter("h", start_val=None, tag=tag)
            # twomu = lambda3?
            self[tag].add_parameter("twomu", start_val=None, tag=tag)
            # m is not fittable
            self[tag].add_parameter(
                "m", start_val=1.0, fitable=False, tag=tag
            )  # 3 or 1
            # 2 and 3 body
            # default enable False for cutoff parameters
            self[tag].add_parameter("bigd", start_val=None, enabled=False, tag=tag)
            self[tag].add_parameter("bigr", start_val=None, enabled=False, tag=tag)

    def start_vals_from_file(self, file, tag_list=None):
        if tag_list is None:
            tag_list = self.list_groups()
        tag_dict = _tagdict_from_taglist(tag_list)
        t_params = TersoffParameters(tag_dict)
        t_params.start_vals_from_file(file, tag_list=tag_list)
        for tag, params in t_params.items():
            lam1 = params["lambda1"].start_val
            lam2 = params["lambda2"].start_val
            A = params["A"].start_val
            B = params["B"].start_val

            self[tag]["h"].start_val = -params["theta0"].start_val

            if tag_dict[tag]:
                self[tag]["r0"].start_val = get_r0(lam1, lam2, A, B)
                self[tag]["D0"].start_val = get_D0(lam1, lam2, A, B)
                self[tag]["S"].start_val = get_S(lam1, lam2)
                self[tag]["beta"].start_val = get_beta(lam1, lam2)
            else:  # pair only parameters are not read for these tags and are set to 0 to prevent 0 division errors
                self[tag]["r0"].start_val = 0.0
                self[tag]["D0"].start_val = 0.0
                self[tag]["S"].start_val = 0.0
                self[tag]["beta"].start_val = 0.0

            # Other parameters
            self[tag]["m"].start_val = params["m"].start_val
            self[tag]["powern"].start_val = params["n"].start_val
            self[tag]["c"].start_val = params["c"].start_val
            self[tag]["d"].start_val = params["d"].start_val
            self[tag]["bigd"].start_val = params["D"].start_val
            self[tag]["bigr"].start_val = params["R"].start_val
            self[tag]["gamma"].start_val = params["gamma"].start_val
            self[tag]["twomu"].start_val = params["lambda3"].start_val
            self[tag]["beta2"].start_val = params["beta"].start_val

    def _write_tersoff_file(self, directory, filename="input.tersoff"):
        tersoff = abop_to_tersoff_params(self)
        tersoff._write_tersoff_file(directory, filename=filename)

    def _to_xml_element(self):
        fit_dof = ET.Element("fit-dof")
        for tag, parameters in self.items():
            for param in parameters.values():
                if param.fitable:
                    fit_dof.append(param._to_xml_element())
        return fit_dof


def abop_to_tersoff_params(abop_parameters):
    """
    Transforms a ABOPParameters instance to TersoffParameters
    Does not take care of min or max vals, only initial values!
    Necessary to write the tersoff file as input.
    Not tested for anything else.
    Args:
        tersoff_parameters ([type]): [description]
    """

    tag_dict = _tagdict_from_taglist(abop_parameters.groups())
    tersoff = TersoffParameters(tag_dict)

    for tag, params in tersoff.items():
        if tag_dict[tag]:
            beta = abop_parameters[tag]["beta"].start_val
            S = abop_parameters[tag]["S"].start_val
            D0 = abop_parameters[tag]["D0"].start_val
            r0 = abop_parameters[tag]["r0"].start_val

            # parameters that need calculations
            params["A"].start_val = get_A(beta, S, D0, r0)
            params["B"].start_val = get_B(beta, S, D0, r0)
            params["lambda1"].start_val = get_lam1(beta, S)
            params["lambda2"].start_val = get_lam2(beta, S)
        else:  # pair only parameters are not read for these tags and are set to 0 to prevent 0 division errors
            params["A"].start_val = 0.0
            params["B"].start_val = 0.0
            params["lambda1"].start_val = 0.0
            params["lambda2"].start_val = 0.0

        params["theta0"].start_val = -abop_parameters[tag]["h"].start_val

        # Other parameters
        params["m"].start_val = abop_parameters[tag]["m"].start_val
        params["n"].start_val = abop_parameters[tag]["powern"].start_val
        params["c"].start_val = abop_parameters[tag]["c"].start_val
        params["d"].start_val = abop_parameters[tag]["d"].start_val
        params["D"].start_val = abop_parameters[tag]["bigd"].start_val
        params["R"].start_val = abop_parameters[tag]["bigr"].start_val
        params["gamma"].start_val = abop_parameters[tag]["gamma"].start_val
        params["lambda3"].start_val = abop_parameters[tag]["twomu"].start_val
        params["beta"].start_val = abop_parameters[tag]["beta2"].start_val
    return tersoff


class LJPotential(AbstractPotential):
    """
    Lennard Jones potential. Lacking some functionality,
    mostly implemented for testing and demonstatration purposes.
    TODO: Check if this still works after alls changes
    TODO: Implement missing functionality.
    """

    def __init__(
        self,
        init=None,
        sigma=None,
        epsilon=None,
        cutoff=None,
        species=None,
        identifier=None,
    ):
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


class EAMlikeMixin:
    def copy_final_to_initial_params(self, filter_func=None):
        """
        Copies final values of function paramters to start values.
        This f.e. allows to continue global with local minimization.
        """
        for functions in self._function_tuple:
            for f in functions.values():
                f.copy_final_to_initial_params(filter_func=filter_func)

    def randomize_parameters(self, rng, filter_func=None):
        for functions in self._function_tuple:
            for f in functions.values():
                f.randomize_parameters(rng=rng, filter_func=filter_func)

    def lock_parameters(self, filter_func=None):
        for functions in self._function_tuple:
            for f in functions.values():
                f.lock_parameters(filter_func=filter_func)

    def count_parameters(self, enabled_only=True):
        parameters = 0
        for functions in self._function_tuple:
            for f in functions.values():
                parameters += f.count_parameters(enabled_only=enabled_only)
        return parameters

    @property
    def _function_tuple(self):
        raise NotImplementedError("Implement a tuple with functions in subclass")

    @property
    def _function_dict(self):
        raise NotImplementedError("Implement a tuple with functions in subclass")

    def _mapping_functions_xml(self, pot):
        mappingxml = ET.SubElement(pot, "mapping")
        functionsxml = ET.SubElement(pot, "functions")

        for k, functions in self._function_dict.items():
            for f in functions.values():
                fxml = ET.SubElement(mappingxml, k)
                if len(f.species) == 1:
                    fxml.set("species", f"{f.species[0]}")
                elif len(f.species) == 2:
                    fxml.set("species-a", f"{f.species[0]}")
                    fxml.set("species-b", f"{f.species[1]}")
                elif len(f.species) == 2:
                    fxml.set("species-a", f"{f.species[0]}")
                    fxml.set("species-b", f"{f.species[1]}")
                    fxml.set("species-c", f"{f.species[2]}")
                fxml.set("function", f"{f.identifier}")
                functionsxml.append(f._to_xml_element())

    def _eam_parse_final_parameters(self, lines):
        """
        Internal Function.
        Parse function parameters from atomicrex output.

        Args:
            lines (list[str]): atomicrex output lines

        Raises:
            KeyError: Raises if a parsed parameter can't be matched to a function.
        """
        for l in lines:
            identifier, leftover, value = _parse_parameter_line(l)
            found = False
            for functions in self._function_tuple:
                if identifier in functions:
                    functions[identifier]._parse_final_parameter(leftover, value)
                    found = True
            if not found:
                raise KeyError(
                    f"Can't find {identifier} in potential, probably something went wrong during parsing.\n"
                    "Fitting parameters of screening functions probably doesn't work right now"
                )


class EAMPotential(AbstractPotential, EAMlikeMixin):
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
        species=None,
    ):
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

    @property
    def _function_tuple(self):
        return (
            self.pair_interactions,
            self.electron_densities,
            self.embedding_energies,
        )

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

        pot = pd.DataFrame(
            {
                "Name": f"{self.identifier}",
                "Filename": [[f"{job.working_directory}/{self.export_file}"]],
                "Model": ["Custom"],
                "Species": [species],
                "Config": [
                    [
                        "pair_style eam/fs\n",
                        f"pair_coeff * * {job.working_directory}/{self.export_file} {species_str}\n",
                    ]
                ],
            }
        )
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
            identifier, leftover, value = _parse_parameter_line(l)
            if identifier in self.pair_interactions:
                self.pair_interactions[identifier]._parse_final_parameter(
                    leftover, value
                )
            elif identifier in self.electron_densities:
                self.electron_densities[identifier]._parse_final_parameter(
                    leftover, value
                )
            elif identifier in self.embedding_energies:
                self.embedding_energies[identifier]._parse_final_parameter(
                    leftover, value
                )
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
        if filename is None:
            filename = f"{job.working_directory}/{self.export_file}"

        elements, r_values, rho_values = read_eam_fs_file(filename=filename)

        # TODO: figure out how to index ax for multiple elements
        fig, ax = plt.subplots(
            nrows=3 * len(elements),
            ncols=len(elements),
            figsize=(len(elements) * 8, len(elements) * 3 * 6),
            squeeze=False,
        )
        for i, (el, pot_dict) in enumerate(elements.items()):
            V_count = 0
            rho_count = 0
            for pot, y in pot_dict.items():
                if pot == "F":
                    xdata = rho_values
                    xlabel = "$\\rho $ [a.u.]"
                    k = 0
                    ylim = (np.min(y) - 0.5, 5)
                    ax[i * 3 + k, 0].plot(xdata, y)
                    ax[i * 3 + k, 0].set(ylim=ylim, title=f"{el} {pot}", xlabel=xlabel)
                elif "rho" in pot:
                    xdata = r_values
                    xlabel = "r [$\AA$]"
                    k = 1
                    ylim = (np.min(y) - 0.1, 1)
                    ax[i * 3 + k, rho_count].plot(xdata, y)
                    ax[i * 3 + k, rho_count].set(
                        ylim=ylim, title=f"{el} {pot}", xlabel=xlabel
                    )
                    rho_count += 1
                elif "V" in pot:
                    xdata = r_values[1:]
                    y = y[1:]
                    xlabel = "r [$\AA$]"
                    k = 2
                    ylim = (np.min(y) - 0.1, 2)
                    ax[i * 3 + k, V_count].plot(xdata, y)
                    ax[i * 3 + k, V_count].set(
                        ylim=ylim,
                        title=f"{el} {pot}",
                        xlabel=xlabel,
                    )
                    V_count += 1

        fig.tight_layout()
        return fig, ax

    def count_local_extrema(
        self, job, filename=None, count_minima=True, count_maxima=False
    ):
        if filename is None:
            filename = f"{job.working_directory}/{self.export_file}"
        elements, _r_values, _rho_values = read_eam_fs_file(filename=filename)
        extrema_dict = {}
        for el, func in elements.items():
            extrema_dict[el] = {}
            for func_name, a in func:
                extrema = 0
                if count_minima:
                    min_arr = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
                    extrema += len(min_arr[min_arr])
                if count_maxima:
                    max_arr = np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]
                    extrema += len(max_arr[max_arr])
                extrema_dict[el][func_name] = extrema
        return extrema_dict


class ADPotential(AbstractPotential, EAMlikeMixin):
    """
    Angular dependent potential.
    Usage: Create using the potential factory class.
    Add functions defined using the function_factory
    to self.pair_interactions, self.electron_densities
    and self.embedding_energies, self.u_functions and
    self.w_functions in dictionary style,
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
        species=None,
    ):
        super().__init__(init=init)
        if init is None:
            self.pair_interactions = DataContainer(table_name="pair_interactions")
            self.electron_densities = DataContainer(table_name="electron_densities")
            self.embedding_energies = DataContainer(table_name="embedding_energies")
            self.u_functions = DataContainer(table_name="u_functions")
            self.w_functions = DataContainer(table_name="w_functions")
            self.identifier = identifier
            self.export_file = export_file
            self.rho_range_factor = rho_range_factor
            self.resolution = resolution
            self.species = species

    @property
    def _function_tuple(self):
        return (
            self.pair_interactions,
            self.electron_densities,
            self.embedding_energies,
            self.u_functions,
            self.w_functions,
        )

    @property
    def _function_dict(self):
        return {
            "pair-interaction": self.pair_interactions,
            "electron-density": self.electron_densities,
            "embedding-energy": self.embedding_energies,
            "u-function": self.u_functions,
            "w-function": self.w_functions,
        }

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

        pot = pd.DataFrame(
            {
                "Name": f"{self.identifier}",
                "Filename": [[f"{job.working_directory}/{self.export_file}"]],
                "Model": ["Custom"],
                "Species": [species],
                "Config": [
                    [
                        "pair_style adp\n",
                        f"pair_coeff * * {job.working_directory}/{self.export_file} {species_str}\n",
                    ]
                ],
            }
        )
        return pot

    def write_xml_file(self, directory):
        """
        Internal function to convert to an xml element
        """
        adp = ET.Element("adp")
        adp.set("id", f"{self.identifier}")
        adp.set("species-a", f"{self.species[0]}")
        adp.set("species-b", f"{self.species[1]}")

        if self.export_file:
            export = ET.SubElement(adp, "export-adp-file")
            export.set("resolution", f"{self.resolution}")
            export.set("rho-range-factor", f"{self.rho_range_factor}")
            export.text = f"{self.export_file}"

        self._mapping_functions_xml(adp)

        filename = posixpath.join(directory, "potential.xml")
        write_pretty_xml(adp, filename)

    def _parse_final_parameters(self, lines):
        return self._eam_parse_final_parameters(lines)


class MEAMPotential(AbstractPotential, EAMlikeMixin):
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

    @property
    def _function_tuple(self):
        return (
            self.pair_interactions,
            self.electron_densities,
            self.embedding_energies,
            self.f_functions,
            self.g_functions,
        )

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
            identifier, leftover, value = _parse_parameter_line(l)
            if identifier in self.pair_interactions:
                self.pair_interactions[identifier]._parse_final_parameter(
                    leftover, value
                )
            elif identifier in self.electron_densities:
                self.electron_densities[identifier]._parse_final_parameter(
                    leftover, value
                )
            elif identifier in self.f_functions:
                self.f_functions[identifier]._parse_final_parameter(leftover, value)
            elif identifier in self.g_functions:
                self.g_functions[identifier]._parse_final_parameter(leftover, value)
            elif identifier in self.embedding_energies:
                self.embedding_energies[identifier]._parse_final_parameter(
                    leftover, value
                )
            else:
                raise KeyError(
                    f"Can't find {identifier} in potential, probably something went wrong during parsing.\n"
                    "Fitting parameters of screening functions probably doesn't work right now"
                )


def _parse_parameter_line(line):
    """
    Internal Function.
    Parses the function identifier, leftover and final value of a function parameter
    from an atomicrex output line looking like:
    EAM[EAM].V_CuCu[morse-B].delta: 0.0306585 [-0.5:0.5]
    EAM[EAM].CuCu_rho[spline].node[1].y: 2.10988 [1:3]
    EAM[EAM].V_CuCu[functionSum].V_rep[user-function].A: 0.680291
    EAM[EAM].CuCu_rho[spline].derivative-left: -1000.19
    Returns tuples like these:
    ("V_CuCu", ["delta:"] ,0.0306585)
    ("CuCu_rho", ["node[1", "y:"] ,2.10988)
    ("V_CuCu", ["V_rep[user-function", "A:"],0.680291)
    ("CuCu_rho", ["derivative-left:"] ,-1000.19)
    This allows to flexibly parse the lines for the different types
    of functional forms by passing leftover to a function defined
    in the corresponding function class.

    TODO: Add parsing of polynomial parameters.
    Returns:
        (str, [str], float): identifier, leftover, value
    """
    line = line.strip().split()
    value = float(line[1])
    info = line[0].split("].")
    identifier = info[1].split("[")[0]
    leftover = info[2:]
    return identifier, leftover, value


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
            tags[k] = True  # True if 2 and 3 body
        else:
            tags[k] = False  # False if only 3 body
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


def _tagdict_from_taglist(taglist):
    tagdict = {}
    for tag in taglist:
        elems = tag_re.findall(tag)
        if elems[1] == elems[2]:
            tagdict[tag] = True
        else:
            tagdict[tag] = False
    return tagdict


def read_eam_fs_file(filename):
    # Read the cotents of an eam/fs file and returns it as a dicitionary of numpy arrays
    # this is copy pasted from one of my old scripts and could probably be reworked
    elements = {}
    with open(filename, "r") as f:
        # skip comment lines
        for _ in range(3):
            f.readline()

        element_list = f.readline().split()
        for element in element_list[1:]:
            elements[element] = {}

        Nrho, drho, Nr, dr, cutoff = f.readline().split()
        Nrho = int(Nrho)
        drho = float(drho)
        Nr = int(Nr)
        dr = float(dr)
        cutoff = float(cutoff)

        rho_values = np.linspace(0, Nrho * drho, Nrho, endpoint=False)
        r_values = np.linspace(0, Nr * dr, Nr, endpoint=False)

        for element in elements:
            # skip a line with unnecessary information
            f.readline()
            elements[element]["F"] = np.fromfile(f, count=Nrho, sep=" ")
            for rho_element in elements:
                elements[element]["rho_{}{}".format(element, rho_element)] = (
                    np.fromfile(f, count=Nr, sep=" ")
                )

        # V_ij = V_ji so it is written only once in the file => avoid attempts to read it twice
        # with a list of elements where it has been read
        # TODO: Have another look how to do this checking
        V_written = []
        for element in elements:
            for V_element in elements:
                elementV_element = "{}{}".format(element, V_element)
                V_elementelement = "{}{}".format(V_element, element)
                if (
                    not elementV_element in V_written
                    and not V_elementelement in V_written
                ):
                    elements[element]["V_{}".format(elementV_element)] = np.fromfile(
                        f, count=Nr, sep=" "
                    )
                    # The tabulated values are not V(r) but V(r) * r, so they are divided by r here,
                    # with exception of the first value to prevent division by 0.
                    elements[element]["V_{}".format(elementV_element)][1:] = (
                        elements[element]["V_{}".format(elementV_element)][1:]
                        / r_values[1:]
                    )
                    V_written.append(elementV_element)
    return elements, r_values, rho_values
