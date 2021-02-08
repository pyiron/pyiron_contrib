import posixpath
import xml.etree.ElementTree as ET

from pyiron_base import PyironFactory, InputList

from pyiron_contrib.atomistic.atomicrex.function_factory import FunctionFactory
from pyiron_contrib.atomistic.atomicrex.utility_functions import write_pretty_xml

class ARPotFactory(PyironFactory):
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
    def lennard_jones_potential(sigma, epsilon, cutoff, species={"a": "*", "b": "*"}, identifier="LJ"):
        return LJPotential(sigma, epsilon, cutoff, species=species, identifier=identifier)


class AbstractPotential(InputList):
    def __init__(self, table_name="potential"):
        super().__init__(table_name=table_name)


class LJPotential(AbstractPotential):
    def __init__(self, sigma=None, epsilon=None, cutoff=None, species=None, identifier=None):
        super().__init__()
        self.sigma = sigma
        self.epsilon = epsilon
        self.cutoff = cutoff
        self.species = species
        self.identifier = identifier

    def write_xml_file(self, directory):
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
    def __init__(self, identifier=None, export_file=None, rho_range_factor=None, resolution=None, species=None):
        super().__init__()
        self.pair_interactions = InputList(table_name="pair_interactions")
        self.electron_densities = InputList(table_name="electron_densities")
        self.embedding_energies = InputList(table_name="embedding_energies")
        self.identifier = identifier
        self.export_file = export_file
        self.rho_range_factor = rho_range_factor
        self.resolution = resolution
        self.species = species

    def write_xml_file(self, directory):

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
        for l in lines:
            identifier, param, value = self._parse_parameter_line(l)
            if identifier in self.pair_interactions:
                self.pair_interactions[identifier].parameters[param].final_value = value

            elif identifier in self.electron_densities:
                self.electron_densities[identifier].parameters[param].final_value = value
            elif identifier in self.embedding_energies:
                self.embedding_energies[identifier].parameters[param].final_value = value
            else:

                raise ValueError(
                    f"Can't find {identifier} in potential, probably something went wrong during parsing.\n"
                    "Fitting parameters of screening functions probably doesn't work right now"
                )


    @staticmethod
    def _parse_parameter_line(line):
        # Parse output line that looks like:
        # EAM[EAM].V_CuCu[morse-B].delta: 0.0306585 [-0.5:0.5]
        # EAM[EAM].CuCu_rho[spline].node[1].y: 2.10988 [1:3]

        line = line.strip().split()
        value = float(line[1])
        info = line[0].split(".")
        identifier = info[1].split("[")[0]
        param = info[2]
        if param.startswith("node"):
            x = float(param.split("[")[1].rstrip("]"))
            param = f"node_{x}"
        else:
            param = param.rstrip(":")
        return identifier, param, value
