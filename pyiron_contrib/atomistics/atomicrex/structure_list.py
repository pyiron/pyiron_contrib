import posixpath
import xml.etree.ElementTree as ET

import numpy as np
from ase import Atoms as ASEAtoms
from pyiron_base import DataContainer
from pyiron_atomistics import pyiron_to_ase, ase_to_pyiron, Atoms

from pyiron_contrib.atomistics.atomistics.job.structurestorage import StructureStorage
from pyiron_contrib.atomistics.atomicrex.fit_properties import ARFitPropertyList, ARFitProperty
from pyiron_contrib.atomistics.atomicrex.utility_functions import write_pretty_xml
from pyiron_contrib.atomistics.atomicrex.fit_properties import FlattenedARProperty, FlattenedARVectorProperty

# TODO: Develop a useful MinimalStructure class for
#
#class MinimalStructure:
#    def __init__(self, symbols, cell, positions):
#        self.cell = cell
#        self.symbols = symbols
#        self.positions = positions


class ARStructureContainer:
    def __init__(self):
        self.fit_properties = {}
        # most init can't be done without some information
        # This allows to preallocate arrays and speed everything up massively
        # when writing and reading hdf5 files

    __version__ = "0.2.0"
    __hdf_version__ = "0.2.0"

    def init_structure_container(
        self,
        num_structures,
        num_atoms,
        fit_properties=["atomic-energy", "atomic-forces"],
        structure_file_path=None
        ):
        for p in fit_properties:
            if p == "atomic-forces":
                self.fit_properties[p] = FlattenedARVectorProperty(num_structures=num_structures, num_atoms=num_atoms, prop=p)
            else:
                self.fit_properties[p] = FlattenedARProperty(num_structures=num_structures, prop=p)
        self._init_structure_container(num_structures, num_atoms)
        self.structure_file_path = structure_file_path

    def _init_structure_container(self, num_structures, num_atoms):
        self.flattened_structures = StructureStorage(num_structures=num_structures, num_atoms=num_atoms)
        self.fit = np.empty(num_structures, dtype=bool)
        self.clamp = np.empty(num_structures, dtype=bool)
        self.relative_weight = np.empty(num_structures)

    def add_structure(self, structure, identifier, fit=True, relative_weight=1, clamp=True):
        self.flattened_structures.add_structure(structure, identifier)
        self.fit[self.flattened_structures.prev_structure_index] = fit
        self.relative_weight[self.flattened_structures.prev_structure_index] = relative_weight
        self.clamp[self.flattened_structures.prev_structure_index] = clamp

    def add_scalar_fit_property(
        self,
        prop="atomic-energy",
        target_value=np.nan,
        fit=True,
        relax=False,
        relative_weight=1,
        residual_style=0,
        output=True,
        tolerance=np.nan,
        min_val=np.nan,
        max_val=np.nan,
        ):
        self.fit_properties[prop].target_value[self.flattened_structures.prev_structure_index] = target_value
        self.fit_properties[prop].fit[self.flattened_structures.prev_structure_index] = fit
        self.fit_properties[prop].relax[self.flattened_structures.prev_structure_index] = relax
        self.fit_properties[prop].relative_weight[self.flattened_structures.prev_structure_index] = relative_weight
        self.fit_properties[prop].residual_style[self.flattened_structures.prev_structure_index] = residual_style
        self.fit_properties[prop].output[self.flattened_structures.prev_structure_index] = output
        self.fit_properties[prop].tolerance[self.flattened_structures.prev_structure_index] = tolerance
        self.fit_properties[prop].min_val[self.flattened_structures.prev_structure_index] = min_val
        self.fit_properties[prop].max_val[self.flattened_structures.prev_structure_index] = max_val

        
    def add_vector_fit_property(
        self,
        prop="atomic-forces",
        target_value=None,
        fit=True,
        relax=False,
        relative_weight=1,
        residual_style=0,
        tolerance=np.nan,
        output=True,
        ):
        if target_value is not None:
            self.fit_properties[prop].target_value[self.flattened_structures.prev_atom_index:self.flattened_structures.current_atom_index] = target_value
        self.fit_properties[prop].fit[self.flattened_structures.prev_structure_index] = fit
        self.fit_properties[prop].relax[self.flattened_structures.prev_structure_index] = relax
        self.fit_properties[prop].relative_weight[self.flattened_structures.prev_structure_index] = relative_weight
        self.fit_properties[prop].residual_style[self.flattened_structures.prev_structure_index] = residual_style
        self.fit_properties[prop].output[self.flattened_structures.prev_structure_index] = output
        self.fit_properties[prop].tolerance[self.flattened_structures.prev_structure_index] = tolerance

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

    def to_hdf(self, hdf, group_name="structures"):
        with hdf.open(group_name) as h:
            self._type_to_hdf(h)
            self.flattened_structures.to_hdf(hdf=h)

            h_fit = h.create_group("fit_properties")
            for k, v in self.fit_properties.items():
                v.to_hdf(h_fit, group_name=k)
            
            h["fit"] = self.fit
            h["clamp"] = self.clamp
            h["relative_weight"] = self.relative_weight
            h["structure_file_path"] = self.structure_file_path

    def from_hdf(self, hdf, group_name="structures"):
        with hdf.open(group_name) as h:
            version = h.get("HDF_VERSION", "0.1.0")
            # Compatibility Old and new StructureStorage
            if version == "0.1.0":
                num_structures = h["flattened_structures/num_structures"]
                num_atoms = h["flattened_structures/num_atoms"]
            else:
                num_structures = h["structures/num_structures"]
                num_atoms = h["structures/num_atoms"]

            self._init_structure_container(num_structures, num_atoms)
            self.flattened_structures.from_hdf(hdf=h)

            h_fit = h["fit_properties"]
            for group in h_fit.list_groups():
                if group == "atomic-forces":
                    self.fit_properties[group] = FlattenedARVectorProperty(num_structures, num_atoms, group)
                    self.fit_properties[group].from_hdf(h_fit, group_name=group)
                else:
                    self.fit_properties[group] = FlattenedARProperty(num_structures, group)
                    self.fit_properties[group].from_hdf(h_fit, group_name=group)

            self.clamp = h["clamp"]
            self.fit = h["fit"]
            self.relative_weight = h["relative_weight"]
            self.structure_file_path = h["structure_file_path"]
    

    def write_xml_file(self, directory, name="structures.xml"):
        """
        Internal helper function that writes an atomicrex style
        xml file containg all structures.

        Args:
            directory (string): Working directory.
            name (str, optional): . Defaults to "structures.xml".
        """        
        root = ET.Element("group")
        if self.structure_file_path is None:

            # write POSCAR and xml
            for i in range(self.flattened_structures.num_structures):
                vec_start = self.flattened_structures.start_index[i]
                vec_end = self.flattened_structures.start_index[i]+self.flattened_structures.length[i]
                write_modified_poscar(
                    identifier=self.flattened_structures.identifier[i],
                    forces=self.fit_properties["atomic-forces"].target_value[vec_start:vec_end],
                    positions=self.flattened_structures.positions[vec_start:vec_end],
                    symbols=self.flattened_structures.symbols[vec_start:vec_end],
                    cell=self.flattened_structures.cell[i],
                    directory=directory
                )
                
                fit_properties_xml = ET.Element("properties")
                for flat_prop in self.fit_properties.values():
                    fit_properties_xml.append(flat_prop.to_xml_element(i))

                struct_xml = structure_meta_xml(
                    identifier=self.flattened_structures.identifier[i],
                    relative_weight=self.relative_weight[i],
                    clamp=self.clamp[i],
                    fit_properties=fit_properties_xml,
                    struct_file_path=self.structure_file_path,
                    fit = self.fit[i],
                )
                root.append(struct_xml)
        else:
            # write only xml and use POSCARs written already to some path
            for i in range(self.flattened_structures.num_structures):
                fit_properties_xml = ET.Element("properties")
                for flat_prop in self.fit_properties.values():
                    fit_properties_xml.append(flat_prop.to_xml_element(i))

                struct_xml = structure_meta_xml(
                    identifier=self.flattened_structures.identifier[i],
                    relative_weight=self.relative_weight[i],
                    clamp=self.clamp[i],
                    fit_properties=fit_properties_xml,
                    struct_file_path=self.structure_file_path,
                    fit = self.fit[i],
                )
                root.append(struct_xml)
        filename = posixpath.join(directory, name)
        write_pretty_xml(root, filename)


    def _parse_final_properties(self, struct_lines):
        """
        Internal function that parses the values of fitted properties
        calculated with the final iteration of the fitted potential.

        Args:
            struct_lines (list[str]): lines from atomicrex output that contain structure information.
        """ 
        force_vec_triggered = False
        for l in struct_lines:
            l = l.strip()
            
            if force_vec_triggered:
                if l.startswith("atomic-forces:"):
                    l = l.split()
                    index = int(l[1])
                    final_forces[index, 0] = float(l[2].lstrip("(").rstrip(","))
                    final_forces[index, 1] = float(l[3].rstrip(","))
                    final_forces[index, 2] = float(l[4].rstrip(")"))
                else:
                    force_vec_triggered = False
                    start_index = self.flattened_structures.start_index[s_index]
                    self.fit_properties["atomic-forces"].final_value[start_index:start_index+len_struct] = final_forces

            # This has to be if and not else because it has to run in the same iteration. Empty lines get skipped.
            if not force_vec_triggered and l:
                if l.startswith("Structure"):
                    s_id = l.split("'")[1]
                    s_index = np.nonzero(self.flattened_structures.identifier==s_id)[0][0]

                else:
                    if not l.startswith("atomic-forces avg/max:"):
                        prop, f_val = ARFitProperty._parse_final_value(line=l)
                        if prop in self.fit_properties.keys():
                            self.fit_properties[prop].final_value[s_index] = f_val
                    else:
                        force_vec_triggered = True
                        len_struct = self.flattened_structures.length[s_index]
                        final_forces = np.empty((len_struct, 3))
    

### This is probably useless like this in most cases because forces can't be passed.
def structure_to_xml_element(structure):
    """
    Converts an ase/pyiron atoms object to an atomicrex xml element
    Right now forces can't be passed in the xml file, so this is not really helpful.
    Args:
        structure (Atoms): ase or pyiron Atoms

    Returns:
        (ET.Element, ET.Element): atomicrex structure xml
    """    
    pbc = ET.Element("pbc")
    pbc.set("x", f"{structure.pbc[0]}".lower())
    pbc.set("y", f"{structure.pbc[1]}".lower())
    pbc.set("z", f"{structure.pbc[2]}".lower())

    c = structure.cell
    cell = ET.Element("cell")
    for i in range(3):
        a = ET.SubElement(cell, f"a{i+1}")
        a.set("x", f"{c[i][0]}")
        a.set("y", f"{c[i][1]}")
        a.set("z", f"{c[i][2]}")

    atoms = ET.SubElement(cell, "atoms")
    for at in structure:
        a = ET.SubElement(atoms, "atom")
        a.set("type", at.symbol)
        a.set("x", f"{at.a}")
        a.set("y", f"{at.b}")
        a.set("z", f"{at.c}")
        a.set("reduced", "false")

    return pbc, cell


class ARStructure(object):
    """
    Class that contains a pyiron structure, an ARFitPropertyList
    and some attributes that define the fitting procedure.
    Provides internal helper methods.
    """

    def __init__(self, structure=None, fit_properties=None, identifier=None, relative_weight=1, clamp=True, fit=True):
        self._structure = None
        self._fit_properties = None
        if structure is not None:
            self.structure = structure
        if fit_properties is not None:
            self.fit_properties = fit_properties
        self.identifier = identifier
        self.relative_weight = relative_weight
        self.clamp = clamp
        self.fit = fit

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        if not isinstance(structure, Atoms):
            raise ValueError("structure has to be a pyiron atoms instance")
        self._structure = structure

    @property
    def fit_properties(self):
        return self._fit_properties

    @fit_properties.setter
    def fit_properties(self, fit_properties=ARFitPropertyList()):
        if not isinstance(fit_properties, ARFitPropertyList):
            raise ValueError("fit_properties have to be an ARFitPropertyList")
        self._fit_properties = fit_properties

    def _write_poscar_return_xml(self, directory, struct_file_path):
        """
        Internal function that writes the structure in an extended POSCAR
        format that contains forces. Returns an atomicrex xml
        element that references the POSCAR file and contains additional
        information like the fit properties.

        Args:
            directory (string): Working directory
            struct_file_path (string): path to a directory containing structure files. If None structure files are written in working directory
        Returns:
            [ElementTree xml element]: xml element referencing POSCAR, contains additional information
        """        
        if struct_file_path is None:
            forces = self.fit_properties["atomic-forces"].target_value
            write_modified_poscar(
                identifier = self.identifier,
                structure = self.structure,
                forces=forces,
                directory=directory
                )

        return structure_meta_xml(
            identifier = self.identifier,
            relative_weight = self.relative_weight,
            clamp = self.clamp,
            fit_properties = self.fit_properties,
            fit = self.fit,
            struct_file_path = struct_file_path,
        )

    def to_hdf(self, hdf=None, group_name="arstructure", full_hdf=True):
        """
        Internal function
        """        
        with hdf.open(group_name) as hdf_s_lst:
            if full_hdf:
                self.structure.to_hdf(hdf=hdf_s_lst, group_name="structure")
            self.fit_properties.to_hdf(hdf=hdf_s_lst, group_name="fit_properties")
            hdf_s_lst["relative_weight"] = self.relative_weight
            hdf_s_lst["identifier"] = self.identifier
            hdf_s_lst["clamp"] = self.clamp

    def from_hdf(self, hdf=None, group_name="arstructure", full_hdf=True):
        """
        Internal function 
        """    
        with hdf.open(group_name) as hdf_s_lst:
            if full_hdf:
                structure = Atoms()
                structure.from_hdf(hdf_s_lst, group_name="structure")
                self.structure = structure
            self.fit_properties = ARFitPropertyList()
            self.fit_properties.from_hdf(hdf=hdf_s_lst, group_name="fit_properties")
            self.relative_weight = hdf_s_lst["relative_weight"]
            self.identifier = hdf_s_lst["identifier"]
            self.clamp = hdf_s_lst["clamp"]


class ARStructureList(object):
    """
    Container class for AR structures. structures attribute
    of the atomicrex job class.
    Provides functions for internal use and a convenient way
    to add additional structures to the atomicrex job.

    Will be replaced with FlattenedStructureContainer or used only for representation of the data
    """    
    def __init__(self):
        self._structure_dict = {}

        # This allows to give a path where structures are stored in atomicrex POSCAR format
        # It modifies the write input function to prevent a lot of file io and slow down performance
        # If it is not given a file is written for every structure of the job.
        self.struct_file_path = None
        self.full_structure_to_hdf = True
        self.num_structures = 0
        self.num_atoms = 0

    def add_structure(self, structure, identifier, fit_properties=None, relative_weight=1, clamp=True, fit=True):
        """
        Provides a convenient way to add additional
        structures to the job.
        
        Args:
            structure (Pyiron Atoms): structure that should be added.
            identifier (string): ID string. Must be unique or overwrites the old structure.
            fit_properties ([ARFitPropertyList], optional): Fit Properties can be conveniently added after appending the structure. Defaults to None.
            relative_weight (int, optional): Assigns a weight for the objective function. Defaults to 1.
            clamp (bool, optional): Clamp the structure (Do not relax it). Defaults to True.

        Raises:
            ValueError: Raises if structure is not a pyiron atoms instance.

        Returns:
            [ARStructure]: Acces to the atomicrex structure in the structure list.
        """        
        if isinstance(structure, Atoms):
            if fit_properties is None:
                fit_properties = ARFitPropertyList()
            ar_struct = ARStructure(structure, fit_properties, identifier, relative_weight=relative_weight, clamp=clamp)
            self.num_structures += 1
            self.num_atoms += len(structure)
            return self._add_ARstructure(ar_struct)
        else:
            raise ValueError("Structure has to be a Pyiron Atoms instance")


    def _add_ARstructure(self, structure):
        """Internal helper function to be able to implement some more checks later on
        """        
        identifier = f"{structure.identifier}"
        self._structure_dict[identifier] = structure
        return self._structure_dict[identifier]

    def write_xml_file(self, directory, name="structures.xml"):
        """
        Internal helper function that write an atomicrex style
        xml file containg all structures.

        Args:
            directory (string): Working directory.
            name (str, optional): . Defaults to "structures.xml".
        """        
        root = ET.Element("group")
        for s in self._structure_dict.values():
            root.append(s._write_poscar_return_xml(directory, self.struct_file_path))
        filename = posixpath.join(directory, name)
        write_pretty_xml(root, filename)


    #def to_hdf(self, hdf=None, group_name="arstructurelist"):

    #def from_hdf(self, hdf=None, group_name="arstructurelist"):
    
    def to_hdf(self, hdf=None, group_name="arstructurelist"):
        """
        Internal function 
        """    
        with hdf.open(group_name) as hdf_s_lst:
            hdf_s_lst["struct_file_path"] = self.struct_file_path
            hdf_s_lst["full_structure_to_hdf"] = self.full_structure_to_hdf
            for k, v in self._structure_dict.items():
                v.to_hdf(hdf=hdf_s_lst, group_name=k, full_hdf=self.full_structure_to_hdf)

    def from_hdf(self, hdf=None, group_name="arstructurelist"):
        """
        Internal function 
        """        
        with hdf.open(group_name) as hdf_s_lst:
            # compatibility with my old jobs. find a way to write this to hdf for them and delete try except block
            try:
                self.struct_file_path = hdf_s_lst["struct_file_path"]
                self.full_structure_to_hdf = hdf_s_lst["full_structure_to_hdf"]
            except ValueError:
                pass

            for g in sorted(hdf_s_lst.list_groups()):
                s = ARStructure()
                s.from_hdf(hdf=hdf_s_lst, group_name=g, full_hdf=self.full_structure_to_hdf)
                self._structure_dict[g] = s


    def _parse_final_properties(self, struct_lines):
        """
        Internal function that parses the values of fitted properties
        calculated with the final iteration of the fitted potential.

        Args:
            struct_lines (list[str]): lines from atomicrex output that contain structure information.
        """ 
        force_vec_triggered = False   
        for l in struct_lines:
            l = l.strip()
            
            if force_vec_triggered:
                if l.startswith("atomic-forces:"):
                    l = l.split()
                    index = int(l[1])
                    final_forces[index, 0] = float(l[2].lstrip("(").rstrip(","))
                    final_forces[index, 1] = float(l[3].rstrip(","))
                    final_forces[index, 2] = float(l[4].rstrip(")"))
                else:
                    force_vec_triggered = False
                    s.fit_properties["atomic-forces"].final_value = final_forces
            
            # This has to be if and not else because it has to run in the same iteration. Empty lines get skipped.
            if not force_vec_triggered and l:
                if l.startswith("Structure"):
                    s_id = l.split("'")[1]
                    s = self._structure_dict[s_id]
            
                else:
                    if not l.startswith("atomic-forces avg/max:"):
                        prop, f_val = ARFitProperty._parse_final_value(line=l)
                        if prop in s.fit_properties:
                            s.fit_properties[prop].final_value = f_val
                    else:
                        force_vec_triggered = True
                        final_forces = np.empty((len(s.structure), 3))



def write_modified_poscar(identifier, forces, directory, structure=None, positions=None, cell=None, symbols=None):
    """
    Internal function. Writes a pyiron structure
    and corresponding forces in a modified POSCAR file.
    Either provide a structure instance or positions and symbols.
    Args:
        identifier (str): Unique identifier used for the filename.
        structure (Atoms): ase or pyiron atoms object. 
        positions (np.array): atomic positions
        symbols (np.array): chemical symbols
        forces (array or list[list]): atomic forces. Must be in same order as positions.
        directory (str): Working directory.
    """    
    filename = posixpath.join(directory, f"POSCAR_{identifier}")
    with open(filename, 'w') as f:
        # Elements as system name
        # Also check if the symbols are disordered
        # to prevent errors resulting from poscar format
        # not having indices or element for each atom.
        elements = []
        counter = 0
        last_elem = "XX"
        if symbols is None:
            symbols = np.array(structure.symbols)
        for elem in symbols:
            if not elem in elements:
                elements.append(elem)
            if not last_elem == elem:
                counter += 1
                last_elem = elem
        if counter != len(elements):
            raise ValueError(
                "Structures with disordered elements are not supported right now\n"
                "They can be sorted together with target forces using numpy.argsort"
            )
            ## maybe this can be fixed returning an argsort array to sort the forces.
        f.write(f"{elements}\n")

        # Cell metric
        f.write('1.0\n')
        if cell is None:
            cell = structure.cell
        for vec in cell:
            for v in vec:
                f.write(f'{v} ')
            f.write('\n')

        # Element names
        for elem in elements:
            f.write(f'{elem} ')
        f.write('\n')

        # Number of elements per type
        for elem in elements:
            n = len(symbols[symbols==elem])
            f.write(f'{n}    ')
        f.write('\n')

        # Scaled coordinates
        if positions is None:
            positions = structure.positions
        f.write('Cartesian\n')

        # Coordinates and forces
        for pos, force in zip(positions, forces):
            for r in pos: f.write((f'{r} '))
            f.write('   ')
            for r in force: f.write((f'{r} '))
            f.write('\n')


def structure_meta_xml(
        identifier,
        relative_weight,
        clamp, ## Not sure if and how this combines with relax in the ARFitParameter sets
        fit_properties,
        struct_file_path,
        fit,
        mod_poscar = True,
        
):
    """
    Internal function. Creates xml element with
    scalar properties, weight and reference to POSCAR
    containg structure and forces.

    Args:
        identifier (str): Unique identifier.
        relative_weight (float): weight in objective function
        clamp (bool): clamp the structure (no relaxation) 
        mod_poscar (bool, optional): Combine with poscar file. Defaults to True.
        struct_file_path (string): path to POSCAR file
    Raises:
        NotImplementedError: mod_poscar has to be True, because forces can't
        be provided in xml format in atomicrex yet.

    Returns:
        [ElementTree xml element]: atomicrex structure xml element.
    """

    struct_xml = ET.Element("user-structure")
    struct_xml.set("id", f"{identifier}")
    struct_xml.set("relative-weight", f"{relative_weight}")
    if fit: 
        struct_xml.set("fit", "true")
    else:
        struct_xml.set("fit", "false")

    if mod_poscar:
        poscar_file = ET.SubElement(struct_xml, "poscar-file")
        if struct_file_path is None:
            poscar_file.text = f"POSCAR_{identifier}"
        else:
            poscar_file.text = f"{struct_file_path}POSCAR_{identifier}"
    else:
        #    struct_xml.extend(structure_to_xml_element(structure))
        raise NotImplementedError(
            "Only writing structure meta information, \n"
            "not structure data for now because this is not implemented in atomicrex"
        )

    if not clamp:
        relax_dof = ET.SubElement(struct_xml, "relax-dof")
        atom_coordinates = ET.SubElement(relax_dof, "atom-coordinates")
    
    if isinstance(fit_properties, ARFitPropertyList):
        properties = ET.SubElement(struct_xml, "properties")
        for prop in fit_properties.values():
            properties.append(prop.to_xml_element())
    else:
        struct_xml.append(fit_properties)

    return struct_xml
