import posixpath
import xml.etree.ElementTree as ET

from ase import Atoms as ASEAtoms
from pyiron_base import InputList
from pyiron import pyiron_to_ase, ase_to_pyiron, Atoms

from pyiron_contrib.atomistic.atomicrex.fit_properties import ARFitPropertyList, ARFitProperty
from pyiron_contrib.atomistic.atomicrex.utility_functions import write_pretty_xml


class StructureList(InputList):
    def __init__(self, table_name="Structures"):
        super().__init__(table_name=table_name)
        
    def add_structure(self, structure, **kw_properties):
        if isinstance(structure, ASEAtoms):
            i = structure.info
            structure = ase_to_pyiron(structure)
            structure.info = i
        elif isinstance(structure, Atoms):
            pass
        else:
            raise ValueError("Structures have to be supplied as pyiron or ase atoms")
            
        structure.info.update(kw_properties)
        self.append(structure)
        
    def to_hdf(self, hdf, group="structure_list"):
        with hdf.open(group) as hdf_s_lst:
            for k, struct in self:
                struct.to_hdf(hdf=hdf_s_lst, group=f"structure_{k}")
            
    def from_hdf(self, hdf, group="structure_list"):
        with hdf.open(group) as hdf_s_lst:
            for g in sorted(hdf_s_lst.list_groups()):
                structure = Atoms()
                structure.from_hdf(hdf, group_name = g)
                self.append(structure)
        
    def to_ase_db(self, db):
        print("NOT TESTED FUNCTION")
        for struct in self.structures:
            struct = pyiron_to_ase(struct)
            db.write(atoms=struct, forces=self.forces, key_value_pairs=struct.info)
            
    def from_ase_db(self, db):
        for row in db.select():
            kvp = row.key_value_pairs
            if "pyiron_id" in kvp.keys():
                p_id = kvp["pyiron_id"]
            else:
                raise KeyError("pyiron_id has to be supplied as key in row to transform to structure list")
            i = row.data            
            i.update(kvp)
            a = row.toatoms()
            structure = ase_to_pyiron(a)
            structure.info = i
            print(i)
            self[f"pyiron_id{p_id}"] = structure


### This is probably useless like this because forces can't be passed.
def structure_to_xml_element(structure):
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


class ARStructure(InputList):
    #def __new__(cls, *args, **kwargs):
    #    instance = super().__new__(cls)
    #    object.__setattr__(instance, "structure", Atoms())

    def __init__(self, structure, fit_properties, identifier, relative_weight=1, clamp=True,):
        self.structure = structure
        self.fit_properties = fit_properties
        self.identifier = identifier
        self.relative_weight = relative_weight
        self.clamp = clamp
    
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

    #@fit_properties.getter
    #def fit_properties(self):
    #    return self._fit_properties
    
    def _write_poscar_return_xml(self, directory):
        forces = self.fit_properties["atomic-forces"].target_value
        write_modified_poscar(self.identifier, self.structure, forces, directory)
        return structure_meta_xml(
            self.identifier,
            self.relative_weight,
            self.clamp,
            self.fit_properties
        )
    
    def to_hdf(self, hdf=None, group_name=None):
        self.structure.to_hdf(hdf=hdf, group_name=group_name)
        super().to_hdf(hdf=hdf, group_name=group_name)


class ARStructureList(InputList):
    def __init__(self):
        super().__init__(table_name="Structures")
        
    def add_structure(self, structure, identifier, fit_properties=None, relative_weight=1, clamp=True):
        if isinstance(structure, Atoms):
            if fit_properties is None:
                fit_properties = ARFitPropertyList()
            ar_struct = ARStructure(structure, fit_properties, identifier, relative_weight=relative_weight, clamp=clamp)
            return self._add_ARstructure(ar_struct)
        else:
            raise ValueError("Structure has to be an ARStructure instance")
        
    ## Split into another function to be able to implement some checks later
    def _add_ARstructure(self, structure):
        identifier = f"{structure.identifier}"
        self[identifier] = structure
        return self[identifier]

    def write_xml_file(self, directory, name="structures.xml"):
        root = ET.Element("group")
        for s in self.values():
            root.append(s._write_poscar_return_xml(directory))
        filename = posixpath.join(directory, name)
        write_pretty_xml(root, filename)
    
    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(hdf=hdf, group_name=group_name)
    
    def from_hdf(self, hdf=None):
        raise NotImplementedError("TODO")
    
    def _parse_final_properties(self, struct_lines):
        for l in struct_lines:
            l = l.strip()
            if l.startswith("Structure"):
                s_id = l.split("'")[1]
                s = self[s_id]
            else:
                prop, f_val = ARFitProperty._parse_final_value(line=l)
                if prop in s.fit_properties:
                    s.fit_properties[prop].final_value = f_val

    
def write_modified_poscar(identifier, structure, forces, directory):
    filename = posixpath.join(directory, f"POSCAR_{identifier}")
    with open(filename, 'w') as f:
        # Elements as system name
        elements = []
        for elem in structure.get_chemical_symbols():
            if not elem in elements: elements.append(elem)
        f.write(f"{elements}\n")
        
        # Cell metric
        f.write('1.0\n')
        for vec in structure.cell:
            for v in vec:
                f.write(f'{v} ')
            f.write('\n')

        # Element names
        for elem in elements:
            f.write(f'{elem} ')
        f.write('\n')

        # Number of elements per type
        
        for elem in elements:
            symbols = structure.get_chemical_symbols()
            n = len(symbols[symbols==elem])
            f.write(f'{n}    ')
        f.write('\n')

        # Scaled or cartesian coordinates
        coordinates = structure.get_scaled_positions()
        f.write('Direct\n')

        # Coordinates and forces
        for coord, force in zip(coordinates, forces):
            for r in coord: f.write((f'{r} '))
            f.write('   ')
            for r in force: f.write((f'{r} '))
            f.write('\n')

                 
def structure_meta_xml(
    identifier,
    relative_weight,
    clamp, ## Not sure if and how this combines with relax in the ARFitParameter sets
    fit_properties,
    mod_poscar = True,
    ):
    
    struct_xml = ET.Element("user-structure")
    struct_xml.set("id", f"{identifier}")
    struct_xml.set("relative-weight", f"{relative_weight}")
    
    if mod_poscar:
        poscar_file = ET.SubElement(struct_xml, "poscar-file")
        poscar_file.text = f"POSCAR_{identifier}"
    else:
    #    struct_xml.extend(structure_to_xml_element(structure))
        raise NotImplementedError(
            "Only writing structure meta information, \n"
            "not structure data for now because this is not implemented in atomicrex"
        )

    if not clamp:
        relax_dof = ET.SubElement(struct_xml, "relax-dof")
        atom_coordinates = ET.SubElement(relax_dof, "atom-coordinates")
    
    properties = ET.SubElement(struct_xml, "properties")
    for prop in fit_properties.values():
        properties.append(prop.to_xml_element())
    
    return struct_xml


