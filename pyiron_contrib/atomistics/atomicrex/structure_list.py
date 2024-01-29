# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os, posixpath
import xml.etree.ElementTree as ET

import numpy as np
from numpy import ndarray
from pyiron_base import state, DataContainer, FlattenedStorage
from pyiron_atomistics import Atoms, ase_to_pyiron

from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_contrib.atomistics.atomistics.job.trainingcontainer import (
    TrainingPlots,
    TrainingContainer,
    TrainingStorage,
)
from pyiron_contrib.atomistics.atomicrex.fit_properties import (
    ARFitPropertyList,
    ARFitProperty,
)
from pyiron_contrib.atomistics.atomicrex.utility_functions import write_pretty_xml
from pyiron_contrib.atomistics.atomicrex.fit_properties import (
    FlattenedARScalarProperty,
    FlattenedARVectorProperty,
)

try:
    import atomicrex
except ImportError:
    pass


class ARStructureContainer:
    __version__ = "0.3.0"
    __hdf_version__ = "0.3.0"

    def __init__(self, num_atoms=0, num_structures=0):
        self.fit_properties = DataContainer(table_name="fit_properties")
        self._structures = StructureStorage(
            num_atoms=num_atoms, num_structures=num_structures
        )
        self._predefined_storage = DataContainer(table_name="predefined_structures")
        self._structures.add_array("fit", dtype=bool, per="chunk", fill=True)
        self._structures.add_array("clamp", dtype=bool, per="chunk", fill=True)
        self._structures.add_array("predefined", dtype=bool, per="chunk", fill=False)
        self._structures.add_array("relative_weight", per="chunk", fill=1.0)
        self.structure_file_path = None
        try:
            self._interactive_library = atomicrex.Job()
            for path in state.settings.resource_paths:
                path = posixpath.join(
                    state.settings.resource_paths[0], "atomicrex/util/main.xml"
                )
                if os.path.isfile(path):
                    self._interactive_library.parse_input_file(path)
                    break
        except:
            pass
        self._prepared_plotting = False

    def add_structure(
        self,
        structure,
        identifier,
        fit=True,
        relative_weight=1,
        clamp=True,
    ):
        if "/" in identifier:
            raise ValueError(
                "Structure identifiers must not contain '/'. "
                "Use .structure_file_path to use existing POSCAR files"
            )

        self._structures.add_structure(structure, identifier)
        self._structures._per_chunk_arrays["fit"][
            self._structures.prev_chunk_index
        ] = fit
        self._structures._per_chunk_arrays["relative_weight"][
            self._structures.prev_chunk_index
        ] = relative_weight
        self._structures._per_chunk_arrays["clamp"][
            self._structures.prev_chunk_index
        ] = clamp
        self._structures._per_chunk_arrays["predefined"][
            self._structures.prev_chunk_index
        ] = False

    def add_predefined_structure(
        self,
        identifier,
        lattice,
        lattice_parameter,
        atom_type_A,
        ca_ratio=None,
        atom_type_B=None,
        fit=True,
        relative_weight=1,
        clamp=True,
    ):
        data = {
            "alat": lattice_parameter,
        }
        if atom_type_B is None:
            data["type"] = atom_type_A
        else:
            data["type_A"] = atom_type_A
            data["type_B"] = atom_type_B
        if ca_ratio is not None:
            data["ca_ratio"] = ca_ratio
        struct = self._interactive_library.add_library_structure(
            identifier, lattice, data
        )
        struct = ase_to_pyiron(struct.get_atoms(self._interactive_library))

        self._structures.add_structure(struct, identifier)
        self._structures._per_chunk_arrays["fit"][
            self._structures.prev_chunk_index
        ] = fit
        self._structures._per_chunk_arrays["relative_weight"][
            self._structures.prev_chunk_index
        ] = relative_weight
        self._structures._per_chunk_arrays["clamp"][
            self._structures.prev_chunk_index
        ] = clamp
        self._structures._per_chunk_arrays["predefined"][
            self._structures.prev_chunk_index
        ] = True

        storage = DataContainer(table_name=identifier)
        storage["lattice"] = lattice
        storage["lattice_parameter"] = lattice_parameter
        storage["ca_ratio"] = ca_ratio
        storage["atom_type_A"] = atom_type_A
        storage["atom_type_B"] = atom_type_B
        self._predefined_storage[identifier] = storage

    def add_scalar_fit_property(
        self,
        prop="atomic-energy",
        target_val=np.nan,
        fit=True,
        relax=False,
        relative_weight=1,
        residual_style=0,
        output=True,
        tolerance=np.nan,
        min_val=np.nan,
        max_val=np.nan,
    ):
        try:
            flat = self.fit_properties[prop]
        except KeyError:
            self.fit_properties[prop] = FlattenedARScalarProperty(
                num_chunks=self._structures.num_chunks,
                num_elements=self._structures.num_elements,
            )
            flat = self.fit_properties[prop]
            if prop in ["lattice-parameter", "ca-ratio"]:
                flat.add_array("min_val", per="chunk")
                flat.add_array("max_val", per="chunk")
        try:
            flat._per_chunk_arrays["target_val"][
                self._structures.prev_chunk_index
            ] = target_val
        except (IndexError, ValueError):
            for v in self.fit_properties.values():
                v._resize_chunks(self._structures._num_chunks_alloc)
                v._resize_elements(self._structures._num_elements_alloc)
            flat._per_chunk_arrays["target_val"][
                self._structures.prev_chunk_index
            ] = target_val
        flat._per_chunk_arrays["fit"][self._structures.prev_chunk_index] = fit
        flat._per_chunk_arrays["relax"][self._structures.prev_chunk_index] = relax
        flat._per_chunk_arrays["relative_weight"][
            self._structures.prev_chunk_index
        ] = relative_weight
        flat._per_chunk_arrays["residual_style"][
            self._structures.prev_chunk_index
        ] = residual_style
        flat._per_chunk_arrays["output"][self._structures.prev_chunk_index] = output
        flat._per_chunk_arrays["tolerance"][
            self._structures.prev_chunk_index
        ] = tolerance
        if prop in ["lattice-parameter", "ca-ratio"]:
            flat._per_chunk_arrays["min_val"][
                self._structures.prev_chunk_index
            ] = min_val
            flat._per_chunk_arrays["max_val"][
                self._structures.prev_chunk_index
            ] = max_val

    def add_vector_fit_property(
        self,
        prop="atomic-forces",
        target_val=None,
        fit=True,
        relax=False,
        relative_weight=1,
        residual_style=0,
        tolerance=np.nan,
        output=True,
    ):
        try:
            flat = self.fit_properties[prop]
        except KeyError:
            self.fit_properties[prop] = FlattenedARVectorProperty(
                num_chunks=self._structures.num_chunks,
                num_elements=self._structures.num_elements,
            )
            flat = self.fit_properties[prop]
        try:
            flat._per_chunk_arrays["fit"][self._structures.prev_chunk_index] = fit
        except (IndexError, ValueError):
            for v in self.fit_properties.values():
                v._resize_chunks(self._structures._num_chunks_alloc)
            flat._per_chunk_arrays["fit"][self._structures.prev_chunk_index] = fit

        if target_val is not None:
            try:
                flat._per_element_arrays["target_val"][
                    self._structures.prev_element_index : self._structures.current_element_index
                ] = target_val
            except (IndexError, ValueError):
                for v in self.fit_properties.values():
                    v._resize_elements(self._structures._num_elements_alloc)
                flat._per_element_arrays["target_val"][
                    self._structures.prev_element_index : self._structures.current_element_index
                ] = target_val
        flat._per_chunk_arrays["relax"][self._structures.prev_chunk_index] = relax
        flat._per_chunk_arrays["relative_weight"][
            self._structures.prev_chunk_index
        ] = relative_weight
        flat._per_chunk_arrays["residual_style"][
            self._structures.prev_chunk_index
        ] = residual_style
        flat._per_chunk_arrays["output"][self._structures.prev_chunk_index] = output
        flat._per_chunk_arrays["tolerance"][
            self._structures.prev_chunk_index
        ] = tolerance

    def _get_per_structure_index(self, identifier):
        """
        Takes an identifier or an ndarray of indentifiers and returns the corresponding indices in the structure/scalar_fit_properties arrays.
        Args:
            identifiers ([type]): [description]

        Returns:
            [ndarray]: indices corresponding to identifiers in per structure arrays
        """
        if not isinstance(identifier, ndarray):
            identifier = np.array(identifier)
        indices = np.flatnonzero(
            np.isin(
                self._structures._per_chunk_arrays["identifier"],
                identifier,
                assume_unique=True,
            )
        )
        # This is some sorting magic that could lead to strange errors
        # Look here if something goes wrong with the ordering
        ids_stored = np.array(self._structures._per_chunk_arrays["identifier"][indices])
        sorter = ids_stored.argsort()[identifier.argsort()]
        return indices[sorter]

    def get_scalar_property(self, prop, identifier, final=True):
        """
        Returns final or target value of a scalar property based on the identifier used when adding a structure
        If identifier is an array of identifiers the return value is an array of values.

        Args:
            prop (str): property string as used when adding the property
            identifier (str or array(str)): identifier as used when adding the structure
            final (bool, optional): Whether to return the final or target value. Defaults to True.

        Returns:
            [type]: [description]
        """
        index = self._get_per_structure_index(identifier)
        if final:
            return self.fit_properties[prop]._per_chunk_arrays["final_val"][index]
        else:
            return self.fit_properties[prop]._per_chunk_arrays["target_val"][index]

    def get_vector_property(self, prop, identifier, final=True):
        """
        Returns final or target value of a vector property based on the identifier used when adding a structure.
        Currently only allows to return a vector property for a single structure, not for multiple ones like the
        get_scalar_property function.

        Args:
            prop (str): property string as used when adding the property
            identifier (str): identifier as used when adding the structure
            final (bool, optional): Whether to return the final or target value. Defaults to True.

        Returns:
            [type]: [description]
        """
        if not isinstance(identifier, str):
            raise NotImplementedError(
                "Can only look up properties for single identifiers currently"
            )
        index = self._get_per_structure_index(identifier)[0]
        slc = self._structures._get_per_element_slice(index)
        if final:
            return self.fit_properties[prop]._per_element_arrays["final_val"][slc]
        else:
            return self.fit_properties[prop]._per_element_arrays["target_val"][slc]

    def _sync(self):
        for flat in self.fit_properties.values():
            flat.num_elements = self._structures.num_elements
            flat.num_chunks = self._structures.num_chunks

    def _shrink(self):
        self._resize_all(
            num_chunks=self._structures.num_chunks,
            num_elements=self._structures.num_elements,
        )

    def _resize_all(self, num_chunks, num_elements):
        self._structures.num_elements = num_elements
        self._structures.num_chunks = num_chunks
        self._structures._resize_elements(self._structures.num_elements)
        self._structures._resize_chunks(self._structures.num_chunks)
        for flat in self.fit_properties.values():
            flat.num_elements = num_elements
            flat.num_chunks = num_chunks
            flat._resize_elements(num_elements)
            flat._resize_chunks(num_chunks)

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
        self._shrink()
        with hdf.open(group_name) as h:
            self._type_to_hdf(h)
            self._structures.to_hdf(hdf=h)
            self.fit_properties.to_hdf(
                hdf=h,
            )
            self._predefined_storage.to_hdf(hdf=h)
            h["structure_file_path"] = self.structure_file_path

    def from_hdf(self, hdf, group_name="structures"):
        with hdf.open(group_name) as h:
            version = h.get("HDF_VERSION", "0.1.0")
            # Compatibility Old and new StructureStorage
            if version == "0.1.0":
                num_structures = h["flattened_structures/num_structures"]
                num_atoms = h["flattened_structures/num_atoms"]
                group_name_2 = "flattened_structures"
            else:
                try:
                    num_structures = h["structures/num_chunks"]
                    num_atoms = h["structures/num_elements"]
                except:
                    num_structures = h["structures/num_structures"]
                    num_atoms = h["structures/num_atoms"]
                group_name_2 = "structures"

            self._resize_all(num_chunks=num_structures, num_elements=num_atoms)
            self._structures.from_hdf(hdf=h, group_name=group_name_2)
            self.structure_file_path = h["structure_file_path"]

            if version == "0.3.0":
                self.fit_properties.from_hdf(hdf=h)
                self._predefined_storage.from_hdf(hdf=h)
            else:
                with h.open("fit_properties") as g:
                    for k in g.list_groups():
                        if k == "atomic-forces":
                            self.fit_properties[k] = FlattenedARVectorProperty(
                                num_chunks=num_structures, num_elements=num_atoms
                            )
                        else:
                            self.fit_properties[k] = FlattenedARScalarProperty(
                                num_chunks=num_structures, num_elements=num_atoms
                            )
                        self.fit_properties[k].from_hdf(hdf=g, group_name=k)

            if version < "0.3.0":
                self._structures._per_chunk_arrays["clamp"] = h["clamp"]
                self._structures._per_chunk_arrays["fit"] = h["fit"]
                self._structures._per_chunk_arrays["relative_weight"] = h[
                    "relative_weight"
                ]

    def _check_identifiers(self):
        identifiers = self._structures.get_array("identifier")
        if not np.all(np.char.find(identifiers, "/") == -1):
            raise ValueError(
                "Structure identifiers must not contain '/'. "
                "Use .structure_file_path to use existing POSCAR files"
            )

    def write_xml_file(self, directory, name="structures.xml"):
        """
        Internal helper function that writes an atomicrex style
        xml file containg all structures.

        Args:
            directory (string): Working directory.
            name (str, optional): . Defaults to "structures.xml".
        """
        self._check_identifiers()
        self._shrink()
        root = ET.Element("group")
        if self.structure_file_path is None and "atomic-forces" in self.fit_properties:
            # write POSCARs
            for i in range(self._structures.num_chunks):
                vec_start = self._structures.start_index[i]
                vec_end = self._structures.start_index[i] + self._structures.length[i]
                forces = self.fit_properties["atomic-forces"]._per_element_arrays[
                    "target_val"
                ][vec_start:vec_end]
                if not self._structures._per_chunk_arrays["predefined"][i]:
                    write_modified_poscar(
                        identifier=self._structures.identifier[i],
                        forces=forces,
                        positions=self._structures.positions[vec_start:vec_end],
                        symbols=self._structures.symbols[vec_start:vec_end],
                        cell=self._structures.cell[i],
                        directory=directory,
                    )
                # Maybe implement a check for forces when setting up a predefined structure?
                # else:
                #    if not np.all(np.isnan(forces)):

        # write xml
        for i in range(self._structures.num_chunks):
            fit_properties_xml = ET.Element("properties")
            if not self._structures._per_chunk_arrays["predefined"][i]:
                for prop, flat_prop in self.fit_properties.items():
                    if not prop in ("lattice-parameter", "ca-ratio"):
                        fit_properties_xml.append(flat_prop.to_xml_element(i, prop))
                struct_xml = structure_meta_xml(
                    identifier=self._structures.identifier[i],
                    relative_weight=self._structures._per_chunk_arrays[
                        "relative_weight"
                    ][i],
                    clamp=self._structures._per_chunk_arrays["clamp"][i],
                    fit_properties=fit_properties_xml,
                    struct_file_path=self.structure_file_path,
                    fit=self._structures._per_chunk_arrays["fit"][i],
                )
            else:
                for prop, flat_prop in self.fit_properties.items():
                    fit_properties_xml.append(flat_prop.to_xml_element(i, prop))
                data = self._predefined_storage[self._structures.identifier[i]]
                struct_xml = predefined_structure_xml(
                    identifier=self._structures.identifier[i],
                    lattice=data["lattice"],
                    lattice_param=data["lattice_parameter"],
                    ca_ratio=data["ca_ratio"],
                    atom_type_A=data["atom_type_A"],
                    atom_type_B=data["atom_type_B"],
                    relative_weight=self._structures._per_chunk_arrays[
                        "relative_weight"
                    ][i],
                    clamp=self._structures._per_chunk_arrays["clamp"][i],
                    fit_properties=fit_properties_xml,
                    fit=self._structures._per_chunk_arrays["fit"][i],
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
                    start_index = self._structures.start_index[s_index]
                    self.fit_properties["atomic-forces"]._per_element_arrays[
                        "final_val"
                    ][start_index : start_index + len_struct] = final_forces

            # This has to be if and not else because it has to run in the same iteration. Empty lines get skipped.
            if not force_vec_triggered and l:
                if l.startswith("Structure"):
                    s_id = l.split("'")[1]
                    s_index = np.nonzero(self._structures.identifier == s_id)[0][0]

                else:
                    if not l.startswith("atomic-forces avg/max:"):
                        l = l.split()
                        prop, f_val = l[0].rstrip(":"), float(l[1])
                        if prop in self.fit_properties.keys():
                            self.fit_properties[prop]._per_chunk_arrays["final_val"][
                                s_index
                            ] = f_val
                    else:
                        force_vec_triggered = True
                        len_struct = self._structures._per_chunk_arrays["length"][
                            s_index
                        ]
                        final_forces = np.empty((len_struct, 3))

    @property
    def plot(self):
        """
        :class:`.TrainingPlots`: plotting interface
        """
        return TrainingPlots(self._structures)

    def prepare_plotting(self, final_values=False):
        val_str = "target_val"
        if final_values:
            val_str = "final_val"
        self._structures._per_chunk_arrays["energy"] = (
            self.fit_properties["atomic-energy"]._per_chunk_arrays[val_str]
            * self._structures.length
        )

    #### PotentialFit methods
    def add_training_data(self, container: TrainingContainer) -> None:
        storage = container._container
        atomic_energy_storage = FlattenedARScalarProperty(
            num_chunks=storage.num_chunks, num_elements=storage.num_elements
        )
        atomic_energy_storage.num_chunks = storage.num_chunks
        atomic_energy_storage.num_elements = storage.num_elements
        atomic_energy_storage._per_chunk_arrays["fit"][0 : storage.num_chunks] = True
        atomic_energy_storage._per_chunk_arrays["tolerance"][
            0 : storage.num_chunks
        ] = 0.001
        atomic_energy_storage._per_chunk_arrays["target_val"] = (
            storage._per_chunk_arrays["energy"][0 : storage.num_chunks]
            / storage.length[0 : storage.num_chunks]
        )

        atomic_forces_storage = FlattenedARVectorProperty(
            num_chunks=storage.num_chunks, num_elements=storage.num_elements
        )
        atomic_forces_storage.num_chunks = storage.num_chunks
        atomic_forces_storage.num_elements = storage.num_elements
        atomic_forces_storage._per_chunk_arrays["fit"][0 : storage.num_chunks] = True
        atomic_forces_storage._per_chunk_arrays["tolerance"][
            0 : storage.num_chunks
        ] = 0.01
        atomic_forces_storage._per_element_arrays["target_val"] = (
            storage._per_element_arrays["forces"][0 : storage.num_elements]
        )

        if "atomic-energy" not in self.fit_properties:
            self.fit_properties["atomic-energy"] = FlattenedARScalarProperty(
                num_chunks=self._structures.num_chunks,
                num_elements=self._structures.num_elements,
            )
        if "atomic-forces" not in self.fit_properties:
            self.fit_properties["atomic-forces"] = FlattenedARVectorProperty(
                num_chunks=self._structures.num_chunks,
                num_elements=self._structures.num_elements,
            )
        self._sync()

        self._structures.extend(storage)
        self.fit_properties["atomic-energy"].extend(atomic_energy_storage)
        self.fit_properties["atomic-forces"].extend(atomic_forces_storage)

    def _to_TrainingStorage(self, final: bool = False) -> TrainingStorage:
        self._shrink()
        storage = TrainingStorage()
        storage.extend(self._structures)
        if final:
            val_str = "final_val"
        else:
            val_str = "target_val"
        storage._per_chunk_arrays["energy"][0 : storage.num_chunks] = (
            self.fit_properties["atomic-energy"][val_str]
            * self._structures._per_chunk_arrays["length"]
        )
        storage.add_array(name="forces", shape=(3,), per="element")
        storage._per_element_arrays["forces"][0 : storage.num_elements] = (
            self.fit_properties["atomic-forces"][val_str]
        )
        return storage

    def get_training_data(self) -> TrainingStorage:
        return self._to_TrainingStorage(final=False)

    def get_predicted_data(self) -> FlattenedStorage:
        return self._to_TrainingStorage(final=True)

    @property
    def training_data(self):
        return self.get_training_data()

    @property
    def predicted_data(self):
        return self.get_predicted_data()


### This is probably useless like this in most cases because forces can't be passed.
def user_structure_to_xml_element(structure):
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


def write_modified_poscar(
    identifier,
    forces,
    directory,
    structure=None,
    positions=None,
    cell=None,
    symbols=None,
):
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
    with open(filename, "w") as f:
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
        f.write("1.0\n")
        if cell is None:
            cell = structure.cell
        for vec in cell:
            for v in vec:
                f.write(f"{v} ")
            f.write("\n")

        # Element names
        for elem in elements:
            f.write(f"{elem} ")
        f.write("\n")

        # Number of elements per type
        for elem in elements:
            n = len(symbols[symbols == elem])
            f.write(f"{n}    ")
        f.write("\n")

        # Scaled coordinates
        if positions is None:
            positions = structure.positions
        f.write("Cartesian\n")

        # Coordinates and forces
        for pos, force in zip(positions, forces):
            for r in pos:
                f.write((f"{r} "))
            f.write("   ")
            for r in force:
                f.write((f"{r} "))
            f.write("\n")


def structure_meta_xml(
    identifier,
    relative_weight,
    clamp,  ## Not sure if and how this combines with relax in the ARFitParameter sets
    fit_properties,
    struct_file_path,
    fit,
    mod_poscar=True,
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


def predefined_structure_xml(
    identifier,
    lattice,
    lattice_param,
    ca_ratio,
    atom_type_A,
    atom_type_B,
    relative_weight,
    clamp,  ## Not sure if and how this combines with relax in the ARFitParameter sets
    fit_properties,
    fit,
):
    """
    Internal function. Creates xml element for predefined structure.
    Args:
        identifier (str): Unique identifier.
        relative_weight (float): weight in objective function
        clamp (bool): clamp the structure (no relaxation)
        fit (bool): whether to fit the structure

    Returns:
        [ElementTree xml element]: atomicrex predefined structure xml element.
    """

    struct_xml = ET.Element(f"{lattice}-lattice")
    struct_xml.set("id", f"{identifier}")
    struct_xml.set("relative-weight", f"{relative_weight}")
    if fit:
        struct_xml.set("fit", "true")
    else:
        struct_xml.set("fit", "false")

    if atom_type_A is None:
        raise ValueError("atom type A has to be given for predefined structures")
    else:
        if atom_type_B is None:
            atomA = ET.SubElement(struct_xml, "atom-type")
        else:
            atomA = ET.SubElement(struct_xml, "atom-type-A")
            atomB = ET.SubElement(struct_xml, "atom-type-B")
            atomB.text = atom_type_B
        atomA.text = atom_type_A

    if lattice_param is None:
        raise ValueError("lattice parameter has to be set for predefined structures")
    else:
        a = ET.SubElement(struct_xml, "lattice-parameter")
        a.text = f"{lattice_param}"

    if ca_ratio is not None:
        ca = ET.SubElement(struct_xml, "ca-ratio")
        ca.text = f"{ca_ratio}"

    if not clamp:
        relax_dof = ET.SubElement(struct_xml, "relax-dof")
        ET.SubElement(relax_dof, "atom-coordinates")

    if isinstance(fit_properties, ARFitPropertyList):
        properties = ET.SubElement(struct_xml, "properties")
        for prop in fit_properties.values():
            properties.append(prop.to_xml_element())
    else:
        struct_xml.append(fit_properties)
    return struct_xml


def sort_structure(structure, forces=None):
    sort_array = get_sort_array(structure)
    if forces is None:
        return structure[sort_array]
    return structure[sort_array], forces[sort_array]


def get_sort_array(structure):
    sort_array = np.argsort(structure.numbers, kind="stable")
    return sort_array
