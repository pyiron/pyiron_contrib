# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import re
import warnings

import numpy as np
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_base import GenericParameters, Settings
from pyiron_atomistics.dft.job.generic import GenericDFTJob, get_k_mesh_by_density

__author__ = "Yury Lysogorskiy"
__copyright__ = "Copyright 2020, ICAMS-RUB "
__version__ = "1.0"
__maintainer__ = ""
__email__ = ""
__status__ = "trial"
__date__ = "Aug 11, 2020"

FHI_OUT_KEYWORD_AIMS_UUID_TAG = "aims_uuid"
EXPORT_AIMS_UUID = FHI_OUT_KEYWORD_AIMS_UUID_TAG
EXPORT_FHI_AIMS_VERSION = "fhi-aims-version"
EXPORT_FHI_AIMS_PARAMETERS = "fhi-aims-parameters"
EXPORT_TOTAL_TIME = "total_time"

DEFAULT_IONIC_STEPS = 1000

s = Settings()


def job_input_to_dict(inp):
    dct = inp.get_pandas()[["Parameter", "Value"]].set_index("Parameter")["Value"].to_dict()
    return {k: v for k, v in dct.items() if k != ""}


class FHIAims(GenericDFTJob):
    def __init__(self, project, job_name):
        super(FHIAims, self).__init__(project, job_name)
        self.__name__ = "FHIaims"
        self._executable_activate(enforce=True)
        self.input = FHIAimsInput()

    @property
    def exchange_correlation_functional(self):
        return self.input.control_input["xc"]

    @exchange_correlation_functional.setter
    def exchange_correlation_functional(self, val):
        self.input.control_input["xc"] = val

    def set_mixing_parameters(
            self,
            method=None,
            n_pulay_steps=None,
            density_mixing_parameter=None,
            spin_mixing_parameter=None,
    ):
        if method is not None:
            self.input.control_input["mixer"] = method
        if n_pulay_steps is not None:
            self.input.control_input["n_max_pulay"] = n_pulay_steps
        if density_mixing_parameter is not None:
            self.input.control_input["charge_mix_param"] = density_mixing_parameter

        if spin_mixing_parameter is not None:
            raise NotImplementedError(
                "spin_mixing_parameter is not (yet) implemented for FHIaims interface."
            )

    def _set_kpoints(
            self,
            mesh=None,
            scheme="GC",
            center_shift=None,
            symmetry_reduction=True,
            manual_kpoints=None,
            weights=None,
            reciprocal=True,
            n_trace=None,
            trace=None,
    ):
        if self.structure is not None:
            is_pbc = np.all(self.structure.pbc)
            if not is_pbc and mesh is not None:
                raise ValueError("Couldn't setup mesh for non-periodic structure calculation")

        if mesh is not None:
            if len(mesh) != 3:
                raise ValueError("kpoint 'mesh' should be a length-3 array of ints")
            self.input.control_input["k_grid"] = " ".join([str(int(m)) for m in mesh])
        else:
            del self.input.control_input["k_grid"]

        if scheme != "GC":
            raise NotImplementedError(("{} k-points scheme is not (yet) implemented. Only Gamma-centered (GC) is " +
                                       "possible, but ignored").format(scheme))

        if center_shift is not None:
            if len(center_shift) != 3:
                raise ValueError("kpoint 'center_shift' should be a length-3 array of ints")
            self.input.control_input["k_offset"] = " ".join([str(int(m)) for m in center_shift])
        else:
            del self.input.control_input["k_offset"]

        # TODO: use k_points_external
        if manual_kpoints is not None:
            raise NotImplementedError("manual_kpoints is not (yet) implemented for FHIaims interface.")

        if symmetry_reduction is True and mesh is not None:
            self.input.control_input["symmetry_reduced_k_grid"] = ".true."
        elif mesh is not None:
            self.input.control_input["symmetry_reduced_k_grid"] = ".false."
        else:
            del self.input.control_input["symmetry_reduced_k_grid"]

        if weights is not None:
            raise NotImplementedError("weights for k-points is not (yet) implemented for FHIaims interface.")

        if n_trace is not None:
            raise NotImplementedError("n_trace for k-points is not (yet) implemented for FHIaims interface.")

        if trace is not None:
            raise NotImplementedError("trace for k-points is not (yet) implemented for FHIaims interface.")

    def set_kpoints(
            self,
            mesh=None,
            scheme="GC",
            center_shift=None,
            symmetry_reduction=True,
            manual_kpoints=None,
            weights=None,
            reciprocal=True,
            kpoints_per_angstrom=None,
            n_trace=None,
            trace=None,
            kmesh_density_per_inverse_angstrom=None
    ):
        """
        Function to setup the k-points

        Args:
            mesh (list): Size of the mesh (ignored if scheme is not set to 'MP' or kpoints_per_angstrom is set)
            scheme (str): Type of k-point generation scheme (MP/GC(gamma centered)/GP(gamma point)/Manual/Line)
            center_shift (list): Shifts the center of the mesh from the gamma point by the given vector in relative coordinates
            symmetry_reduction (boolean): Tells if the symmetry reduction is to be applied to the k-points
            manual_kpoints (list/numpy.ndarray): Manual list of k-points
            weights(list/numpy.ndarray): Manually supplied weights to each k-point in case of the manual mode
            reciprocal (bool): Tells if the supplied values are in reciprocal (direct) or cartesian coordinates (in
            reciprocal space)
            kpoints_per_angstrom (float): Number of kpoint per angstrom in each direction
            n_trace (int): Number of points per trace part for line mode
            trace (list): ordered list of high symmetry points for line mode
            kmesh_density_per_inverse_angstrom (float): spacing of kpoints (recommended value is 0.1 for tight settings)
        """

        if kmesh_density_per_inverse_angstrom is not None:
            if mesh is not None:
                warnings.warn("mesh value is overwritten by kmesh_density_per_inverse_angsrtrom")

            self.input.set_kmesh_density(kmesh_density_per_inverse_angstrom)

            if self.structure is not None:
                is_pbc = np.all(self.structure.pbc)
                if is_pbc:
                    self.input.update_kmesh(self.structure)
                    mesh = self.get_k_mesh_by_density(
                        kmesh_density_per_inverse_angstrom=kmesh_density_per_inverse_angstrom)
                else:
                    mesh = None
        else:
            if mesh is None:
                if self.input.kmesh_density_per_inverse_angstrom is not None:
                    mesh = self.get_k_mesh_by_density(kmesh_density_per_inverse_angstrom=self.input.kmesh_density_per_inverse_angstrom)
            self.input.set_kmesh_density(kmesh_density_per_inverse_angstrom)


        if kpoints_per_angstrom is not None:
            if mesh is not None:
                warnings.warn("mesh value is overwritten by kpoints_per_angstrom")
            mesh = self.get_k_mesh_by_cell(kpoints_per_angstrom=kpoints_per_angstrom)
        if mesh is not None:
            if np.min(mesh) <= 0:
                raise ValueError("mesh values must be larger than 0")
        if center_shift is not None:
            if np.min(center_shift) < 0 or np.max(center_shift) > 1:
                warnings.warn("center_shift is given in relative coordinates")
        self._set_kpoints(
            mesh=mesh,
            scheme=scheme,
            center_shift=center_shift,
            symmetry_reduction=symmetry_reduction,
            manual_kpoints=manual_kpoints,
            weights=weights,
            reciprocal=reciprocal,
            n_trace=n_trace,
            trace=trace,
        )

    def calc_static(
            self,
            electronic_steps=500,
            algorithm=None,
            retain_charge_density=False,
            retain_electrostatic_potential=False,
    ):
        self._generic_input["fix_symmetry"] = True
        super(GenericDFTJob, self).calc_static()

        if electronic_steps is not None:
            self.input.control_input["sc_iter_limit"] = electronic_steps
        else:
            del self.input.control_input["sc_iter_limit"]

        del self.input.control_input["relax_geometry"]
        del self.input.control_input["relax_unit_cell"]

        if algorithm is not None:
            raise NotImplementedError("calc_static.algorithm parameter is not (yet) implemented for FHIaims interface.")

        if retain_charge_density:
            raise NotImplementedError(
                "calc_static.retain_charge_density=True parameter is not (yet) implemented for FHIaims interface.")

        if retain_electrostatic_potential:
            raise NotImplementedError(
                "calc_static.retain_electrostatic_potential=True parameter is not (yet) implemented for FHIaims interface.")

    def calc_minimize(self,
                      electronic_steps=60,
                      ionic_steps=DEFAULT_IONIC_STEPS,
                      max_iter=None,
                      pressure=None,
                      algorithm="bfgs",
                      retain_charge_density=False,
                      retain_electrostatic_potential=False,
                      ionic_force_tolerance=1e-2
                      # volume_only
                      ):
        self._generic_input["fix_symmetry"] = True
        super(GenericDFTJob, self).calc_minimize(max_iter=max_iter, pressure=pressure)

        if electronic_steps is not None:
            self.input.control_input["sc_iter_limit"] = electronic_steps
        else:
            del self.input.control_input["sc_iter_limit"]

        if ionic_steps != DEFAULT_IONIC_STEPS:
            raise NotImplementedError(
                "Different number of ionic steps rather than {} is not supported".format(DEFAULT_IONIC_STEPS))
        if max_iter is not None:
            raise NotImplementedError(
                "`max_iter` is ignored".format(DEFAULT_IONIC_STEPS))

        if pressure is None:  # atomic positions only
            # relax_geometry type tolerance
            algorithm_str = "none" if algorithm is None else str(algorithm)
            if ionic_force_tolerance is None:
                self.input.control_input["relax_geometry"] = algorithm_str
            else:
                self.input.control_input["relax_geometry"] = "{algo} {tol}".format(algo=algorithm_str,
                                                                                   tol=ionic_force_tolerance)
            del self.input.control_input["external_pressure"]
        elif pressure == 0.0:  # full relaxation
            # relax_geometry type tolerance
            # relax_unit_cell type
            is_pbc = np.all(self.structure.pbc)
            if not is_pbc:
                raise ValueError("Couldn't relax non-periodic structure")

            algorithm_str = "none" if algorithm is None else str(algorithm)
            if ionic_force_tolerance is None:
                self.input.control_input["relax_geometry"] = algorithm_str
            else:
                self.input.control_input["relax_geometry"] = "{algo} {tol}".format(algo=algorithm_str,
                                                                                   tol=ionic_force_tolerance)
            self.input.control_input["relax_unit_cell"] = "full"
            del self.input.control_input["external_pressure"]
        else:
            # self.input.control_input["external_pressure"] = pressure
            raise ValueError("'pressure' could be only None or 0.0")

    def calc_md(
            self,
            temperature=None,
            n_ionic_steps=1000,
            n_print=1,
            time_step=1.0,
            retain_charge_density=False,
            retain_electrostatic_potential=False,
            **kwargs
    ):
        raise NotImplementedError("calc_md not (yet) implemented")

    def set_input_to_read_only(self):
        """
        This function enforces read-only mode for the input classes, but it has to be implement in the individual
        classes.
        """
        super(FHIAims, self).set_input_to_read_only()
        self.input.control_input.read_only = True
        self.input.control_potential.read_only = True

    def write_input(self):
        # methods, called externally
        self.input.write(structure=self.structure, working_directory=self.working_directory)

    def collect_output(self):
        output_dict, output_dft_dict, meta_info_dict = collect_output(working_directory=self.working_directory,
                                                     FHI_output_file='FHI.out')

        with self.project_hdf5.open("output") as hdf5_output:
            with hdf5_output.open("generic") as hdf5_generic:
                for k, v in output_dict.items():
                    hdf5_generic[k] = v
            with hdf5_output.open("dft") as hdf5_dft:
                for k, v in output_dft_dict.items():
                    hdf5_dft[k] = v
            hdf5_output["meta_info"] = meta_info_dict

    def to_hdf(self, hdf=None, group_name=None):
        super(FHIAims, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.structure.to_hdf(hdf5_input)
            self.input.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None, group_name=None):
        super(FHIAims, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)
            self.structure = Atoms().from_hdf(hdf5_input)

    @property
    def structure(self):
        """

        Returns:

        """
        return self._structure

    @structure.setter
    def structure(self, basis):
        """

        Args:
            basis:

        Returns:

        """
        self._generic_input["structure"] = "atoms"
        self._structure = basis
        # TODO: update settings depending on the PBC
        is_pbc = np.all(self.structure.pbc)
        if is_pbc:
            self.set_pbc_settings()
        else:
            self.set_non_pbc_settings()

    def set_pbc_settings(self):
        # kpoints, stress
        if self.input.kmesh_density_per_inverse_angstrom is not None:
            self.set_kpoints(kmesh_density_per_inverse_angstrom=self.input.kmesh_density_per_inverse_angstrom)
        else:
            if not self.input.control_input["k_grid"]:
                self.set_kpoints(mesh=[1, 1, 1])

        self.input.control_input["symmetry_reduced_k_grid"] = ".true."
        self.input.control_input["compute_analytical_stress"] = ".true."

    def set_non_pbc_settings(self):
        # remove kpoints, remove stress
        del self.input.control_input["k_grid"]
        del self.input.control_input["k_offset"]
        del self.input.control_input["symmetry_reduced_k_grid_spg"]
        del self.input.control_input["symmetry_reduced_k_grid"]
        del self.input.control_input["compute_analytical_stress"]
        del self.input.control_input["compute_numerical_stress"]


class FHIAimsControlInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(FHIAimsControlInput, self).__init__(input_file_name=input_file_name,
                                                  table_name="control_in",
                                                  comment_char="#")

    def load_default(self):
        """
        Loading the default settings for the input file.
        """

        input_str = """\
xc                 pbe
charge             0.
spin               none
occupation_type    gaussian 0.10
mixer              pulay
n_max_pulay        10
charge_mix_param   0.05   
sc_iter_limit      500
sc_accuracy_rho  1E-5
sc_accuracy_eev  1E-3
sc_accuracy_etot 1E-7
relativistic       atomic_zora scalar
compute_forces .true.
clean_forces sayvetz
sc_accuracy_forces 1E-4
final_forces_cleaned .true.
"""
        self.load_string(input_str)


class FHIAimsControlPotential(GenericParameters):
    def __init__(self, input_file_name=None):
        super(FHIAimsControlPotential, self).__init__(input_file_name=input_file_name,
                                                      table_name="control_potential",
                                                      comment_char="#")
        self._structure = None

    def load_default(self):
        """
        Loading the default settings for the input file.
        """
        input_str = """\
potential          tight  # Options: light, tight, really_tight
"""
        self.load_string(input_str)

    def set_structure(self, structure):
        self._structure = structure
        chem_symb = self._structure.get_chemical_symbols()
        atom_numb = self._structure.get_atomic_numbers()
        chem_symb_dict = {k: v for (k, v) in zip(chem_symb, atom_numb)}
        self._chem_symb_lst = sorted(chem_symb_dict.items())

    def _return_potential_file(self, file_name):
        for resource_path in s.resource_paths:
            resource_path_potcar = os.path.join(
                resource_path, "fhiaims", "potentials", self["potential"], file_name
            )
            if os.path.exists(resource_path_potcar):
                return resource_path_potcar
        return None

    def get_string_lst(self):
        settings = self["potential"]
        lines = []
        for elem, atom_num in self._chem_symb_lst:
            file_name = "{atom_num:02d}_{elem}_default".format(atom_num=atom_num, elem=elem)
            full_potential_file_name = self._return_potential_file(file_name)
            if full_potential_file_name is None:
                raise ValueError("Couldn't read file {} for settings '{}'".format(file_name, settings))
            with open(full_potential_file_name, "r") as f:
                lines += f.readlines()
        return lines


class FHIAimsInput:
    def __init__(self):
        self.control_input = FHIAimsControlInput()
        self.control_potential = FHIAimsControlPotential()
        self._kmesh_density_per_inverse_angstrom = None

    @property
    def kmesh_density_per_inverse_angstrom(self):
        return self._kmesh_density_per_inverse_angstrom

    @kmesh_density_per_inverse_angstrom.setter
    def kmesh_density_per_inverse_angstrom(self, val):
        self._kmesh_density_per_inverse_angstrom = val

    def set_kmesh_density(self, kmesh_density_per_inverse_angstrom):
        self.kmesh_density_per_inverse_angstrom = kmesh_density_per_inverse_angstrom

    def update_kmesh(self, structure):
        if (self.kmesh_density_per_inverse_angstrom is not None) and (structure is not None):
            if self.kmesh_density_per_inverse_angstrom != 0.0:
                k_mesh = get_k_mesh_by_density(
                    structure.get_cell(),
                    kmesh_density_per_inverse_angstrom=self.kmesh_density_per_inverse_angstrom,
                )
                print("kmesh_density_per_inverse_angstrom = ", self.kmesh_density_per_inverse_angstrom)
                print("Update k-mesh =", k_mesh)
                self.control_input["k_grid"] = " ".join([str(int(m)) for m in k_mesh])

    def write(self, structure, working_directory):
        self.control_potential.set_structure(structure)
        control_in_filename = os.path.join(working_directory, "control.in")
        control_in_lst = self.control_input.get_string_lst()
        control_in_lst += self.control_potential.get_string_lst()

        with open(control_in_filename, "w") as f:
            print("".join(control_in_lst), file=f)

        pbc = structure.pbc
        is_periodic = np.all(pbc)
        if not is_periodic and not np.all(~pbc):
            raise ValueError("Structure for FHI-aims could be either fully periodic or fully non-periodic")

        lines = ["# pyiron generated geometry.in"]
        if is_periodic:
            cell = structure.get_cell()
            for lattice_vec in cell:
                lines.append(
                    "lattice_vector {:.15f} {:.15f} {:.15f}".format(lattice_vec[0], lattice_vec[1], lattice_vec[2]))
        lines.append("")

        chem_symbs = structure.get_chemical_symbols()
        positions = structure.get_positions()

        for symb, pos in zip(chem_symbs, positions):
            lines.append("atom {:.15f} {:.15f} {:.15f}   {}".format(pos[0], pos[1], pos[2], symb))

        with open(os.path.join(working_directory, "geometry.in"), "w") as f:
            print("\n".join(lines), file=f)

    def to_hdf(self, hdf=None):
        with hdf.open("control_input") as hdf5_input:
            self.control_input.to_hdf(hdf5_input)

        with hdf.open("control_potential") as hdf5_input:
            self.control_potential.to_hdf(hdf5_input)

        vasp_dict = hdf["vasp_dict"] if "vasp_dict" in hdf.list_nodes() else {}

        if self.kmesh_density_per_inverse_angstrom is not None:
            vasp_dict.update({"kmesh_density_per_inverse_angstrom":
                                  self.kmesh_density_per_inverse_angstrom})
        if len(vasp_dict) > 0:
            hdf["vasp_dict"] = vasp_dict

    def from_hdf(self, hdf=None):
        with hdf.open("control_input") as hdf5_input:
            self.control_input.from_hdf(hdf5_input)
        with hdf.open("control_potential") as hdf5_input:
            self.control_potential.from_hdf(hdf5_input)

        vasp_dict = hdf["vasp_dict"] if "vasp_dict" in hdf.list_nodes() else {}
        if "kmesh_density_per_inverse_angstrom" in vasp_dict.keys():
            self.kmesh_density_per_inverse_angstrom = vasp_dict["kmesh_density_per_inverse_angstrom"]


class InitialGeometryStreamParser:
    def __init__(self):
        self._input_geometry_block_flag = False
        self._inp_geom_unit_cell_flag = False

        # accumulator lists for current atomic structure
        self.lattice_vectors_lst = []
        self.positions_lst = []
        self.species_lst = []

        self._stop_processing = False

    def process_line(self, line):
        if self._stop_processing:
            return

        line = line.strip(" \t\n")

        if not line.startswith("|") and self._input_geometry_block_flag:
            self._input_geometry_block_flag = False
            self._stop_processing = True

        if line.startswith("Input geometry:"):
            self._input_geometry_block_flag = True

        if self._input_geometry_block_flag and line.startswith("|"):
            line = line.strip(" \t\n|")

            if len(self.lattice_vectors_lst) >= 3:
                self._inp_geom_unit_cell_flag = False

            if self._inp_geom_unit_cell_flag:
                self.lattice_vectors_lst.append([float(s) for s in line.split()[:3]])

            if line.startswith("Unit cell:"):
                self._inp_geom_unit_cell_flag = True

            if "Species" in line:
                line_tags = line.split()
                atom_positions = [float(s) for s in line_tags[3:6]]
                atom_species = line_tags[2]

                self.positions_lst.append(atom_positions)
                self.species_lst.append(atom_species)


class UpdateGeometryStreamParser:
    def __init__(self):
        self._atomic_structure_block_flag = False

        # accumulator lists for all atoomic structures
        self.lattice_vectors_traj = []
        self.positions_traj = []
        self.species_traj = []

        # accumulator lists for current atomic structure
        self._lattice_vectors_lst = []
        self._positions_lst = []
        self._species_lst = []

    def process_line(self, line):

        line = line.strip(" \t\n")

        if line.startswith("Updated atomic structure:"):  # or line.startswith("Final atomic structure:"):
            self._atomic_structure_block_flag = True

        if line.startswith("-------------") and self._atomic_structure_block_flag:
            self._atomic_structure_block_flag = False

            # save collected structure
            if len(self._lattice_vectors_lst) > 0:
                self.lattice_vectors_traj.append(self._lattice_vectors_lst)
            self.positions_traj.append(self._positions_lst)
            self.species_traj.append(self._species_lst)

            # reset accumulator lists
            self._lattice_vectors_lst = []
            self._positions_lst = []
            self._species_lst = []

        if self._atomic_structure_block_flag:

            if "lattice_vector" in line:
                self._lattice_vectors_lst.append([float(s) for s in line.split()[1:4]])

            if line.startswith("atom "):
                line_tags = line.split()
                atom_positions = [float(s) for s in line_tags[1:4]]
                atom_species = line_tags[4]

                self._positions_lst.append(atom_positions)
                self._species_lst.append(atom_species)


class EnergyForcesStressesStreamParser:
    def __init__(self):
        self.free_energies_list = []
        self.energies_corrected_list = []
        self.energies_uncorrected_list = []

        self.forces_lst = []
        self.stresses_lst = []

        self.block_flag = False
        self.force_block_flag = False
        self.stress_block_flag = False

        self.stress_line_counter = 0
        self.current_forces = []
        self.current_stresses = []

    def process_line(self, line):
        if "Energy and forces in a compact form:" in line:
            self.block_flag = True

        if self.block_flag and "------------------------------------" in line:
            self.block_flag = False
            self.force_block_flag = False
            self.forces_lst.append(self.current_forces)

        if self.block_flag and 'Total energy corrected        :' in line:
            E0 = float(line.split()[5])
            self.energies_corrected_list.append(E0)
        elif self.block_flag and 'Electronic free energy        :' in line:
            F = float(line.split()[5])
            self.free_energies_list.append(F)
        elif self.block_flag and 'Total energy uncorrected      :' in line:
            E_uncorr = float(line.split()[5])
            self.energies_uncorrected_list.append(E_uncorr)

        if self.block_flag and "Total atomic forces" in line:
            self.force_block_flag = True
            self.current_forces = []

        if self.force_block_flag and line.strip().startswith("|"):
            self.current_forces.append([float(f) for f in line.split()[-3:]])

        if "|              Analytical stress tensor" in line or "Numerical stress tensor" in line:
            self.stress_block_flag = True
            self.current_stresses = []
            self.stress_line_counter = 0

        if self.stress_block_flag:
            self.stress_line_counter += 1

        if self.stress_block_flag and self.stress_line_counter in [6, 7, 8]:
            sline = [float(f) for f in line.split()[2:5]]
            self.current_stresses.append(sline)

        if self.stress_line_counter > 8:
            self.stress_block_flag = False
            self.stress_line_counter = 0
            self.stresses_lst.append(self.current_stresses)


class MetaInfoStreamParser:
    _total_time_pattern = re.compile("Total time\s*:\s*([0-9.]*)\s*s")

    FHI_OUT_KEYWORD_TOTAL_TIME = "Total time"
    FHI_OUT_KEYWORD_AIMS_UUID_TAG = "aims_uuid"
    FHI_OUT_KEYWORD_VERSION_TAG = "Version"

    def __init__(self):
        self.version = None
        self.aims_uuid = None
        self.total_time = None

    def process_line(self, line):

        line = line.strip(" \t\n")

        if line.startswith(self.FHI_OUT_KEYWORD_VERSION_TAG):
            self.version = " ".join(line.split()[1:])
        elif line.startswith(self.FHI_OUT_KEYWORD_AIMS_UUID_TAG):
            self.aims_uuid = " ".join(line.split(":")[1:]).strip()

        if (self.FHI_OUT_KEYWORD_TOTAL_TIME in line) and (self.total_time is None):
            line = line.strip()
            res = self._total_time_pattern.findall(line)
            if len(res) > 0:
                self.total_time = res[0]


def collect_output(working_directory="", FHI_output_file="FHI.out"):
    FHI_output_file = os.path.join(working_directory, FHI_output_file)

    init_geom_stream_parser = InitialGeometryStreamParser()
    upd_geom_stream_parser = UpdateGeometryStreamParser()
    efs_stream_parser = EnergyForcesStressesStreamParser()
    metainfo_stream_parser = MetaInfoStreamParser()

    with open(FHI_output_file, 'r') as f:
        for line in f:
            line = line.strip(" \t\n")
            if line.startswith("#"):
                continue

            init_geom_stream_parser.process_line(line)
            upd_geom_stream_parser.process_line(line)
            efs_stream_parser.process_line(line)
            metainfo_stream_parser.process_line(line)

    if len(efs_stream_parser.free_energies_list) == 0 or len(efs_stream_parser.forces_lst) == 0:
        raise ValueError("No free electronic energies found. Calculation could be broken")

    if len(init_geom_stream_parser.lattice_vectors_lst) > 0:
        lattice_vectors_traj = [
                                   init_geom_stream_parser.lattice_vectors_lst] + upd_geom_stream_parser.lattice_vectors_traj
    else:
        lattice_vectors_traj = upd_geom_stream_parser.lattice_vectors_traj

    positions_traj = [init_geom_stream_parser.positions_lst] + upd_geom_stream_parser.positions_traj

    if len(positions_traj) == 0:
        raise ValueError("No cells or positions info found. Calculation could be broken")

    output_generic_dict = {
        'cells': np.array(lattice_vectors_traj),
        'energy_pot': np.array(efs_stream_parser.free_energies_list),
        'energy_tot': np.array(efs_stream_parser.free_energies_list),
        'forces': np.array(efs_stream_parser.forces_lst),
        'positions': np.array(positions_traj),
        # 'steps'
        # 'temperature'
        # 'computation_time'
        # 'unwrapped_positions'
        # 'indices'
    }

    output_dft_dict = {
        'free_energy': np.array(efs_stream_parser.free_energies_list),
        'energy_corrected': np.array(efs_stream_parser.energies_corrected_list),
        'energy_uncorrected': np.array(efs_stream_parser.energies_uncorrected_list),
    }

    if len(efs_stream_parser.stresses_lst) > 0:
        output_generic_dict["stresses"] = efs_stream_parser.stresses_lst
        stresses = output_generic_dict["stresses"]
        output_generic_dict["pressures"] = np.array([-np.trace(stress) / 3.0 for stress in stresses])

    if len(output_generic_dict["cells"]) > 0:
        cells = output_generic_dict["cells"]
        output_generic_dict["volume"] = np.array([np.linalg.det(cell) for cell in cells])

    meta_info_dict = {}
    if metainfo_stream_parser.version is not None:
        meta_info_dict[EXPORT_FHI_AIMS_VERSION] = metainfo_stream_parser.version
    if metainfo_stream_parser.aims_uuid is not None:
        meta_info_dict[EXPORT_AIMS_UUID] = metainfo_stream_parser.aims_uuid
    if metainfo_stream_parser.total_time is not None:
        meta_info_dict[EXPORT_TOTAL_TIME] = metainfo_stream_parser.total_time

    return output_generic_dict, output_dft_dict, meta_info_dict
