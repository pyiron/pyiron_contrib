# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from scipy import stats
from scipy.constants import physical_constants

from pyiron_base.jobs.job.generic import GenericJob
from pyiron_base.storage.datacontainer import DataContainer
from pyiron_atomistics.atomistics.structure.atoms import Atoms

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

KB = physical_constants['Boltzmann constant in eV/K'][0]
myc = {'r': (1.0, 0.0, 44. / 255.), 'b': (71. / 255., 0.0, 167. / 255.), 'o': (1.0, 180. / 255., 7. / 255.),
       'g': (0.0 / 255.0, 180. / 255., 7. / 255.)}


class _BAInput(DataContainer):
    """
    Class to store input parameters for the Bond Analysis classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._structure = None
        self._n_shells = None
        self._cutoff_radius = None

    @property
    def structure(self) -> Atoms:
        return self._structure

    @structure.setter
    def structure(self, atoms: Atoms):
        if not isinstance(atoms, Atoms):
            raise TypeError(f'structure must be of type Atoms but got {type(atoms)}')
        self._structure = atoms

    @property
    def n_shells(self):
        return self._n_shells

    @n_shells.setter
    def n_shells(self, n):
        if not isinstance(n, int):
            raise TypeError(f"n_shells must be an integer but got {type(n)}")
        self._n_shells = n

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @cutoff_radius.setter
    def cutoff_radius(self, r):
        if not isinstance(r, (int, float)):
            raise TypeError(f"n_shells must be an integer/float but got {type(r)}")
        self._cutoff_radius = r


class _BondAnalysisParent(GenericJob):
    def __init__(self, project, job_name):
        super(_BondAnalysisParent, self).__init__(project, job_name)
        self._python_only_job = True
        self.input = _BAInput(table_name="job_input")
        self.output = DataContainer(table_name="job_output")
        self._nn = None
        self._n_bonds_per_shell = None
        self._all_nn_bond_vectors = None
        self.histogram = _Histograms()

    def validate_ready_to_run(self):
        """
        Check if necessary inputs are provided, and everything is in order for the computation to run.
        """
        if self.input.structure is None:
            raise AttributeError('<job>.input.structure must be set')
        if (self.input.n_shells is None) and (self.input.cutoff_radius is None):
            raise AttributeError('either <job>.input.n_shells or <job>.input.cutoff_radius must be set')

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the StructureToBonds object in the HDF5 File.

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(_BondAnalysisParent, self).to_hdf()
        self.input.to_hdf(self.project_hdf5)
        self.output.to_hdf(self.project_hdf5)

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the StructureToBonds object from the HDF5 File.

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super(_BondAnalysisParent, self).from_hdf()
        self.input.from_hdf(self.project_hdf5)
        self.output.from_hdf(self.project_hdf5)


class StaticBondAnalysis(_BondAnalysisParent):
    """
    A job that analyzes the bond relations in a minimized structure.
    """

    def __init__(self, project, job_name):
        super(StaticBondAnalysis, self).__init__(project, job_name)

    @staticmethod
    def _set_cutoff_radius(structure, n_shells, cutoff_radius=None, eps=1e-5):
        """
        If cutoff radius is not specified as an input, set it based off of the n_shells.
        Args:
            structure:
            n_shells:
            cutoff_radius:
            eps:

        Returns:
            cutoff_radius
        """
        if cutoff_radius is None:
            nn = structure.get_neighbors(num_neighbors=1000)
            all_shells = nn.shells[0]
            needed_dists = len(all_shells[all_shells < n_shells + 1])
            cutoff_radius = nn.distances[0][:needed_dists][-1] + eps
        return cutoff_radius

    @staticmethod
    def _get_nn(structure, cutoff_radius):
        """
        Return the neighbors object.
        Args:
            structure:
            cutoff_radius:

        Returns:
            neighbors
            n_bonds_per_shell
        """
        neighbors = structure.get_neighbors(num_neighbors=None, cutoff_radius=cutoff_radius)
        nn_distances = np.around(neighbors.distances, decimals=5)
        _, _, n_bonds_per_shell = np.unique(np.around(nn_distances[0], decimals=5), return_inverse=True,
                                            return_counts=True)
        return neighbors, n_bonds_per_shell

    @staticmethod
    def _set_n_shells(nn_distances, n_shells=None):
        """
        If n_shells is not specified as an input, set it based off of the cutoff radius.
        Args:
            nn_distances:
            n_shells:

        Returns:
            n_shells
        """
        if n_shells is None:
            nn_dists = np.around(nn_distances[0], decimals=5)
            n_shells = int(np.unique(nn_dists, return_index=True)[1][-1] + 1)
        return n_shells

    @staticmethod
    def _populate_shells(n_bonds_per_shell, vectors, indices):
        """
        Arrange data according to shells.
        Args:
            n_bonds_per_shell:
            vectors:
            indices:

        Returns:
            vectors_per_shell
        """
        vectors_per_shell = []
        sums = 0
        for n in n_bonds_per_shell:
            sorted_indices = np.argsort(indices[sums:sums + n])
            vectors_per_shell.append(vectors[sums:sums + n][sorted_indices])
            sums += n
        return vectors_per_shell

    @staticmethod
    def _rotation_matrix_from_vectors(vec_1, vec_2):
        """
        Find the rotation matrix that aligns vec_1 to vec_2.
        Args:
            vec_1: A 3d "source" vector
            vec_2: A 3d "destination" vector

        Returns:
            A transformation matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec_1 / np.linalg.norm(vec_1)).reshape(3), (vec_2 / np.linalg.norm(vec_2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if np.any(v):  # if not all zeros then
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        elif not np.any(v) and c < 0.:
            return np.eye(3) * -1  # for opposite directions
        else:
            return np.eye(3)  # for identical directions

    def _get_irreducible_bond_vector_per_shell(self, nn, n_bonds_per_shell):
        """
        Get one irreducible bond vector per nearest neighbor shell. If symmetries are known, this irreducible bond can
            be used to generate all other bonds for the corresponding shell.
        """
        # arrange bond vectors according to their shells
        bond_vectors_per_shell = self._populate_shells(vectors=np.around(nn.vecs[0], decimals=5),
                                                       indices=nn.indices[0],
                                                       n_bonds_per_shell=n_bonds_per_shell)
        # save 1 irreducible bond vector per shell
        per_shell_irreducible_bond_vectors = np.array([b[0] for b in bond_vectors_per_shell])
        # get all the rotations from spglib
        data = self.input.structure.get_symmetry()
        # account for only 0 translation rotations
        all_rotations = data['rotations'][np.argwhere(data['translations'][:48].sum(-1) < 1e-10).flatten()]
        # and sort them doing the following:
        per_shell_0K_rotations = []
        all_nn_bond_vectors_list = []
        for b in enumerate(per_shell_irreducible_bond_vectors):
            # get 'unique bonds' from the spglib data
            unique_bonds = np.unique(np.dot(b[1], all_rotations), axis=0)
            args_list = []
            for bond in unique_bonds:
                # collect the arguments of the rotations that that give the unique bonds
                args = []
                for i, bd in enumerate(np.dot(b[1], all_rotations)):
                    if np.array_equal(np.round(bd, decimals=5), np.round(bond, decimals=5)):
                        args.append(i)
                args_list.append(args[0])
            # sort the arguments and append the rotations to a list...
            per_shell_0K_rotations.append(all_rotations[np.sort(args_list)])
            # and the unique bonds also, to another list...
            all_nn_bond_vectors_list.append(np.dot(b[1], per_shell_0K_rotations[-1]))
        # and clump the unique bonds into a single array
        all_nn_bond_vectors_list = np.array([bonds for per_shell_bonds in all_nn_bond_vectors_list
                                             for bonds in per_shell_bonds])
        return per_shell_irreducible_bond_vectors, per_shell_0K_rotations, all_nn_bond_vectors_list

    def _get_transformations(self, n_bonds_per_shell, all_nn_bond_vectors):
        """
        Get the transformation matrices that take the bond vectors of each shell from cartesian [x, y, z] axes to
            [longitudinal, transverse1 and transverse2] axes.
        Args:
            n_bonds_per_shell:
            all_nn_bond_vectors:

        Returns:
            per_shell_transformations
        """
        sums = 0
        per_shell_transformation_matrices = []
        for i in n_bonds_per_shell:
            bonds = all_nn_bond_vectors[sums:sums + i].copy()
            sums += i
            # normalize the bonds
            bonds /= np.linalg.norm(bonds, axis=1)[:, np.newaxis]
            transformation_matrices = []
            for b in bonds:
                b1 = b.copy()  # first bond is the longitudinal bond
                # second bond is normal to the first (transverse1). If multiple normal bonds, select 1
                b2 = bonds[np.argwhere(np.round(bonds@b1, decimals=5) == 0.).flatten()[0]]
                # third bond is then  normal to both the first and second bonds (transverse2)
                b3 = np.cross(b1, b2)
                if b1.dot(np.cross(b2, b3)) < 0.:  # if this condition is not met
                    b2, b3 = b3, b2  # reverse b2 and b3
                transformation_matrices.append(np.array([b1, b2, b3]))
            per_shell_transformation_matrices.append(transformation_matrices)
        return per_shell_transformation_matrices

    @staticmethod
    def _get_bond_indexed_neighbor_list(nn, n_bonds_per_shell, all_nn_bond_vectors, structure):
        """
        For each atom with index i, obtain the index of another atom M[i][j],
            the difference between which gives bond vector index j.

        If M_ij is the matrix, i belongs to [1,N], j belongs to [1,m], where,
            i = index of atom 1
            M_ij = index of atom 2
            j = index of the unique bond vector between atoms 1 and 2
            N = number of atoms
            m = number of unique bonds
        Args:
            nn:
            n_bonds_per_shell:
            all_nn_bond_vectors:
            structure:

        Returns:
            per_shell_bond_indexed_neighbor_list
        """
        nn_vecs = np.around(nn.vecs, decimals=5)
        nn_indices = nn.indices
        sums = 0
        per_shell_bond_indexed_neighbor_list = []
        for n in n_bonds_per_shell:
            nn_vecs_per_shell = nn_vecs[:, sums:sums + n].copy()
            nn_indices_per_shell = nn_indices[:, sums:sums + n].copy()
            bond_vectors_list = all_nn_bond_vectors[sums:sums + n].copy()
            sums += n
            # initialize the M_ij matrix with shape [atoms x unique_bonds]
            x_0 = np.around(structure.positions, decimals=5)  # zero Kelvin positions
            M_ij = np.zeros([len(x_0), len(nn_vecs_per_shell[0])]).astype(int)
            # populate the M_ij matrix
            for i, per_atom_nn in enumerate(nn_vecs_per_shell):
                for vec, ind in zip(per_atom_nn, nn_indices_per_shell[i]):
                    try:
                        j = np.argwhere(np.all(np.isclose(vec, bond_vectors_list), axis=1))[0, 0]
                    except IndexError:  # this is an exception for HCP!
                        j = np.argwhere(np.all(np.isclose(-vec, bond_vectors_list), axis=1))[0, 0]
                    M_ij[i][j] = ind
            per_shell_bond_indexed_neighbor_list.append(M_ij)
        return per_shell_bond_indexed_neighbor_list

    @staticmethod
    def _get_bond_relations_list(per_shell_bond_indexed_neighbor_list):
        """
        Use the per_shell_bond_indexed_neighbor_list for each shell to generate a 'bond relations' list of the form [[i, m_ij, j], ...]
            connecting every atom (with index i) in the structure to its nearest neighbor atom/s (with index m_ij),
            giving a bond vector/s, which can be transformed to the direction of the irreducible bond vector of that
            shell using a symmetry operation (with index j).
        Args:
            per_shell_bond_indexed_neighbor_list:

        Returns:
            per_shell_bond_relations
        """
        per_shell_bond_relations = []
        for M_ij in per_shell_bond_indexed_neighbor_list:  # enumerate over shells
            per_shell = []
            for j, row in enumerate(M_ij.T):  # enumerate over bonds
                per_bond = []
                for i, m_ij in enumerate(row):  # enumerate over atoms
                    per_bond.append([i+1, m_ij+1, j+1])  # atom_1_index, atom_2_index, symmetry_op_index
                per_shell.append(per_bond)
            per_shell_bond_relations.append(np.array(per_shell))
        return per_shell_bond_relations

    def analyze_bonds(self):
        # set cutoff radius, if not already set
        self.input.cutoff_radius = self._set_cutoff_radius(structure=self.input.structure,
                                                           n_shells=self.input.n_shells,
                                                           cutoff_radius=self.input.cutoff_radius,
                                                           eps=1e-5)
        # run get_neighbors
        self._nn, self._n_bonds_per_shell = self._get_nn(structure=self.input.structure,
                                                         cutoff_radius=self.input.cutoff_radius)
        # if n_shells is not set in the input, make sure it is now
        self.input.n_shells = self._set_n_shells(nn_distances=self._nn.distances,
                                                 n_shells=self.input.n_shells)

        # get irreducible bond vectors and 0K rotations
        irr_bvs = self._get_irreducible_bond_vector_per_shell(nn=self._nn,
                                                              n_bonds_per_shell=self._n_bonds_per_shell)
        self.output.per_shell_irreducible_bond_vectors, self.output.per_shell_0K_rotations, \
        self._all_nn_bond_vectors = irr_bvs
        # get transformations
        self.output.per_shell_transformation_matrices = self._get_transformations(
            n_bonds_per_shell=self._n_bonds_per_shell,
            all_nn_bond_vectors=self._all_nn_bond_vectors)
        # get per_shell_bond_indexed_neighbor_list
        self.output.per_shell_bond_indexed_neighbor_list = \
            self._get_bond_indexed_neighbor_list(nn=self._nn, n_bonds_per_shell=self._n_bonds_per_shell,
                                                 all_nn_bond_vectors=self._all_nn_bond_vectors,
                                                 structure=self.input.structure)
        # get bond relations
        self.output.per_shell_bond_relations = \
            self._get_bond_relations_list(per_shell_bond_indexed_neighbor_list=
                                          self.output.per_shell_bond_indexed_neighbor_list)

    def run_static(self):
        self.status.running = True
        self.analyze_bonds()
        self.to_hdf()
        self.status.finished = True


class MDBondAnalysis(_BondAnalysisParent):
    """
    A job which, given an MD trajectory, gives 'bond data' based off of the bond relations of the minimized structure.
    """

    def __init__(self, project, job_name):
        super(MDBondAnalysis, self).__init__(project, job_name)
        self.input.md_job = None
        self.input.thermalize_snapshots = 20
        self.input.md_trajectory = None
        self.input.md_cells = None
        self._structure = None
        self._md_trajectory = None
        self._md_cells = None
        
    @staticmethod
    def _find_mic(cell, vectors, pbc=[True, True, True]):
        """
        Find vectors following minimum image convention (mic).
            cell: The cell in reference to which the vectors are expressed.
            vectors (list/numpy.ndarray): 3d vector or a list/array of 3d vectors.
            pbc: Periodic bondary condition along each coordinate axis.
        Returns: numpy.ndarray of the same shape as input with mic
        """
        vecs = np.asarray(vectors).reshape(-1, 3)
        if any(pbc):
            vecs = np.einsum('ji,nj->ni', np.linalg.inv(cell), vecs)
            vecs[:, pbc] -= np.rint(vecs)[:, pbc]
            vecs = np.einsum('ji,nj->ni', cell, vecs)
        return vecs.reshape(np.asarray(vectors).shape)

    def validate_ready_to_run(self):
        """
        Check if necessary inputs are provided, and everything is in order for the computation to run.
        """          
        if self.input.md_job is not None:
            self.input.md_trajectory = self.input.md_job.output.unwrapped_positions[self.input.thermalize_snapshots:]
            self.input.md_cells = self.input.md_job.output.cells[self.input.thermalize_snapshots:]
        elif self.input.md_trajectory is None:
            raise AttributeError('Either <job>.input.md_job or <job>.input.md_trajectory must be set')
            
        static = StaticBondAnalysis(project=self.project_hdf5, job_name=self.job_name + '_static')
        static.input = self.input
        static.analyze_bonds()
        self.output = static.output

    def get_xyz_bond_vectors(self, per_atom=False):
        """
        Use the 'bond relations' list to obtain the MD bond vectors for each shell.
        """
        cell = np.mean(self.input.md_cells, axis=0)
        per_shell_xyz_bond_vectors = []
        for i, br_per_s in enumerate(self.output.per_shell_bond_relations):
            per_shell = []
            for bond in br_per_s:
                bond_vectors = self.input.md_trajectory[:, bond[:, 1] - 1] - self.input.md_trajectory[:, bond[:, 0] - 1]
                bond_vectors_mic = []
                for j, snapshot in enumerate(bond_vectors):
                    bond_vectors_mic.append(self._find_mic(cell, snapshot))
                if per_atom:
                    per_shell.append(np.array(bond_vectors_mic))
                else:
                    per_shell.append(np.array(bond_vectors_mic).reshape(-1, 3))
            per_shell_xyz_bond_vectors.append(np.array(per_shell))
        return per_shell_xyz_bond_vectors

    def get_l_t1_t2_bond_vectors(self, per_atom=False, return_xyz=False):
        """
        Convert the MD bond vectors from cartesian [x, y, z] axes to [longitudinal, transverse1 and transverse2] axes,
            using the transformation matrices for each bond in each shell.
        """
        per_shell_xyz_bond_vectors = self.get_xyz_bond_vectors(per_atom=per_atom)
        per_shell_l_t1_t2_bond_vectors = []
        for shell in np.arange(self.input.n_shells):
            per_shell = []
            for i, transform in enumerate(self.output.per_shell_transformation_matrices[shell]):
                bond_vectors = per_shell_xyz_bond_vectors[shell][i]
                transformed_vectors = np.dot(bond_vectors, transform.T)
                # the next 4 lines are an exception for HCP!
                if np.any(transformed_vectors[:, 0] < 0.):
                    for j, vec in enumerate(transformed_vectors):
                        if vec[0] < 0.:
                            transformed_vectors[j] = -vec
                per_shell.append(transformed_vectors)
            per_shell_l_t1_t2_bond_vectors.append(np.array(per_shell))
        if return_xyz:
            return per_shell_l_t1_t2_bond_vectors, per_shell_xyz_bond_vectors
        else:
            return per_shell_l_t1_t2_bond_vectors
            
    def get_r_t1_t2_bond_vectors(self, per_atom=False, return_xyz=False):
        """
        Convert the MD bond vectors from cartesian [x, y, z] axes to [radial, transverse1 and transverse2] axes,
            using the transformation matrices for each bond in each shell.
        """
        per_shell_xyz_bond_vectors = self.get_xyz_bond_vectors(per_atom=per_atom)
        per_shell_r_t1_t2_bond_vectors = []
        for shell in np.arange(self.input.n_shells):
            per_shell = []
            for i, transform in enumerate(self.output.per_shell_transformation_matrices[shell]):
                bond_vectors = per_shell_xyz_bond_vectors[shell][i]
                transformed_vectors = np.dot(bond_vectors, transform.T)
                transformed_vectors[:, 0] = np.linalg.norm(bond_vectors, axis=-1)
                # the next 4 lines are an exception for HCP!
                if np.any(transformed_vectors[:, 0] < 0.):
                    for j, vec in enumerate(transformed_vectors):
                        if vec[0] < 0.:
                            transformed_vectors[j] = -vec
                per_shell.append(transformed_vectors)
            per_shell_r_t1_t2_bond_vectors.append(np.array(per_shell))
        if return_xyz:
            return per_shell_r_t1_t2_bond_vectors, per_shell_xyz_bond_vectors
        else:
            return per_shell_r_t1_t2_bond_vectors 

#     @staticmethod
#     def _cartesian_to_cylindrical(vector):
#         """
#         Helper method for get_md_cylindrical_long_t1_t2.
#         """
#         if len(vector.shape) == 1:
#             vector = np.array([vector])
#         r = np.linalg.norm(vector[:, -2:], axis=1)
#         phi = np.arctan2(vector[:, 2], vector[:, 1])
#         return np.array([vector[:, 0], r, phi]).T

#     def _get_long_r_phi_bond_vectors(self):
#         """
#         Convert the [longitudinal, transverse1 and transverse2] which are cartesian axes to cylindrical axes
#             [longitudinal, r and phi].
#         """
#         self.output.per_shell_long_r_phi_bond_vectors = []
#         for shell in self.output.per_shell_long_t1_t2_bond_vectors:
#             per_bond = []
#             for bond in shell:
#                 per_bond.append(self._cartesian_to_cylindrical(bond))
#             self.output.per_shell_long_r_phi_bond_vectors.append(np.array(per_bond))

    def run_static(self):
        self.status.running = True
        self.to_hdf()
        self.status.finished = True

    def get_1d_histogram_xyz(self, shell=0, bond=None, n_bins=20, d_range=None, density=True, axis=0,
                             moment=True):
        per_shell_xyz_bond_vectors = self.get_xyz_bond_vectors()
        return self.histogram.get_per_shell_1d_histogram(per_shell_xyz_bond_vectors, shell=shell, bond=bond,
                                                         n_bins=n_bins, d_range=d_range, density=density, axis=axis,
                                                         moment=moment)

    def get_1d_histogram_l_t1_t2(self, shell=0, bond=None, n_bins=20, d_range=None, density=True, axis=0,
                                    moment=True):
        per_shell_l_t1_t2_bond_vectors = self.get_l_t1_t2_bond_vectors()
        return self.histogram.get_per_shell_1d_histogram(per_shell_l_t1_t2_bond_vectors, shell=shell,
                                                         bond=bond, n_bins=n_bins, d_range=d_range, density=density,
                                                         axis=axis, moment=moment)
    
    def get_1d_histogram_r_t1_t2(self, shell=0, bond=None, n_bins=20, d_range=None, density=True, axis=0,
                                    moment=True):
        per_shell_r_t1_t2_bond_vectors = self.get_r_t1_t2_bond_vectors()
        return self.histogram.get_per_shell_1d_histogram(per_shell_r_t1_t2_bond_vectors, shell=shell,
                                                         bond=bond, n_bins=n_bins, d_range=d_range, density=density,
                                                         axis=axis, moment=moment)

#     def get_1d_histogram_long_r_phi(self, shell=0, bond=None, n_bins=20, d_range=None, density=True, axis=0,
#                                     moment=True):
#         return self.histogram.get_per_shell_1d_histogram(self.output.per_shell_long_r_phi_bond_vectors, shell=shell,
#                                                          bond=bond, n_bins=n_bins, d_range=d_range, density=density,
#                                                          axis=axis, moment=moment)

    def get_3d_histogram_xyz(self, shell=0, bond=None, n_bins=20, d_range=None, density=True):
        per_shell_xyz_bond_vectors = self.get_xyz_bond_vectors()
        return self.histogram.get_per_shell_3d_histogram(per_shell_xyz_bond_vectors, supp_data=None, shell=shell, bond=bond,
                                                         n_bins=n_bins, d_range=d_range, density=density)

    def get_3d_histogram_l_t1_t2(self, shell=0, bond=None, n_bins=20, d_range=None, density=True):
        per_shell_l_t1_t2_bond_vectors, per_shell_xyz_bond_vectors = self.get_l_t1_t2_bond_vectors(return_xyz=True)
        return self.histogram.get_per_shell_3d_histogram(per_shell_l_t1_t2_bond_vectors, 
                                                         supp_data=per_shell_xyz_bond_vectors, shell=shell,
                                                         bond=bond, n_bins=n_bins, d_range=d_range, density=density)
    
    def get_3d_histogram_r_t1_t2(self, shell=0, bond=None, n_bins=20, d_range=None, density=True):
        per_shell_r_t1_t2_bond_vectors, per_shell_xyz_bond_vectors = self.get_r_t1_t2_bond_vectors(return_xyz=True)
        return self.histogram.get_per_shell_3d_histogram(per_shell_r_t1_t2_bond_vectors, 
                                                         supp_data=per_shell_xyz_bond_vectors, shell=shell,
                                                         bond=bond, n_bins=n_bins, d_range=d_range, density=density)

#     def get_3d_histogram_long_r_phi(self, shell=0, bond=None, n_bins=20, d_range=None, density=True):
#         return self.histogram.get_per_shell_3d_histogram(self.output.per_shell_long_r_phi_bond_vectors, shell=shell, supp_data=None,
#                                                          bond=bond, n_bins=n_bins, d_range=d_range, density=density)

#     def get_potential_long_r_phi(self, temperature=300., shell=0, bond=None, n_bins=20, d_range=None, density=True):
#         pd, bins = self.get_3d_histogram_long_r_phi(shell=shell, bond=bond, n_bins=n_bins, d_range=d_range,
#                                                     density=density)
#         r_bins = bins[1][0, :, 0]
#         delta_r = (r_bins[1] - r_bins[0]) / 2
#         mean_over_phi_pd = pd.mean(axis=-1)
#         # since in cylindrical coordinates, the pd of r needs to be divided by the bins,
#         mean_over_phi_pd /= np.outer(np.ones(n_bins), r_bins + delta_r)
#         mean_over_phi_pd /= mean_over_phi_pd.sum()
#         potential = -KB * temperature * np.log(mean_over_phi_pd + 1e-10)
#         potential = potential.T
#         return potential - potential.min(), np.meshgrid(bins[0][:, 0, 0], bins[1][0, :, 0])

#     def get_potential_long_t1_t2(self, temperature=300., shell=0, bond=None, n_bins=20, d_range=None, density=True,
#                                  mean_over_final_axis=False):
#         pd, bins = self.get_3d_histogram_long_t1_t2(shell=shell, bond=bond, n_bins=n_bins, d_range=d_range,
#                                                     density=density)
#         if mean_over_final_axis:
#             pd = pd.mean(axis=-1)
#         pd /= pd.sum()
#         potential = -KB * temperature * np.log(pd + 1e-10)
#         if mean_over_final_axis:
#             return (potential - potential.min()).T, np.meshgrid(bins[0][:, 0, 0], bins[1][0, :, 0])
#         else:
#             return potential - potential.min(), bins


    @staticmethod
    def _get_rho_corr_and_uncorr(x_data, y_data, n_bins=101):
        rho_corr, x_edges, y_edges = np.histogram2d(x_data, y_data, (n_bins, n_bins), density=True)
        rho_uncorr = np.outer(rho_corr.sum(axis=1), rho_corr.sum(axis=0))
        return rho_corr / rho_corr.sum(), rho_uncorr / rho_uncorr.sum(), x_edges, y_edges

    @staticmethod
    def _get_corr_column_value(axis):
        if axis == 'long':
            return 0
        elif axis == 't1':
            return 1
        elif axis == 't2':
            return 2
        else:
            raise ValueError("choose between long, t1, and t2")

    def _get_correlations(self, shell, bond_x, bond_y, axis_x, axis_y, n_bins):
        per_shell_r_t1_t2_bond_vectors = self.get_r_t1_t2_bond_vectors()
        all_bonds = per_shell_r_t1_t2_bond_vectors[shell]
        n_bonds = len(all_bonds)
        if (bond_x >= n_bonds) or (bond_y >= n_bonds):
            raise ValueError("there are only {} bonds in shell {}".format(n_bonds, shell))
        axis_0 = self._get_corr_column_value(axis_x)
        axis_1 = self._get_corr_column_value(axis_y)
        return self._get_rho_corr_and_uncorr(x_data=all_bonds[bond_y, :, axis_0], y_data=all_bonds[bond_x, :, axis_1],
                                             n_bins=n_bins)

    def get_mutual_information(self, shell=0, bond_x=0, bond_y=0, axis_x='long', axis_y='long', n_bins=101):
        rho_corr, rho_uncorr, _, _ = self._get_correlations(shell=shell, bond_x=bond_x, bond_y=bond_y, axis_x=axis_x,
                                                            axis_y=axis_y, n_bins=n_bins)
        sel = (rho_uncorr>0.0)*(rho_corr>0.0)
        return np.sum(rho_corr[sel]*np.log(rho_corr[sel]/rho_uncorr[sel]))

    def plot_correlations(self, shell=0, bond_x=0, bond_y=0, axis_x='long', axis_y='long', n_bins=101):
        rho_corr, rho_uncorr, x_edges, y_edges = self._get_correlations(shell=shell, bond_x=bond_x,
                                                                        bond_y=bond_y, axis_x=axis_x,
                                                                        axis_y=axis_y, n_bins=n_bins)
        rho_diff = rho_corr - rho_uncorr
        cm = LinearSegmentedColormap.from_list('my_spec', [myc['b'], (1, 1, 1), myc['o']], N=n_bins)
        plims = x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()  # plot limits
        plt.imshow(rho_diff[::-1, :], cmap=cm, extent=plims)
        plt.xticks(np.around(np.linspace(plims[0], plims[1], 5), decimals=2))
        plt.yticks(np.around(np.linspace(plims[2], plims[3], 5), decimals=2))
        plt.xlabel(axis_x + '_' + str(bond_x))
        plt.ylabel(axis_y + '_' + str(bond_y))
        plt.show()


class _Histograms(object):
    """
    A helper class which gives histograms of the input 'bond_representation', which is bond data of a single bond in a
        shell or all the bonds in the shell, represented in either [x, y, z] coordinates, [longitudinal, transverse1,
        transverse2] coordinates or [longitudinal, r, phi] coordinates.
    """

    @staticmethod
    def _get_bin_centers(bins):
        if len(bins) == 3:
            x_bins, y_bins, z_bins = bins
            return [(x_bins[1:] + x_bins[:-1]) / 2, (y_bins[1:] + y_bins[:-1]) / 2, (z_bins[1:] + z_bins[:-1]) / 2]
        else:
            return (bins[1:] + bins[:-1]) / 2

    @staticmethod
    def _check_histogram_bonds(required_data, bond, shell):
        """
        Helper method to get_per_shell_3d_histogram and get_per_shell_1d_histogram
        Args:
            required_data:
            bond:
            shell:

        Returns:
            required_data
        """
        if bond is None:
            required_data = required_data.reshape(-1, 3)  # flatten over all bonds in the shell
        else:
            n_bonds = len(required_data)
            if bond >= n_bonds:
                raise ValueError("there are only {} bonds in shell {}".format(n_bonds, shell))
            required_data = required_data[bond]
        return required_data

    def get_per_shell_1d_histogram(self, bond_representation, shell=0, bond=None, n_bins=20, d_range=None,
                                   density=True, axis=0, moment=True):
        """
        Get the 1d-histograms of data:
        Args:
            bond_representation: Histogram of which data to be obtained. 'bond_vectors' or 'long_t1_t2' or 'cyl_long_t1_t2'
                (Default is 'bond_vectors')
            shell: The shell (Default is 0, the 0th shell)
            bond: The bond whose 3D histogram is to be obtained (Default is None, flatten over all the bonds)
            n_bins: Number of bins along each axis (Default is 20 bins)
            d_range: The range of the bins along each axis (Default is None, set range to the min/max of the bins)
            density: If True, returns the probability densities, else returns counts (Default is True)
            axis: Along which axis to get the histogram (Default is 0, the 0th axis)
            moment: If True, returns the moments 1-4 of the distribution (Default is True)

        Returns:
            rho (bins): The counts/probability densities
            centered_bins (list[x-bins, y-bins, z-bins]): The center values of the bins
        """
        if axis not in [0, 1, 2, -1, -2, -3]:
            raise ValueError("axis can only take values 0, 1, 2, -1, -2, -3")
        required_data = bond_representation[shell]
        required_data = self._check_histogram_bonds(required_data=required_data, bond=bond, shell=shell)
        rho, bins = np.histogram(required_data[:, axis], bins=n_bins, range=d_range, density=density)
        if moment:
            moments = [np.mean(required_data[:, axis])]
            moments += [stats.moment(a=required_data[:, axis], moment=i) for i in np.arange(2, 5)]
            return rho/rho.sum(), self._get_bin_centers(bins=bins), moments
        else:
            return rho/rho.sum(), self._get_bin_centers(bins=bins)

    def get_per_shell_3d_histogram(self, bond_representation, supp_data=None, shell=0, bond=None, n_bins=20, d_range=None,
                                   density=True):
        """
        Get the 3d-histograms of data:
        Args:
            bond_representation: Histogram of which data to be obtained. 'bond_vectors' or 'long_t1_t2' or 'cyl_long_t1_t2'
                (Default is 'bond_vectors')
            shell: The shell (Default is 0, the 0th shell)
            bond: The bond whose 3D histogram is to be obtained (Default is None, flatten over all the bonds)
            n_bins: Number of bins along each axis (Default is 20 bins)
            d_range: The range of the bins along each axis (Default is None, set range to the min/max of the bins)
            density: If True, returns the probability densities, else returns counts (Default is True)

        Returns:
            rho (bins x bins x bins): The counts/probability densities
            centered_bins (list[x-bins, y-bins, z-bins]): The center values of the bins
        """
        required_data = bond_representation[shell]
        required_data = self._check_histogram_bonds(required_data=required_data, bond=bond, shell=shell)
        rho, bins = np.histogramdd(required_data, bins=n_bins, range=d_range, density=density)
        rho /= rho.sum()
        bins = self._get_bin_centers(bins=bins)
        bins_3d = np.meshgrid(bins[0], bins[1], bins[2], indexing='ij')
        return rho, bins_3d
