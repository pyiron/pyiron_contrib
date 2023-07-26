import numpy as np
from hashlib import sha1
from tqdm.auto import tqdm
from .base import get_potential, ProjectContainer, Hydrogen
from sklearn.gaussian_process import GaussianProcessRegressor


def get_job_name(structure):
    return sha1(structure.__repr__().encode()).hexdigest()


class Points:
    """
    Class to create a meshgrid and sort the points by box symmetry.

    Exp:
    >>> points = Points(structure)
    >>> unique_data = [do_something(xx) for xx in points.unique_positions]
    >>> total_data = points.unravel(unique_data)
    """
    def __init__(self, structure, mesh_distance=0.2):
        """
        Args:
            structure (pyiron_atomistics.atomistics.structure.atoms): structure
                for which the meshgrid is to be produced
            mesh_distance (float): mesh spacing
        """
        self.mesh_distance = mesh_distance
        self._labels = None
        self._structure = structure

    @property
    def _n_points(self):
        return np.rint(
            self._structure.cell.diagonal() / self.mesh_distance / 2
        ).astype(int) * 2

    @property
    def _lin_space(self):
        linspace_lst = []
        for ll, nn in zip(self._structure.cell.diagonal(), self._n_points):
            linspace, dx = np.linspace(0, ll, nn, endpoint=False, retstep=True)
            linspace_lst.append(linspace + 0.5 * dx)
        return linspace_lst

    @property
    def _meshgrid(self):
        meshgrid = np.meshgrid(*self._lin_space, indexing='ij')
        return np.einsum('xijk->ijkx', meshgrid)

    @property
    def _meshgrid_flat(self):
        return self._meshgrid.reshape(-1, 3)

    @property
    def _symmetry(self):
        return self._structure.get_symmetry()

    @property
    def _all_labels(self):
        if self._labels is None:
            self._labels = self._symmetry.get_arg_equivalent_sites(self._meshgrid_flat)
        return self._labels

    @property
    def unique_positions(self):
        """
        Unique positions among the meshgrid.
        """
        return self._meshgrid_flat[self._unique_indices]

    @property
    def min_distance(self):
        """
        Minimum distance of a grid point and atoms. This should not be too
        small for the simulation to not explode. Currently, there is no
        automatic check system implemented, meaning this attribute is not
        internally used anywhere.
        """
        return self._structure.get_neighborhood(
            self.unique_positions, num_neighbors=1
        ).distances.min()

    @property
    def _uniques(self):
        return np.unique(
            self._all_labels, return_index=True, return_counts=True
        )

    @property
    def _unique_counts(self):
        return self._uniques[2]

    @property
    def _unique_labels(self):
        return self._uniques[0]

    @property
    def _unique_indices(self):
        return self._uniques[1]

    def unravel(self, **kwargs):
        """
        Args:
            kwargs (numpy.ndarray): Data to unravel (must have the same length
                as unique_positions)

        Returns:
            (numpy.ndarray): unravelled data
        """
        results = {"positions": self._lin_space}
        for k, v in kwargs.items():
            y_all = np.zeros(len(self._all_labels))
            for ll, yy in zip(self._unique_labels, v):
                y_all[self._all_labels==ll] = yy
            results[k] = y_all.reshape(*self._n_points)
        return results

    def get_position_from_index(self, index):
        """
        Args:
            index (int): index

        Returns:
            (numpy.ndarray): position for the index
        """
        unraveled_index = np.unravel_index(index, self._n_points)
        return self._meshgrid[unraveled_index]


def setup_lmp_input(lmp, n_atoms=None, direction=None, fix_id=-1):
    """
    Change input for LAMMPS to run a drag calculation.

    Args:
        lmp (pyiron_atomistics.lammps.lammps.Lammps): LAMMPS job
        n_atoms (int): number of free atoms (default: None, i.e. it is
            determined from the job structure)
        direction (None/numpy.ndarray): direction along which the force is
            cancelled. None if all forces are to be cancelled (default: None)
        fix_id (None/int): id of the atom to be fixed (default: -1, i.e. last
            atom)

    Returns:
        None (input of lmp is changed in-place)

    In the context of this function, a drag calculation is a constraint energy
    minimization, in which one atom is either not allowed to move at all, or
    not allowed to move along a given direction. In order for the system to not
    fall to the energy minimum, the sum of the remaining forces is set to 0.

    Exp: Hydrogen diffusion

    >>> from pyiron_atomistics import Project
    >>> pr = Project("DRAG")
    >>> bulk = pr.create.structure.bulk('Ni', cubic=True)
    >>> a_0 = bulk.cell[0, 0]
    >>> x_octa = np.array([0, 0, 0.5 * a_0])
    >>> x_tetra = np.array(3 * [0.25 * a_0])
    >>> dx = x_tetra - x_octa
    >>> transition = np.linspace(0, 1, 101)
    >>> x_lst = transition[:, None] * dx + x_octa
    >>> structure = bulk.repeat(4) + pr.create.structure.atoms(
    ...     positions=[x_octa],
    ...     elements=['H'],
    ...     cell=structure.cell
    ... )
    >>> lmp = pr.create.job.Lammps('lmp')
    >>> lmp.structure = structure
    >>> lmp.calc_minimize()
    >>> lmp.potential = potential_of_your_choice
    >>> setup_lmp_input(lmp, direction=dx)
    >>> lmp.interactive_open()
    >>> for xx in x_lst:
    >>>     lmp.structure.positions[-1] = xx
    >>>     lmp.run()
    >>> lmp.interactive_close()
    """
    if lmp.input.control["minimize"] is None:
        raise ValueError("set calc_minimize first")
    if n_atoms is None:
        try:
            n_atoms = len(lmp.structure) - 1
        except TypeError:
            raise AssertionError("either `n_atoms` or the structure must be set")
    fix_id = np.arange(n_atoms)[fix_id] + 2
    lmp.input.control['atom_modify'] = 'map array'
    lmp.input.control["group___fixed"] = f"id {fix_id}"
    lmp.input.control["group___free"] = "subtract all fixed"
    if direction is None:
        for ii, xx in enumerate(['x', 'y', 'z']):
            lmp.input.control[f'variable___f{xx}_free'] = f'equal f{xx}[{fix_id}]/{n_atoms}'
            lmp.input.control[f'variable___f{xx}_fixed'] = f'equal -f{xx}[{fix_id}]'
    else:
        direction = np.array(direction) / np.linalg.norm(direction)
        direction = np.outer(direction, direction)
        direction = np.around(direction, decimals=8)
        for grp, ss in zip(["free", "fixed"], [f"1/{n_atoms}*", "-"]):
            for ii, xx in enumerate(['x', 'y', 'z']):
                txt = "+".join([f"({ss}f{xxx}[{fix_id}]*({direction[ii][iii]}))" for iii, xxx in enumerate(['x', 'y', 'z'])])
                lmp.input.control[f'variable___f{xx}_{grp}'] = f" equal {txt}"
    lmp.input.control['variable___energy'] = "atom 0"
    for key in ["free", "fixed"]:
        txt = " ".join([f"v_f{x}_{key}" for x in ["x", "y", "z"]])
        lmp.input.control[f"fix___f_{key}"] = f"{key} addforce {txt} energy v_energy"
    lmp.input.control['min_style'] = 'quickmin'


class Drag(ProjectContainer):
    """
    Run drag calculation

    Internal routines:

    1. Create a meshgrid for the box inserted
    2. Create a Lammps job, to which the repeated box is inserted, if the box
        length does not exceed `min_length`.
    3. Compute the Voronoi vertices
    4. Run drag calculations for the Voronoi vertices
    5. Suggest a new point new sample point based on the covariance matrix

    The data can be accessed via `drag.field`, in which case the data for the
    meshgrid is given, or `drag.data`, which contains the positions of
    measurements including their symmetrically equivalent points, as well as
    their energy values.

    Exp:

    >>> import matplotlib.pylab as plt
    >>> from pyiron_atomistics import Project
    >>> pr = Project("DRAG")
    >>> bulk = pr.create.structure.bulk('Ni', cubic=True)
    >>> drag = Drag(pr, bulk)
    >>> drag.run()
    >>> layer = 0
    >>> for tag in ['energy', 'error']:
    ...     plt.contourf(*drag.field['positions'][:2], drag.field[tag].T[layer]);
    ...     plt.colorbar().set_label(tag)
    ...     plt.show();

    This gives the energy landscape in the lowest layer.
    """
    def __init__(
        self,
        pr,
        structure,
        min_length=10,
        mesh_distance=0.2,
        buffer=3,
        distance_threshold=0.1,
    ):
        """
        Args:
            pr (pyiron_atomistics.project.Project): Project
            structure (pyiron_atomistics.atomistics.structure.atoms): structure
            min_length (float): minimum box length (default: 10)
            mesh_distance (float): mesh spacing
            buffer (float): distance up to which the atoms beyond box
                boundaries are considered
            distance_threshold (float): distance below which the Voronoi
                vertices are considered to be one point
        """
        super().__init__(pr=pr)
        self._min_length = min_length
        self._structure = structure
        self._buffer = buffer
        self._lmp = None
        self._data = None
        self._field = None
        self._gp = None
        self._points = Points(structure, mesh_distance=mesh_distance)
        self._distance_threshold = distance_threshold

    @property
    def lmp(self):
        """Job attribute"""
        if self._lmp is None:
            self._lmp = self.pr.create.job.Lammps(
                ('lmp', sha1(self._structure.__repr__().encode()).hexdigest())
            )
            self._lmp.potential = get_potential()
            self._lmp.calc_minimize(n_print=1000)
            setup_lmp_input(self._lmp, len(self._rep_structure))
            self._lmp.interactive_open()
        return self._lmp

    def _initialize(self):
        self._data = None
        self._field = None
        self._gp = None

    @property
    def _next_positions(self):
        if self.lmp.status.initialized:
            return self._unique_voro
        return self._points.get_position_from_index(np.argmax(self.field['error']))

    def run(self, positions=None):
        """
        Run command.

        Args:
            positions (numpy.ndarray): Positions for which the energy sampling
                should take place. None if the new position should be
                given either from the Voronoi vertices (at the beginning) or
                calculated from the maximum error value given by the Gaussian
                process.

        Returns:
            None
        """
        if positions is None:
            positions = self._next_positions
        for xx in tqdm(np.atleast_2d(positions)):
            self.lmp.structure = self.append_hydrogen(xx)
            self.lmp.run()
        self._initialize()

    @property
    def gaussian_process(self):
        """Gaussian process regressor"""
        if self._gp is None:
            xx, ii = self._structure.get_extended_positions(
                self._buffer, return_indices=True, positions=self.data['positions']
            )
            self._gp = GaussianProcessRegressor().fit(xx, self.data['energy'][ii])
        return self._gp

    @property
    def field(self):
        if self._field is None:
            E, err = self.gaussian_process.predict(
                self._points.unique_positions, return_std=True
            )
            self._field = self._points.unravel(energy=E, error=err)
        return self._field

    @property
    def data(self):
        if self._data is None:
            x = self._symmetry.generate_equivalent_points(
                self.lmp.output.positions[:, -1], return_unique=False
            )
            E = np.tile(self.lmp.output.energy_pot - self._ref_energy, len(x))
            x = x.reshape(-1, 3)
            unique_indices = np.unique(
                np.round(x, decimals=3), return_index=True, axis=0
            )[1]
            self._data = {
                "positions": x[unique_indices],
                "energy": E[unique_indices],
            }
        return self._data

    @property
    def _repeat(self):
        return np.ceil(self._min_length / self._structure.cell.diagonal()).astype(int)

    @property
    def _rep_structure(self):
        return self._structure.repeat(self._repeat)

    @property
    def _symmetry(self):
        return self._structure.get_symmetry()

    @property
    def _unique_voro(self):
        voro = self._structure.analyse.get_voronoi_vertices(
            distance_threshold=self._distance_threshold
        )
        ids = self._symmetry.get_arg_equivalent_sites(voro)
        return voro[np.unique(ids, return_index=True)[1]]

    def append_hydrogen(self, x):
        return self._rep_structure + self.pr.create.structure.atoms(
            elements=['H'], positions=[x], cell=self._rep_structure.cell
        )

    @property
    def _energy_hydrogen(self):
        h = Hydrogen(3, self.pr)
        return h.E_octa

    @property
    def _energy_base(self):
        return self.get_minimize(self._structure).output.energy_pot[-1]

    @property
    def _ref_energy(self):
        return self._energy_hydrogen + np.prod(self._repeat) * self._energy_base
