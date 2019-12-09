# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function

from pyiron_mpie.flexible.protocol.generic import Protocol
from pyiron_mpie.flexible.protocol.utils import ensure_iterable
from pyiron_mpie.flexible.protocol.primitive.one_state import ExternalHamiltonian, Counter, Norm, Max, GradientDescent
from pyiron_mpie.flexible.protocol.primitive.two_state import IsGEq, IsLEq
from pyiron_mpie.flexible.protocol.utils import Pointer, IODictionary
import numpy as np
from pyiron_mpie.flexible.protocol.compound.qmmm import AddDisplacements
import matplotlib.pyplot as plt

"""
Protocol for running the force-based quantum mechanics/molecular mechanics concurrent coupling scheme described in 
Huber et al., Comp. Mat. Sci. 118 (2016) 259-268. Here the coupling is explicitly intended to be MM/MM as a simple 
toy for learning.
"""

__author__ = "Dominik Noeger, Liam Huber"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "June 6, 2019"


class MMMM(Protocol):
    """
    Relax a MM/MM coupled system. Obviously that's silly, this class is just for educational purposes (since MM/MM
    coupling runs *much* faster than QM/QM coupling!)

    Input attributes:
        structure (Atoms): The structure to partition into domains and minimize.
        qm_potential (str): The classical potential to use for the 'QM' domain.
        mm_potential (str): The classical potential to use for the true MM domain.
        domain_ids (dict): A dictionary with entries 'core', 'seed', 'buffer', and 'filler', each containing a list or
            numpy.ndarray of mutually-exclusive integers defining which atoms from the structure make up which
            computational domains. Alternatively, this can be left unprovided and `seed_ids`, `n_core_shells`,
            `n_buffer_shells`, and `filler_width` can be provided instead to decompose the structure into various
            domains automatically.
        seed_ids (int/list/numpy.ndarray): Integer id from the provided structure whose species is set according to
            the `seed_species`. If `domain_ids` is not explicitly provided, these also serve as seed sites for
            decomposing the system into QM and MM domains by building neighbor shells. (Default is None, which raises an
            error unless `domain_ids` was explicitly provided -- else this parameter must be provided.)
        seed_species (str/list): The species for each 'seed' atom in the QM domain. Value(s) should be a atomic symbol,
            e.g. 'Mg'. (Default is None, which leaves all seeds the same as they occur in the MM representation.)
        shell_cutoff (float): Maximum distance for two atoms to be considered neighbours in the construction of shells
            for automatic system partitioning. (Default is None, which will raise an error -- this parameter must be
            provided.)
        n_core_shells (int): How many neighbour shells around the seed(s) to relax using QM forces. (Default is 2.)
        n_buffer_shells (int): How many neighbour shells around the region I core to relax using MM forces. (Default is
            2.)
        filler_width (float): Length added to the bounding box of the region I atoms to create a new box from which to
            draw filler atoms using the initial MM I+II superstructure. Influences the final simulation cell for the QM
            calculation. If the value is not positive, no filler atoms are used. The second, larger box uses the same
            center as the bounding box of the QM region I, so this length is split equally to the positive and negative
            sides for each cartesian direction. Negative filler width ensures NO filler atoms will be used. (Default is
            6.)
        vacuum_width (float): Minimum vacuum distance between atoms in two periodic images of the QM domain. Influences
            the final simulation cell for the QM calculation. (Default is 2.)
        n_steps (int): How many steps to run for. (Default is 100.)
        f_tol (float): Ionic force convergence (largest atomic force). (Default is 1e-4.)
        gamma0 (float): Step size for gradient descent. (Default is 0.1.)
        fix_com (bool): Whether the to fix the center of mass during minimization. (Default is True.)
        use_adagrad (bool): Whether to adjust the minimization step size using Adagrad. (Default is True.)

    Setup attributes:
        qm_structure (Atoms): The structure used for the 'QM' domain calculations. This attribute should *never* be
            explicitly set, but is constructed when `set_qm_structure` is called after *all other* input has been
            provided. `set_qm_structure` can be called explicitly by the user prior to runtime, e.g. to look at the
            structure before running the calculation, but is otherwise called automatically before the run starts.
            (Default is None, and don't touch it.)
    """

    def __init__(self, project=None, name=None, job_name=None):
        self.setup = IODictionary()
        super(MMMM, self).__init__(project=project, name=name, job_name=job_name)

        # Defaults
        self.input.seed_ids = None
        self.input.vacuum_width = 2.
        self.input.shell_cutoff = None
        self.input.n_core_shells = 2
        self.input.n_buffer_shells = 2
        self.input.filler_width = 6.
        self.input.default.gamma0 = 0.1
        self.input.default.fix_com = True
        self.input.default.use_adagrad = True
        self.input.default.n_steps = 100
        self.input.default.f_tol = 1e-4

    def define_vertices(self):
        # Components
        self.graph.clock = Counter()
        self.graph.force_norm_mm = Norm()
        self.graph.force_norm_qm = Norm()
        self.graph.max_force_mm = Max()
        self.graph.max_force_qm = Max()
        self.graph.check_force_mm = IsLEq()
        self.graph.check_force_qm = IsLEq()
        self.graph.check_steps = IsGEq()
        self.graph.calc_static_mm = ExternalHamiltonian()
        self.graph.calc_static_qm = ExternalHamiltonian()
        self.graph.gradient_descent_mm = GradientDescent()
        self.graph.gradient_descent_qm = GradientDescent()
        self.graph.update_buffer_qm = AddDisplacements()
        self.graph.update_core_mm = AddDisplacements()
        self.graph.calc_static_small = ExternalHamiltonian()

    def define_execution_flow(self):
        # Order of execution through the graph
        self.graph.make_pipeline(
            self.graph.clock,
            self.graph.force_norm_mm,
            self.graph.max_force_mm,
            self.graph.force_norm_qm,
            self.graph.max_force_qm,
            self.graph.check_steps, 'false',
            self.graph.check_force_mm, 'true',
            self.graph.check_force_qm, 'false',
            self.graph.calc_static_mm,
            self.graph.calc_static_qm,
            self.graph.gradient_descent_mm,
            self.graph.gradient_descent_qm,
            self.graph.update_buffer_qm,
            self.graph.update_core_mm,
            self.graph.clock)
        self.graph.make_pipeline(self.graph.check_force_mm, 'false', self.graph.calc_static_mm)
        self.graph.make_pipeline(self.graph.check_force_qm, 'true', self.graph.calc_static_small)
        self.graph.make_pipeline(self.graph.check_steps, 'true', self.graph.calc_static_small)
        self.graph.starting_vertex = self.graph.clock
        self.graph.restarting_vertex = self.graph.clock

        self.protocol_finished += self._compute_qmmm_energy

    def define_information_flow(self):
        # Wiring the data inputs and outputs
        g = self.graph
        gp = Pointer(self.graph)
        ip = Pointer(self.input)
        sp = Pointer(self.setup)
        # I'm playing around with abbreviating everything. Not sure yet whether I like it.

        self.archive.clock = gp.clock.output.n_counts[-1]

        g.check_steps.input.target = gp.clock.output.n_counts[-1]
        g.check_steps.input.threshold = ip.n_steps

        g.force_norm_mm.input.default.x = [np.inf]  # In the first iteration, no calc static output is available
        g.force_norm_mm.input.x = gp.calc_static_mm.output.forces[-1][Pointer(self)._except_core]
        g.force_norm_mm.input.ord = 2
        g.force_norm_mm.input.axis = -1

        g.max_force_mm.input.default.a = [np.inf]
        g.max_force_mm.input.a = gp.force_norm_mm.output.n[-1]
        g.check_force_mm.input.target = gp.max_force_mm.output.amax[-1]
        g.check_force_mm.input.threshold = ip.f_tol

        g.force_norm_qm.input.default.x = [np.inf]  # In the first iteration, no calc static output is available
        g.force_norm_qm.input.x = gp.calc_static_qm.output.forces[-1][Pointer(self)._only_core]
        g.force_norm_qm.input.ord = 2
        g.force_norm_qm.input.axis = -1

        g.max_force_qm.input.default.a = [np.inf]
        g.max_force_qm.input.a = gp.force_norm_qm.output.n[-1]
        g.check_force_qm.input.target = gp.max_force_qm.output.amax[-1]
        g.check_force_qm.input.threshold = ip.f_tol

        g.calc_static_mm.setup.job_type = 'Lammps'  # This is MM/MM hard-coded!
        g.calc_static_mm.setup.project_path = self.project.path
        g.calc_static_mm.setup.user = self.project.user
        g.calc_static_mm.setup.protocol_name = self.name
        g.calc_static_mm.setup.potential = ip.mm_potential
        g.calc_static_mm.setup.structure = ip.structure
        g.calc_static_mm.input.default.positions = ip.structure.positions
        g.calc_static_mm.input.positions = gp.gradient_descent_mm.output.positions[-1]

        g.calc_static_qm.setup.job_type = 'Lammps'  # This is MM/MM hard-coded!
        g.calc_static_qm.setup.project_path = self.project.path
        g.calc_static_qm.setup.user = self.project.user
        g.calc_static_qm.setup.protocol_name = self.name
        g.calc_static_qm.setup.potential = ip.qm_potential
        g.calc_static_qm.setup.structure = sp.qm_structure
        g.calc_static_qm.input.default.positions = sp.qm_structure.positions
        g.calc_static_qm.input.positions = gp.gradient_descent_qm.output.positions[-1]

        g.gradient_descent_mm.input.default.positions = ip.structure.positions
        g.gradient_descent_mm.input.forces = gp.calc_static_mm.output.forces[-1]
        g.gradient_descent_mm.input.positions = gp.update_core_mm.output.positions[-1]
        g.gradient_descent_mm.input.masses = ip.structure.get_masses
        g.gradient_descent_mm.input.gamma0 = ip.gamma0
        g.gradient_descent_mm.input.fix_com = ip.fix_com
        g.gradient_descent_mm.input.use_adagrad = ip.use_adagrad
        g.gradient_descent_mm.input.mask = Pointer(self)._except_core

        g.gradient_descent_qm.input.default.positions = sp.qm_structure.positions
        g.gradient_descent_qm.input.forces = gp.calc_static_qm.output.forces[-1]
        g.gradient_descent_qm.input.positions = gp.update_buffer_qm.output.positions[-1]
        g.gradient_descent_qm.input.masses = sp.qm_structure.get_masses
        g.gradient_descent_qm.input.gamma0 = ip.gamma0
        g.gradient_descent_qm.input.fix_com = ip.fix_com
        g.gradient_descent_qm.input.use_adagrad = ip.use_adagrad
        g.gradient_descent_qm.input.mask = Pointer(self)._only_core

        g.update_core_mm.input.target = gp.gradient_descent_mm.output.positions[-1]
        g.update_core_mm.input.target_mask = [
            ip.domain_ids['seed'],
            ip.domain_ids['core']
        ]
        g.update_core_mm.input.displacement = gp.gradient_descent_qm.output.displacements[-1]
        g.update_core_mm.input.displacement_mask = [
            ip.domain_ids_qm['seed'],
            ip.domain_ids_qm['core']
        ]

        g.update_buffer_qm.input.target = gp.gradient_descent_qm.output.positions[-1]
        g.update_buffer_qm.input.target_mask = ip.domain_ids_qm['buffer']
        g.update_buffer_qm.input.displacement = gp.gradient_descent_mm.output.displacements[-1]
        g.update_buffer_qm.input.displacement_mask = ip.domain_ids['buffer']

        g.calc_static_small.setup.job_type = 'Lammps'
        g.calc_static_small.setup.project_path = self.project.path
        g.calc_static_small.setup.user = self.project.user
        g.calc_static_small.setup.protocol_name = self.name
        g.calc_static_small.setup.potential = ip.mm_potential
        g.calc_static_small.setup.structure = sp.qm_structure
        g.calc_static_small.input.default.positions = sp.qm_structure.positions
        g.calc_static_small.input.positions = gp.calc_static_qm.input.positions

        self.output.energy_mm = gp.calc_static_mm.output.energy_pot[-1]
        self.output.energy_qm = gp.calc_static_qm.output.energy_pot[-1]
        self.output.energy_mm_one = gp.calc_static_small.output.energy_pot[-1]

    def _compute_qmmm_energy(self):
        self.output.energy_qmmm = self.output.energy_mm + self.output.energy_qm - self.output.energy_mm_one

    def _except_core(self):
        seed_ids = np.array(self.input.domain_ids['seed'])
        core_ids = np.array(self.input.domain_ids['core'])
        return np.setdiff1d(np.arange(len(self.input.structure)), np.concatenate([seed_ids, core_ids]))

    def _only_core(self):
        return np.concatenate([self.input.domain_ids_qm['seed'], self.input.domain_ids_qm['core']])

    def save(self):
        # Before we run, we need to make sure the QM structure is set
        # This would make sense to do in `run`, right before we call super
        # However, before running, pyiron saves jobs to hdf5 in case they're about to be sent to the queue
        # So we'll create the job here
        self.set_qm_structure()
        # A better solution would be to
        super(MMMM, self).save()

    def set_qm_structure(self):
        try:
            structure_already_exists = self.setup.qm_structure
            self.logger.info('set_qm_structure called, but setup.qm_structure already exists.')
        except KeyError:  # Or it doesn't, so create it
            superstructure = self.input.structure

            if 'domain_ids' not in self.input:
                if self.input.seed_ids is None:
                    raise ValueError('Either the domain ids must be provided explicitly, or seed ids must be given.')
                seed_ids = np.array(self.input.seed_ids, dtype=int)
                shells = self._build_shells(superstructure,
                                            self.input.n_core_shells + self.input.n_buffer_shells,
                                            self.input.seed_ids,
                                            self.input.shell_cutoff)
                core_ids = np.concatenate(shells[:self.input.n_core_shells])
                buffer_ids = np.concatenate(shells[self.input.n_core_shells:])
                region_I_ids = np.concatenate((seed_ids, core_ids, buffer_ids))

                bb = self._get_bounding_box(superstructure[region_I_ids])
                extra_box = 0.5 * np.array(self.input.filler_width)
                bb[:, 0] -= extra_box
                bb[:, 1] += extra_box

                # Store it because get bounding box return a tight box and is different
                if self.input.filler_width >= 0:
                    filler_ids = self._get_ids_within_box(superstructure, bb)
                    filler_ids = np.setdiff1d(filler_ids, region_I_ids)
                else:
                    filler_ids = np.array([], dtype=int)

                bb = self._get_bounding_box(superstructure[np.concatenate((region_I_ids, filler_ids))])

                self.input.domain_ids = {'seed': seed_ids, 'core': core_ids, 'buffer': buffer_ids, 'filler': filler_ids}
            elif 'seed_ids' not in self.input:
                raise ValueError('Only *one* of `seed_ids` and `domain_ids` may be provided.')
            # Use domains provided
            else:
                seed_ids = self.input.domain_ids['seed']
                core_ids = self.input.domain_ids['core']
                buffer_ids = self.input.domain_ids['buffer']
                filler_ids = self.input.domain_ids['filler']
                all_specified_ids = np.concatenate((seed_ids, core_ids, buffer_ids, filler_ids))
                if len(all_specified_ids) > len(np.unique(all_specified_ids)):
                    raise ValueError('Some of the provided domain ids occur in more than one domain.')
                region_I_ids = np.concatenate((seed_ids, core_ids, buffer_ids))
                bb = self._get_bounding_box(superstructure[np.concatenate((region_I_ids, filler_ids))])
            # Extract the relevant atoms

            # Build the domain ids in the qm structure
            qm_structure = None
            domain_ids_qm = {}
            offset = 0
            for key, ids in self.input.domain_ids.items():
                if qm_structure is None:
                    qm_structure = superstructure[ids]
                else:
                    qm_structure += superstructure[ids]
                id_length = len(ids)
                domain_ids_qm[key] = np.arange(id_length) + offset
                offset += id_length

            self.input.domain_ids_qm = domain_ids_qm
            # And put everything in a box near (0,0,0)
            extra_vacuum = 0.5 * self.input.vacuum_width
            bb[:, 0] -= extra_vacuum
            bb[:, 1] += extra_vacuum

            # If the bounding box is larger than the MM superstructure
            bs = np.abs(bb[:, 1] - bb[:, 0])
            supercell_lengths = [np.linalg.norm(row) for row in superstructure.cell]
            shrinkage = np.array([(box_size - cell_size) / 2.0 if box_size > cell_size else 0.0
                                  for box_size, cell_size in zip(bs, supercell_lengths)])
            if np.any(shrinkage > 0):
                self.logger.info('The QM box is larger than the MM Box therefore I\'ll shrink it')
                bb[:, 0] += shrinkage
                bb[:, 1] -= shrinkage
            elif any([0.9 < box_size / cell_size < 1.0 for box_size, cell_size in zip(bs, supercell_lengths)]):
                # Check if the box is just slightly smaller than the superstructure cell
                self.logger.warn(
                    'Your cell is nearly as large as your supercell. Probably you want to expand it a little bit')

            box_center = tuple(np.dot(np.linalg.inv(superstructure.cell), np.mean(bb, axis=1)))
            qm_structure.wrap(box_center)
            qm_structure.cell = np.identity(3) * np.ptp(bb, axis=1)
            # Wrap it to the unit cell
            qm_structure.positions = np.dot(qm_structure.get_scaled_positions(), qm_structure.cell)

            self.setup.qm_structure = qm_structure
            self._change_qm_species()

    def reset_qm_structure(self):
        """Sets the QM structure -- if it already exists it destroys it first and re-sets it."""
        self.setup.pop('qm_structure', None)
        self.set_qm_structure()

    @staticmethod
    def _build_shells(structure, n_shells, seed_ids, cutoff):
        indices = [seed_ids]
        current_shell_ids = seed_ids
        for _ in range(n_shells):
            neighbors = structure.get_neighbors(id_list=current_shell_ids, cutoff=cutoff)
            new_ids = np.unique(neighbors.indices)
            # Make it exclusive
            for shell_ids in indices:
                new_ids = np.setdiff1d(new_ids, shell_ids)
            indices.append(new_ids)
            current_shell_ids = new_ids
        # Pop seed ids
        indices.pop(0)
        return indices

    @staticmethod
    def _get_bounding_box(structure):
        """
        Finds the smallest rectangular prism which encloses all atoms in the structure after accounting for periodic
        boundary conditions.

        So what's the problem?

        |      ooooo  |, easy, wrap by CoM
        |ooo        oo|, easy, wrap by CoM
        |o    ooo    o|,


        Args:
            structure (Atoms): The structure to bound.

        Returns:
            numpy.ndarray: A 3x2 array of the x-min through z-max values for the bounding rectangular prism.
        """
        wrapped_structure = structure.copy()
        # Take the frist positions and wrap the atoms around there to determine the size of the bounding box
        wrap_center = tuple(np.dot(np.linalg.inv(structure.cell), structure.positions[0, :]))
        wrapped_structure.wrap(wrap_center)

        bounding_box = np.vstack([
            np.amin(wrapped_structure.positions, axis=0),
            np.amax(wrapped_structure.positions, axis=0)
        ]).T
        return bounding_box

    def show_boxes(self):

        self._plot_boxes([self.input.structure.cell, self.setup.qm_structure.cell], colors=['r', 'b'],
                         titles=['MM Superstructure', 'QM Structure'])

    @staticmethod
    def _get_ids_within_box(structure, box):
        """
        Finds all the atoms in a structure who have a periodic image inside the bounding box.

        Args:
            structure (Atoms): The structure to search.
            box (np.ndarray): A 3x2 array of the x-min through z-max values for the bounding rectangular prism.

        Returns:
            np.ndarray: The integer ids of the atoms inside the box.
        """
        box_center = np.mean(box, axis=1)
        box_center_direct = np.dot(np.linalg.inv(structure.cell), box_center)
        # Wrap atoms so that they are the closest image to the box center
        wrapped_structure = structure.copy()
        wrapped_structure.wrap(tuple(box_center_direct))
        pos = wrapped_structure.positions
        # Keep only atoms inside the box limits
        masks = []
        for d in np.arange(len(box)):
            masks.append(pos[:, d] > box[d, 0])
            masks.append(pos[:, d] < box[d, 1])
        total_mask = np.prod(masks, axis=0).astype(bool)
        return np.arange(len(structure), dtype=int)[total_mask]

    def _change_qm_species(self):
        # The seed sites are the first in the qm structure

        for index, species in zip(self.input.domain_ids_qm['seed'], self.input.seed_species):
            self.setup.qm_structure[index] = species

    def run_static(self):
        self.set_qm_structure()
        super(MMMM, self).run_static()

    def _plot_boxes(self, cells, translate=None, colors=None, titles=None, default_color='b', size=(29, 21)):
        """
        Plots one or a list of cells in xy, yz and xt projection
        Args:
            cells (numpy.ndarray): The cells to plot. A list of (3,3) or a single (3,3) matrix
            translate (list): list of translations vectors for each cell list of (3,) numpy arrays.
            colors (list): list of colors for each cell in the list. e.g ['b', 'g', 'y'] or 'bgky'
            titles (list): list of names displayed for each cell. list of str
            default_color (str): if colors is not specified this color will be applied to all cells
            size (float,float): A tuple of two float specifiyng the size of the plot in centimeters

        Returns:
            matplotlib.figure: The figure which will be displayed
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # Make sure that looping over this properties works smoothly
        cells = ensure_iterable(cells)
        translate = translate or [np.zeros(3, )] * len(cells)
        colors = colors or [default_color] * len(cells)
        titles = ensure_iterable(titles)

        # Scaled coordinates for all edges of the cell
        edges = [
            [(0, 0, 0), (1, 0, 0)],
            [(0, 0, 0), (0, 1, 0)],
            [(0, 0, 0), (0, 0, 1)],
            [(1, 1, 0), (1, 0, 0)],
            [(1, 1, 0), (0, 1, 0)],
            [(1, 1, 0), (1, 1, 1)],
            [(1, 0, 1), (1, 0, 0)],
            [(1, 0, 1), (1, 1, 1)],
            [(1, 0, 1), (0, 0, 1)],
            [(0, 1, 1), (1, 1, 1)],
            [(0, 1, 1), (0, 0, 1)],
            [(0, 1, 1), (0, 1, 0)]
        ]

        # Map axis to indices
        axis_mapping = {a: i for a, i in zip('xyz', range(3))}
        # Planes to plot
        planes = ['xy', 'yz', 'xz']
        # Get subplots and set the size in inches
        fig, axes = plt.subplots(1, len(planes))
        fig.set_size_inches(*[s / 2.54 for s in size])

        for i, plane in enumerate(planes):
            axis = axes[i]
            axis.set_aspect('equal')
            axis.title.set_text('{} plane'.format(plane))
            indices = [axis_mapping[a] for a in plane]
            for j, (cell, color, translation) in enumerate(zip(cells, colors, translate)):
                calc = lambda mul_, cell_: sum([m_ * vec_ for m_, vec_ in zip(mul_, cell_)])
                coords = [(calc(start, cell) + translation, calc(end, cell) + translation) for start, end in edges]
                for start, end in coords:
                    sx, sy = start[indices]
                    ex, ey = end[indices]
                    axis.plot([sx, ex], [sy, ey], color=color)
        # Create dummy legend outside
        legend_lines = [Line2D([0], [0], color=col or default_color, lw=2) for col in colors]
        legend_titles = [title or 'Box {}'.format(i + 1) for i, title in enumerate(titles)]
        plt.figlegend(legend_lines, legend_titles, loc='lower center', fancybox=True, shadow=True)
        return plt

    def plot_qm_structure(self):
        """Plot the QM domain (seed/core + buffer + filler) coloured from blue to red."""
        if self.setup.qm_structure is None:
            print("Please run `set_qm_structure` or run the protocol before trying to plot the QM domain.")
        color_scalar = np.zeros(len(self.setup.qm_structure))
        for n, k in enumerate(self.input.domain_ids_qm.keys()):
            ids = self.input.domain_ids_qm[k]
            color_scalar[ids] = n
        return self.setup.qm_structure.plot3d(scalar_field=color_scalar)

    def plot_force_convergence(self, equil_frames=0):
        """Plot the archived force convergence, starting at a given frame"""
        step = self.graph.clock.archive.output.n_counts[equil_frames:-1]
        qm_force = self.graph.max_force_qm.archive.output.amax[equil_frames + 1:]
        mm_force = self.graph.max_force_mm.archive.output.amax[equil_frames + 1:]
        plt.plot(step, qm_force,
                 marker='s', label='QM core')
        plt.plot(step, mm_force,
                 marker='o', label='Buffer and MM')
        plt.xlabel('Step')
        plt.ylabel('Max atomic force [eV/$\mathrm{\AA}$]')
        plt.legend()

    def to_hdf(self, hdf=None, group_name=None):
        super(MMMM, self).to_hdf(hdf=hdf, group_name=group_name)
        if hdf is None:
            hdf = self.project_hdf5
        self.input.to_hdf(hdf=hdf, group_name="setup")

    def from_hdf(self, hdf=None, group_name=None):
        super(MMMM, self).from_hdf(hdf=hdf, group_name=group_name)
        if hdf is None:
            hdf = self.project_hdf5
        self.input.from_hdf(hdf=hdf, group_name="setup")
