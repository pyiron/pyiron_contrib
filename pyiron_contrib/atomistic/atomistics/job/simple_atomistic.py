# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_atomistics.atomistics.job.interactive import GenericInteractive
import numpy as np

class SimpleAtomistic(GenericInteractive):  # Create a custom job class
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.neigh = None
        self.cutoff = 3
        self.potential = None
        self._neighbor_chemical_symbols = None
        self.uptodate = False

    def interactive_initialize_interface(self):
        pass

    def validate_ready_for_run(self):
        super(SimpleAtomistic, self).validate_ready_for_run()
        if self.potential is None:
            raise ValueError("This job does not contain a valid potential: {}".format(self.job_name))

    def run_if_interactive(self):
        super(SimpleAtomistic, self).run_if_interactive()
        if not self.uptodate:
            self._update_neighbors()
            c_old = self._neighbor_chemical_symbols
            c_new = self.neigh.chemical_symbols
            if c_old is None or c_old.shape!=c_new.shape or not self.potential._uptodate or np.any(c_old!=c_new):
                self.potential.update_coeff(self._get_chemical_pairs())
        self.uptodate = True
        self.interactive_collect()

    def _update_neighbors(self):
        self.neigh = self.structure.get_neighbors(num_neighbors=None, cutoff_radius=self.cutoff)

    def _get_chemical_pairs(self):
        c = self.structure.get_chemical_symbols()
        self._neighbor_chemical_symbols = self.neigh.chemical_symbols
        master = np.tile(c, self._neighbor_chemical_symbols.shape[1]).reshape(-1, len(c)).T
        pairs = np.sort(np.stack((master, self._neighbor_chemical_symbols), axis=-1).reshape(master.shape+(2,)))
        return np.char.add(pairs[:,:,0], pairs[:,:,1])

    def interactive_forces_getter(self):
        return self.potential.get_forces(v=self.neigh.vecs, r=self.neigh.distances)

    def interactive_energy_pot_getter(self):
        return self.potential.get_energy(r=self.neigh.distances)

    def interactive_pressures_getter(self):
        forces = self.potential.get_forces(v=self.neigh.vecs, r=self.neigh.distances, per_atom=True)
        r = self.neigh.vecs
        V = self.structure.get_volume()
        return np.einsum('nki,nkj->ij', r, forces)/V

    def interactive_positions_setter(self, positions):
        self.uptodate = False

    def interactive_cells_setter(self, cell);
        self.uptodate = False

    def interactive_indices_setter(self, indices):
        self.uptodate = False


