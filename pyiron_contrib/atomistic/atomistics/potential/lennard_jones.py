# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_contrib.atomistic.atomistics.potential.simple_potential import SimplePotential

class LennardJones(SimplePotential):   
    def _potential(self, r):
        epsilon = self._matrix['epsilon']
        sigma = self._matrix['sigma']
        cutoff = self._matrix['cutoff']
        try:
            r_t = r.T
        except AttributeError:
            r_t = r
        return epsilon*((sigma/r_t)**12-(sigma/r_t)**6)*(r_t<=cutoff)

    def get_energy(self, r, per_atom=False):
        E = np.sum(self._potential(r)-self._potential(self._matrix['cutoff']))
        return E

    def get_forces(self, v, r=None, per_atom=False):
        epsilon = self._matrix['epsilon']
        sigma = self._matrix['sigma']
        cutoff = self._matrix['cutoff']
        if r is None:
            r = np.linalg.norm(v, axis=-1)
        r_t = r.T
        if per_atom:
            return -(epsilon*sigma**2*v.T*(12*(sigma/r_t)**10-6*(sigma/r_t)**4)*(r_t<=cutoff)).T
        else:
            return -np.sum(epsilon*sigma**2*v.T*(12*(sigma/r_t)**10-6*(sigma/r_t)**4)*(r_t<=cutoff), axis=-2).T
