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
        return epsilon*((sigma/r)**12-(sigma/r)**6)*(r<=cutoff)

    def _get_energy(self, r):
        E = np.sum(self._potential(r)-self._potential(self._matrix['cutoff']))
        return E

    def _get_forces(self, v, r, per_atom=False):
        epsilon = self._matrix['epsilon']
        sigma = self._matrix['sigma']
        cutoff = self._matrix['cutoff']
        f = epsilon*sigma**2*(12*(sigma/r)**10-6*(sigma/r)**4)*(r<=cutoff)
        if per_atom:
            return -np.einsum('ijk,ij->ijk', v, f)
        else:
            return -np.einsum('ijk,ij->ik', v, f)
