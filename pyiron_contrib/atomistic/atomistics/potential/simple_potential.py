# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base.generic.inputlist import InputList
import numpy as np

class SimplePotential(InputList):
    def __init__(self):
        self.coeff = InputList()
        self._matrix = {}
        self._uptodate = False

    def append_coeff(self, key, value, element_one=None, element_two=None):
        if element_one is not None and element_two is not None:
            element = ''.join(np.sort([element_one, element_two]))
            self.coeff[key+'/'+element] = value
        else:
            self.coeff[key] = value
        self._uptodate = False
        
    def get_coeff(self, key, element_one=None, element_two=None):
        if element_one is not None and element_two is not None:
            element = ''.join(np.sort([element_one, element_two]))
            return self.coeff[key+'/'+element]
        else:
            return self.coeff[key]
    
    def update_coeff(self, pairs):
        for k,v in self.coeff.items():
            if type(v)==InputList:
                self._matrix[k] = np.array([[v.get(pp, 0) for pp in p] for p in pairs])
            else:
                self._matrix[k] = self.coeff[k]
        self._uptodate = True

    def get_energy(self, r):
        return self._get_energy(np.atleast_2d(r))

    def get_forces(self, v, r=None, per_atom=False):
        if r is None:
            r = np.linalg.norm(v, axis=-1)
        v[r>self._matrix['cutoff']] = 0
        return self._get_forces(
            v=np.atleast_3d(v), r=np.atleast_2d(r), per_atom=per_atom
        ).squeeze()
