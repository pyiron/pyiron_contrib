# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base.generic.inputlist import InputList
import numpy as np

class SimplePotential(InputList):
    def __init__(self):
        self.coeff = InputList()
        self._matrix = InputList()
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
        for k in self.coeff.keys():
            if type(self.coeff[k])==InputList:
                self._matrix[k] = np.vectorize(self.coeff[k].__getitem__)(pairs).T
            else:
                self._matrix[k] = self.coeff[k]
        self._uptodate = True
