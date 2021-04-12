# coding: utf-8
# http://gitlab.skoltech.ru/shapeev/mlip-dev/blob/master/src/external/python/mlippy/cfgs.py

from __future__ import print_function
import numpy as np


class Cfg:
    pos = None
    lat = None
    types = None
    energy = None
    forces = None
    stresses = None
    desc = None
    grade = None


def readcfg(f):
    cfg = Cfg()
    cfg.lat = np.zeros((3, 3))
    size = -1
    mode = -1
    line = f.readline()
    while line:
        line = line.upper()
        line = line.strip()
        if mode == 0:
            if line.startswith('SIZE'):
                line = f.readline()
                size = int(line.strip())
                cfg.types = np.zeros(size)
                cfg.pos = np.zeros((size, 3))
            elif line.startswith('SUPERCELL'):
                line = f.readline()
                vals = line.strip().split()
                cfg.lat[0, :] = vals[0:3]
                line = f.readline()
                vals = line.strip().split()
                cfg.lat[1, :] = vals[0:3]
                line = f.readline()
                vals = line.strip().split()
                cfg.lat[2, :] = vals[0:3]
            elif line.startswith('ATOMDATA'):
                if line.endswith('FZ'):
                    cfg.forces = np.zeros((size, 3))
                for i in range(size):
                    line = f.readline()
                    vals = line.strip().split()
                    cfg.types[i] = vals[1]
                    cfg.pos[i, :] = vals[2:5]
                    if cfg.forces is not None:
                        cfg.forces[i, :] = vals[5:8]
            elif line.startswith('ENERGY'):
                line = f.readline()
                cfg.energy = float(line.strip())
            elif line.startswith('PLUSSTRESS'):
                line = f.readline()
                vals = line.strip().split()
                cfg.stresses = np.zeros(6)
                cfg.stresses[:] = vals[0:6]
            elif line.startswith('FEATURE   MV_GRADE'):
                cfg.grade = float(line.split()[-1])
            elif line.startswith('FEATURE   PYIRON'):
                cfg.desc = line.split()[-1]
        if line.startswith('BEGIN_CFG'):
            mode = 0
        elif line.startswith('END_CFG'):
            break
        line = f.readline()
    return cfg


def savecfg(f, cfg, desc=None):
    atstr1 = 'AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz'
    atstr2 = 'AtomData:  id type       cartes_x      cartes_y      cartes_z'
    size = len(cfg.types)
    print('BEGIN_CFG', file=f)
    print('Size', file=f)
    print('   %-d' % size, file=f)
    if cfg.lat is not None:
        print('SuperCell', file=f)
        for i in range(3):
            print('         %14f%14f%14f'
                  % (cfg.lat[i, 0], cfg.lat[i, 1], cfg.lat[i, 2]), file=f)
    if cfg.forces is not None:
        print(atstr1, file=f)
    else:
        print(atstr2, file=f)
    for i in range(size):
        if cfg.forces is not None:
            print('         %4d %4d %14f%14f%14f %16.8e %16.8e %16.8e' %
                  (i+1, cfg.types[i], cfg.pos[i, 0], cfg.pos[i, 1], cfg.pos[i, 2],
                   cfg.forces[i, 0], cfg.forces[i, 1], cfg.forces[i, 2]), file=f)
        else:
            print('         %4d %4d %14f%14f%14f' %
                  (i+1, cfg.types[i], cfg.pos[i, 0], cfg.pos[i, 1], cfg.pos[i, 2]),
                  file=f)
    if cfg.energy is not None:
        print('Energy\t%14f' % cfg.energy, file=f)
    if cfg.stresses is not None:
        print('PlusStress:  xx          yy          zz          yz          xz          xy', file=f)
        print('         %14f%14f%14f%14f%14f%14f' %
              (cfg.stresses[0], cfg.stresses[1], cfg.stresses[2],
               cfg.stresses[3], cfg.stresses[4], cfg.stresses[5]), file=f)
    if desc is not None:
        print('Feature   from %s' % desc, file=f)
    if cfg.desc is not None:
        print('Feature %s' % cfg.desc, file=f)
    print('END_CFG', file=f)


class cfgparser:
    def __init__(self, file, max_cfgs=None):
        self.cfgs = []
        self.file = file
        self.max_cfgs = max_cfgs

    def __enter__(self):
        while True:
            if self.max_cfgs is not None and len(self.cfgs) == self.max_cfgs:
                break
            cfg = readcfg(self.file)
            if cfg.types is not None:
                self.cfgs.append(cfg)
            else:
                break
        return self.cfgs

    def __exit__(self, *args):
        self.cfgs = []


def printcfg(cfg):
    savecfg(None, cfg)


def loadcfgs(filename, max_cfgs=None):
    with open(filename, 'r') as file:
        with cfgparser(file, max_cfgs) as cfgs:
            return cfgs


def savecfgs(filename, cfgs, desc=None):
    with open(filename, 'w') as file:
        for cfg in cfgs:
            savecfg(file, cfg, desc)
            print("", file=file)
