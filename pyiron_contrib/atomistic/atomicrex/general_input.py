import xml.etree.ElementTree as ET
import posixpath

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from pyiron_base import InputList, PyironFactory

from pyiron_contrib.atomistic.atomicrex.utility_functions import write_pretty_xml
from pyiron_contrib.atomistic.atomicrex.potential_factory import ARPotFactory
from pyiron_contrib.atomistic.atomicrex.function_factory import FunctionFactory


class GeneralARInput(InputList):
    def __init__(self, table_name="general_input"):
        super().__init__(table_name="general_input")
        self.name = "Atomicrex Job"
        self.verbosity = "medium"
        self.real_precision = 16
        self.validate_potentials = False
        self.atom_types = {}
        self.fit_algorithm = AR_LBFGS(conv_threshold=1e-10, max_iter=50, gradient_epsilon=1e-8)
        self.output_interval = 100
        self.enable_fitting = True
        self.factories = Factories()
        
    def _write_xml_file(self, directory):
        job = ET.Element("job")
        
        name = ET.SubElement(job, "name")
        name.text = self.name
            
        verbosity = ET.SubElement(job, "verbosity")
        verbosity.text = self.verbosity
        
        if self.validate_potentials:
            validate_potentials = ET.SubElement(job, "validate_potentials")
        
        real_precision = ET.SubElement(job, "real-precision")
        real_precision.text = f"{self.real_precision}"
        
        atom_types = ET.SubElement(job, "atom-types")
        for k, v in self.atom_types.items():
            species = ET.SubElement(atom_types, "species")
            species.text = k
            if v is not None:
                mass, index = v
            else:
                index = atomic_numbers[k]
                mass = atomic_masses[index]
                
            species.set("mass", f"{mass}")
            species.set("atomic-number", f"{index}")

        fitting = ET.SubElement(job, "fitting")
        fitting.set("enabled", f"{self.enable_fitting}".lower())
        fitting.set("output-interval", f"{self.output_interval}")
        fitting.append(self.fit_algorithm._to_xml_element())
         
        potentials = ET.SubElement(job, "potentials")
        include = ET.SubElement(potentials, "xi:include")
        include.set("href", "potential.xml")
        include.set("xmlns:xi", "http://www.w3.org/2003/XInclude")
        
        structures = ET.SubElement(job, "structures")
        include = ET.SubElement(structures, "xi:include")
        include.set("href", "structures.xml")
        include.set("xmlns:xi", "http://www.w3.org/2003/XInclude")
        
        file_name = posixpath.join(directory, "main.xml")
        write_pretty_xml(job, file_name)


class Factories:
    def __init__(self):
        self.potentials = ARPotFactory()
        self.functions = FunctionFactory()
        self.algorithms = AlgorithmFactory()


class AlgorithmFactory(PyironFactory):
    
    @staticmethod
    def ar_lbfgs(conv_threshold=1e-10, max_iter=50, gradient_epsilon=None):
        return AR_LBFGS(conv_threshold=conv_threshold, max_iter=max_iter, gradient_epsilon=gradient_epsilon)

    @staticmethod
    def ar_spa(spa_iterations=20, seed=42):
        return SpaMinimizer(spa_iterations=spa_iterations, seed=seed)

    @staticmethod
    def ld_lbfgs(stopval=1e-10, maxeval=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return LD_LBFGS(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def ld_mma(stopval=1e-10, maxeval=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return LD_MMA(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def ln_neldermead(stopval=1e-10, maxeval=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return LN_NELDERMEAD(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def ln_sbplx(stopval=1e-10, maxeval=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return LN_SBPLX(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def gn_crs2_lm(stopval=1e-10, maxeval=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return GN_CRS2_LM(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def gn_esch(stopval=1e-10, maxeval=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return GN_ESCH(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def gn_direct(stopval=1e-10, maxeval=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return GN_DIRECT(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def gn_direct_l(stopval=1e-10, maxeval=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return GN_DIRECT_L(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def gn_isres(stopval=1e-10, maxeval=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return GN_ISRES(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

### Class to inherit from if more algorithms will be implemented in atomicrex at some point they can be implemented similar to the AR_LBFGS class
class AtomicrexAlgorithm(InputList):
    def __init__(self, conv_threshold, max_iter, gradient_epsilon, name):
        super().__init__(table_name="fitting_algorithm")
        self.conv_threshold = conv_threshold
        self.max_iter = max_iter
        self.gradient_epsilon = gradient_epsilon
        self.name = name
    
    def _to_xml_element(self):
        algo = ET.Element(self.name)
        algo.set("conv-threshold", f"{self.conv_threshold}")
        algo.set("max-iter", f"{self.max_iter}")
        if self.gradient_epsilon is not None:
            algo.set("gradient-epsilon", f"{self.gradient_epsilon}")
        return algo


class AR_LBFGS(AtomicrexAlgorithm):
    def __init__(self, conv_threshold, max_iter, gradient_epsilon):
        super().__init__(conv_threshold, max_iter, gradient_epsilon, name="BFGS")
    

class SpaMinimizer:
    def __init__(self, spa_iterations, seed):
        self.spa_iterations = spa_iterations
        self.seed = seed
        self.local_minimizer = None

    def _to_xml_element(self):
        spa = ET.Element("spa")
        spa.set("max-iter", f"{self.spa_iterations}")
        spa.set("seed", f"{self.seed}")
        spa.append(self.local_minimizer._to_xml_element())
        return spa


class NloptAlgorithm(InputList):
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, name, seed):
        super().__init__(table_name="fitting_algorithm")
        self.stopval = stopval
        self.maxeval = maxeval
        self.maxtime = maxtime
        self.ftol_rel = ftol_rel
        self.ftol_abs = ftol_abs
        self.xtol_rel = xtol_rel
        self.seed = seed
        self.name = name

    def _to_xml_element(self):
        nlopt = ET.Element("nlopt")
        nlopt.set("algorithm", self.name)
        if self.stopval is not None:
            nlopt.set("stopval", f"{self.stopval}")
        if self.maxeval is not None:
            nlopt.set("maxeval", f"{self.maxeval}")
        if self.maxtime is not None:
            nlopt.set("maxtime", f"{self.maxtime}")
        if self.ftol_rel is not None:
            nlopt.set("ftol_rel", f"{self.ftol_rel}")
        if self.ftol_abs is not None:
            nlopt.set("ftol_abs", f"{self.ftol_abs}")
        if self.xtol_rel is not None:
            nlopt.set("xtol_rel", f"{self.xtol_rel}") 
        return nlopt

class LD_LBFGS(NloptAlgorithm):
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, seed):
        super().__init__(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed, name="LD_LBFGS")

class LD_MMA(NloptAlgorithm):
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, seed):
        super().__init__(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed, name="LD_MMA")

class LN_NELDERMEAD(NloptAlgorithm):
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, seed):
        super().__init__(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed, name="LN_NELDERMEAD")

class LN_SBPLX(NloptAlgorithm):
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, seed):
        super().__init__(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed, name="LN_SBPLX")

class GN_CRS2_LM(NloptAlgorithm):
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, seed):
        super().__init__(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed, name="GN_CRS2_LM")

class GN_ESCH(NloptAlgorithm):
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, seed):
        super().__init__(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed, name="GN_ESCH")

class GN_DIRECT(NloptAlgorithm):
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, seed):
        super().__init__(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed, name="GN_DIRECT")

class GN_DIRECT_L(NloptAlgorithm):
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, seed):
        super().__init__(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed, name="GN_DIRECT_L")

class GN_ISRES(NloptAlgorithm):
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, seed):
        super().__init__(stopval=stopval, maxeval=maxeval, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed, name="GN_ISRES")

