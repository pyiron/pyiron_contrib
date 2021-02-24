import xml.etree.ElementTree as ET
import posixpath

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from pyiron_base import InputList, PyironFactory

from pyiron_contrib.atomistic.atomicrex.utility_functions import write_pretty_xml
from pyiron_contrib.atomistic.atomicrex.potential_factory import ARPotFactory
from pyiron_contrib.atomistic.atomicrex.function_factory import FunctionFactory


class GeneralARInput(InputList):
    """
    Class to store general input of an atomicrex job,
    f.e. the fit algorithm.
    """  
    def __init__(self, table_name="general_input", *args, **kwargs):
        super().__init__(table_name="general_input", *args, **kwargs)
        self.name = "Atomicrex Job"
        self.verbosity = "medium"
        self.real_precision = 16
        self.validate_potentials = False
        self.atom_types = {}      
        self.fit_algorithm = AR_LBFGS(conv_threshold=1e-10, max_iter=50, gradient_epsilon=1e-8)
        self.output_interval = 100
        self.enable_fitting = True

    @property
    def atom_types(self):
        """
        Dictionary used to specify elements in the fit job.
        Entries should use the element as key and None or a (mass, index) tuple as value.
        Examples:
        {"Cu", None}
        {"Cu", (63.546, 29)}
        If value is None the mass and index are taken from the ase package.

        Returns:
            [dict]: Dict of elements.
        """        
        return self._atom_types
    
    @atom_types.setter
    def atom_types(self, atom_types):
        self._atom_types = atom_types
        
    def _write_xml_file(self, directory):
        """Internal function.
        Write the main input xml file in a directory.   

        Args:
            directory (str): Working directory
        """        
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


class AlgorithmFactory(PyironFactory):
    """
    Factory class to conveniently acces the different
    fitting algorithms available in atomicrex.
    """    

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


class AtomicrexAlgorithm(InputList):
    """
    Class to inherit from. If more algorithms will be implemented in atomicrex
    at some point they can be implemented similar to the AR_LBFGS class.
    """    
    def __init__(self, conv_threshold, max_iter, gradient_epsilon, name, *args. **kwargs):
        super().__init__(table_name="fitting_algorithm", *args, **kwargs)
        self.conv_threshold = conv_threshold
        self.max_iter = max_iter
        self.gradient_epsilon = gradient_epsilon
        self.name = name

    def _to_xml_element(self):
        """Internal function.
        Converts the algorithm to an xml element
        """        
        algo = ET.Element(self.name)
        algo.set("conv-threshold", f"{self.conv_threshold}")
        algo.set("max-iter", f"{self.max_iter}")
        if self.gradient_epsilon is not None:
            algo.set("gradient-epsilon", f"{self.gradient_epsilon}")
        return algo


class AR_LBFGS(AtomicrexAlgorithm):
    """
    Use the atomicrex implementation of the LBFGS minimizer.
    """    
    def __init__(self, conv_threshold, max_iter, gradient_epsilon):
        super().__init__(conv_threshold, max_iter, gradient_epsilon, name="BFGS")


class SpaMinimizer:
    """
    Global optimizer implemented in atomicrex.
    Should be used in combination with a local minimizer.
    See the atomicrex documentation for details.
    """    
    def __init__(self, spa_iterations, seed):
        self.spa_iterations = spa_iterations
        self.seed = seed
        self.local_minimizer = None

    def _to_xml_element(self):
        """Internal function.
        Converts the algorithm to a xml element
        and returns it
        """     
        spa = ET.Element("spa")
        spa.set("max-iter", f"{self.spa_iterations}")
        spa.set("seed", f"{self.seed}")
        if self.local_minimizer is not None:
            spa.append(self.local_minimizer._to_xml_element())
        return spa


class NloptAlgorithm(InputList):
    """
    Nlopt algorithms should inherit from this class.
    """    
    def __init__(self, stopval, maxeval, maxtime, ftol_rel, ftol_abs, xtol_rel, name, seed, *args, **kwargs):
        super().__init__(table_name="fitting_algorithm", *args, **kwargs)
        self.stopval = stopval
        self.maxeval = maxeval
        self.maxtime = maxtime
        self.ftol_rel = ftol_rel
        self.ftol_abs = ftol_abs
        self.xtol_rel = xtol_rel
        self.seed = seed
        self.name = name

    def _to_xml_element(self):
        """Internal Function.
        Converts the algorithm to a xml element
        and returns it
        """        
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
