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
        self.atom_types = AtomTypes()      
        self.fit_algorithm = AtomicrexAlgorithm(conv_threshold=1e-10, max_iter=50, gradient_epsilon=1e-8, name="BFGS")
        self.output_interval = 100
        self.enable_fitting = True
        
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


class AtomTypes(InputList):
    """
        InputList used to specify elements in the fit job.
        Elements can be added using dictionary or attribute syntax.
        Their value should None or a (mass, index) tuple as value.
        If value is None the mass and index are taken from the ase package.
        Examples:
        job.input.atom_types.Cu = None
        job.input.atom_types["Cu"] = None 
        job.input.atom_types.Cu = (63.546, 29)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(table_name="AtomTypes", *args, **kwargs)


class AlgorithmFactory(PyironFactory):
    """
    Factory class to conveniently acces the different
    fitting algorithms available in atomicrex.
    """    

    @staticmethod
    def ar_lbfgs(conv_threshold=1e-10, max_iter=50, gradient_epsilon=None):
        return AtomicrexAlgorithm(name="BFGS", conv_threshold=conv_threshold, max_iter=max_iter, gradient_epsilon=gradient_epsilon)

    @staticmethod
    def ar_spa(spa_iterations=20, seed=42):
        return SpaMinimizer(spa_iterations=spa_iterations, seed=seed)

    @staticmethod
    def ld_lbfgs(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LD_LBFGS", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def ld_mma(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LD_MMA", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def ld_ccsaq(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LD_CCSAQ", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def ld_slsqp(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LD_SLSQP", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def ld_var1(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LD_VAR1", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def ld_var2(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LD_VAR2", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def ln_cobyla(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LN_COBYLA", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def ln_bobyqa(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LN_BOBYQA", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def ln_newuoa(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LN_NEWUOA", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def ln_newuoa_bound(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LN_NEWUOA_BOUND", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)
    
    @staticmethod
    def ln_praxis(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LN_PRAXIS", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def ln_neldermead(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LN_NELDERMEAD", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def ln_sbplx(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="LN_SBPLX", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def gn_crs2_lm(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="GN_CRS2_LM", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def gn_esch(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="GN_ESCH", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def gn_direct(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="GN_DIRECT", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def gn_direct_l(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="GN_DIRECT_L", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)

    @staticmethod
    def gn_isres(stopval=1e-10, max_iter=50, maxtime=None, ftol_rel=None, ftol_abs=None, xtol_rel=None, seed=None):
        return NloptAlgorithm(name="GN_ISRES", stopval=stopval, max_iter=max_iter, maxtime=maxtime, ftol_rel=ftol_rel, ftol_abs=ftol_abs, xtol_rel=xtol_rel, seed=seed)


class AtomicrexAlgorithm(InputList):
    """
    Class to inherit from. If more algorithms will be implemented in atomicrex
    at some point they can be implemented similar to the AR_LBFGS class.
    """    
    def __init__(self, conv_threshold, max_iter, gradient_epsilon, name, *args, **kwargs):
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
    def __init__(self, stopval, max_iter, maxtime, ftol_rel, ftol_abs, xtol_rel, name, seed, *args, **kwargs):
        super().__init__(table_name="fitting_algorithm", *args, **kwargs)
        self.stopval = stopval
        self.max_iter = max_iter
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
        if self.max_iter is not None:
            nlopt.set("maxeval", f"{self.max_iter}")
        if self.maxtime is not None:
            nlopt.set("maxtime", f"{self.maxtime}")
        if self.ftol_rel is not None:
            nlopt.set("ftol_rel", f"{self.ftol_rel}")
        if self.ftol_abs is not None:
            nlopt.set("ftol_abs", f"{self.ftol_abs}")
        if self.xtol_rel is not None:
            nlopt.set("xtol_rel", f"{self.xtol_rel}")
        return nlopt


class NloptGlobalLocal(NloptAlgorithm):
    """
    Nlopt global optimizers that additionally need a local minimizer similar to the spa algorithm implemented in atomicrex
    """    
    def __init__(self, stopval, max_iter, maxtime, ftol_rel, ftol_abs, xtol_rel, name, seed, *args, **kwargs):
        super().__init__(
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            name=name,
            seed=seed,
            *args,
            **kwargs)
        self.local_minimizer = None

    def _to_xml_element(self):
        """Internal Function.
        Converts the algorithm to a xml element
        and returns it
        """ 
        if self.local_minimizer is None:
            raise ValueError("This global minimzer needs an additional local minimzer")
        nlopt = super()._to_xml_element()
        local = self.local_minimizer._to_xml_element()
        nlopt.append(local)
        return nlopt
        