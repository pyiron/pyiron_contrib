import xml.etree.ElementTree as ET
import posixpath

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from pyiron_base import DataContainer, PyironFactory

from pyiron_contrib.atomistics.atomicrex.utility_functions import write_pretty_xml
from pyiron_contrib.atomistics.atomicrex.potential_factory import ARPotFactory
from pyiron_contrib.atomistics.atomicrex.function_factory import FunctionFactory
from pyiron_contrib.atomistics.atomicrex.parameter_constraints import (
    ParameterConstraints,
)


class GeneralARInput(DataContainer):
    """
    Class to store general input of an atomicrex job,
    f.e. the fit algorithm.
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        table_name="general_input",
        name="Atomicrex Job",
        verbosity="medium",
        real_precision=16,
        validate_potentials=False,
        atom_types=None,
        fit_algorithm=None,
        output_interval=10000,
        enable_fitting=True,
    ):
        super().__init__(
            table_name="general_input",
        )

        self.name = name
        self.verbosity = verbosity
        self.real_precision = real_precision
        self.validate_potentials = validate_potentials
        self.atom_types = AtomTypes()
        self.fit_algorithm = fit_algorithm
        self.output_interval = output_interval
        self.enable_fitting = enable_fitting
        # version "0.1.0"
        self.output_file = "atomicrex.out"
        self.parameter_constraints = ParameterConstraints()

    def _write_xml_file(self, directory, job=None):
        """Internal function.
        Write the main input xml file in a directory.

        Args:
            directory (str): Working directory
        """
        root = ET.Element("job")

        output_file = ET.SubElement(root, "output-file")
        output_file.text = self.output_file

        name = ET.SubElement(root, "name")
        name.text = self.name

        verbosity = ET.SubElement(root, "verbosity")
        verbosity.text = self.verbosity

        if self.validate_potentials:
            validate_potentials = ET.SubElement(root, "validate-potentials")

        real_precision = ET.SubElement(root, "real-precision")
        real_precision.text = f"{self.real_precision}"

        atom_types = ET.SubElement(root, "atom-types")
        if len(self.atom_types) == 0:
            eles = job.structures._structures.get_elements()
            for ele in eles:
                self.atom_types[ele] = None

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

        if not isinstance(self.fit_algorithm, ScipyAlgorithm):
            fitting = ET.SubElement(root, "fitting")
            if self.enable_fitting:
                fitting.set("enabled", "true")
            else:
                fitting.set("enabled", "false")
            fitting.set("output-interval", f"{self.output_interval}")
            fitting.append(self.fit_algorithm._to_xml_element())

        potentials = ET.SubElement(root, "potentials")
        include = ET.SubElement(potentials, "xi:include")
        include.set("href", "potential.xml")
        include.set("xmlns:xi", "http://www.w3.org/2003/XInclude")

        structures = ET.SubElement(root, "structures")
        include = ET.SubElement(structures, "xi:include")
        include.set("href", "structures.xml")
        include.set("xmlns:xi", "http://www.w3.org/2003/XInclude")

        if len(self.parameter_constraints) > 0:
            root.append(self.parameter_constraints._to_xml_element())

        file_name = posixpath.join(directory, "main.xml")
        write_pretty_xml(root, file_name)


class AtomTypes(DataContainer):
    """
    DataContainer used to specify elements in the fit job.
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
        return AtomicrexAlgorithm(
            name="BFGS",
            conv_threshold=conv_threshold,
            max_iter=max_iter,
            gradient_epsilon=gradient_epsilon,
        )

    @staticmethod
    def ar_spa(max_iter=20, seed=42):
        return SpaMinimizer(max_iter=max_iter, seed=seed)

    @staticmethod
    def ld_lbfgs(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LD_LBFGS",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ld_mma(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LD_MMA",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ld_ccsaq(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LD_CCSAQ",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ld_slsqp(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LD_SLSQP",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ld_var1(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LD_VAR1",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ld_var2(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LD_VAR2",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ln_cobyla(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LN_COBYLA",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ln_bobyqa(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LN_BOBYQA",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ln_newuoa(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LN_NEWUOA",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ln_newuoa_bound(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LN_NEWUOA_BOUND",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ln_praxis(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LN_PRAXIS",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ln_neldermead(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LN_NELDERMEAD",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def ln_sbplx(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="LN_SBPLX",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def gn_crs2_lm(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="GN_CRS2_LM",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def gn_esch(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="GN_ESCH",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def gn_direct(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="GN_DIRECT",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def gn_direct_l(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="GN_DIRECT_L",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def gn_isres(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="GN_ISRES",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def g_mlsl(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptGlobalLocal(
            name="G_MLSL",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def g_mlsl_lds(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptGlobalLocal(
            name="G_MLSL_LDS",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def gd_stogo(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="GD_STOGO",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def gd_stogo_rand(
        stopval=1e-10,
        max_iter=50,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        seed=None,
    ):
        return NloptAlgorithm(
            name="GD_STOGO_RAND",
            stopval=stopval,
            max_iter=max_iter,
            maxtime=maxtime,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            seed=seed,
        )

    @staticmethod
    def scipy_algorithm():
        return ScipyAlgorithm()


class AtomicrexAlgorithm(DataContainer):
    """
    Class to inherit from. If more algorithms will be implemented in atomicrex
    at some point they can be implemented similar to the AR_LBFGS class.
    """

    def __init__(
        self,
        conv_threshold=None,
        max_iter=None,
        gradient_epsilon=None,
        name=None,
        *args,
        **kwargs,
    ):
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


class SpaMinimizer(DataContainer):
    """
    Global optimizer implemented in atomicrex.
    Should be used in combination with a local minimizer.
    See the atomicrex documentation for details.
    """

    def __init__(self, max_iter=None, seed=None, *args, **kwargs):
        super().__init__(table_name="fitting_algorithm", *args, **kwargs)
        self.max_iter = max_iter
        self.seed = seed
        self.local_minimizer = None

    @property
    def name(self):
        return "spa"

    def _to_xml_element(self):
        """Internal function.
        Converts the algorithm to a xml element
        and returns it
        """
        spa = ET.Element("spa")
        spa.set("max-iter", f"{self.max_iter}")
        spa.set("seed", f"{self.seed}")
        if self.local_minimizer is not None:
            spa.append(self.local_minimizer._to_xml_element())
        else:
            raise ValueError("Set a local minimizer for Spa")
        return spa


class NloptAlgorithm(DataContainer):
    """
    Nlopt algorithms should inherit from this class.
    """

    def __init__(
        self,
        name=None,
        seed=None,
        stopval=None,
        max_iter=None,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        *args,
        **kwargs,
    ):
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
        if self.seed is not None:
            nlopt.set("seed", f"{self.seed}")
        return nlopt


class NloptGlobalLocal(NloptAlgorithm):
    """
    Nlopt global optimizers that additionally need a local minimizer similar to the spa algorithm implemented in atomicrex
    """

    def __init__(
        self,
        stopval=None,
        max_iter=None,
        maxtime=None,
        ftol_rel=None,
        ftol_abs=None,
        xtol_rel=None,
        name=None,
        seed=None,
        *args,
        **kwargs,
    ):
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
            **kwargs,
        )
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


class ScipyAlgorithm:
    __version__ = "0.0.1"
    __hdf_version__ = "0.0.1"

    def __init__(self):
        self.global_minimizer = None
        self.local_minimizer_kwargs = {
            "method": "L-BFGS-B",
            "jac": None,
            "hess": None,
            "bounds": None,
            "constraints": (),
            "tol": None,
            "options": None,
        }
        self.global_minimizer_kwargs = {}

    def to_hdf(self, hdf, group_name):
        with hdf.open(group_name) as h:
            self._type_to_hdf(h)
            h["global_minimizer"] = self.global_minimizer
        """
            with h.open("local_minimizer_kwargs") as loc_hdf:
                for k, v in self.local_minimizer_kwargs.items():
                    try:
                        loc_hdf[k] = v
                    except TypeError:
                        loc_hdf[k] = v.__name__
            with h.open("global_minimizer_kwargs") as glob_hdf:
                for k, v in self.global_minimizer_kwargs.items():
                    if isinstance(v, dict):
                        with glob_hdf.open(k) as v_hdf:
                            for k, v in self.v.items():
                                try:
                                    v_hdf[k] = v
                                except TypeError:
                                    v_hdf[k] = v.__name__
                    else:           
                        try:
                            glob_hdf[k] = v
                        except TypeError:
                            glob_hdf[k] = v.__name__
        """

    def from_hdf(self, hdf, group_name):
        with hdf.open(group_name) as h:
            # self._type_from_hdf(h)
            self.global_minimizer = h["global_minimizer"]

    def _type_to_hdf(self, hdf):
        """
        Internal helper function to save type and version in hdf root

        Args:
            hdf (ProjectHDFio): HDF5 group object
        """
        hdf["NAME"] = self.__class__.__name__
        hdf["TYPE"] = str(type(self))
        hdf["VERSION"] = self.__version__
        hdf["HDF_VERSION"] = self.__hdf_version__
