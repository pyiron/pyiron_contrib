import xml.etree.ElementTree as ET
import copy

import numpy as np
import matplotlib.pyplot as plt

from pyiron_base import PyironFactory, DataContainer


class FunctionFactory(PyironFactory):
    """
    Class to conveniently create different function objects.
    for detailed information about the function visit the
    atomicrex documentation.
    """

    @staticmethod
    def user_function(
        identifier,
        input_variable="r",
        species=["*", "*"],
        is_screening_function=False,
        cutoff=None,
    ):
        return UserFunction(
            identifier,
            input_variable=input_variable,
            species=species,
            is_screening_function=is_screening_function,
            cutoff=cutoff,
        )

    @staticmethod
    def poly(identifier, cutoff, species=["*", "*"]):
        """
        TAKE CARE !!!
        The polynomial function implemented in atomicrex does not handle derivatives at the cutoff right now,
        i.e. when using this as a pair function or similar there will be massive jumps in forces.
        """
        return Poly(identifier, cutoff=cutoff, species=species)

    @staticmethod
    def spline(
        identifier, cutoff, derivative_left=0, derivative_right=0, species=["*", "*"]
    ):
        return Spline(identifier, cutoff, derivative_left, derivative_right, species)

    @staticmethod
    def equidistant_spline(
        identifier,
        n_nodes,
        cutoff,
        initial_value_func,
        min_distance=0.0,
        derivative_left=0.0,
        d_left_enabled=True,
        derivative_right=0.0,
        d_right_enabled=False,
        endpoint_val=0.0,
        species=["*", "*"],
    ):
        """
        Convenience function to create a spline function with equidistant node points.

        Args:
            identifier (str): function identifier. Should be unique within the job
            n_nodes (int): number of node points
            cutoff (float): values after are 0
            initial_value_func (function(x)): function to calculate start values for nodes.
            min_distance (float, optional): x coordinate of first node point. Defaults to 0.
            derivative_left (int, optional): . Defaults to 0.
            d_left_enabled (bool, optional): Whether to fit. Defaults to True.
            derivative_right (int, optional): [description]. Defaults to 0.
            d_right_enabled (bool, optional): Whether to fit. Should be False for most functions, beside Embedding terms. Defaults to False.
            endpoint_val (float, None, bool, optional): Start val for endpoint, enabled=False if float, if False endpoint is not included. If None endpoint is included, enabled and start_val is calculated like other points. Defaults to 0.0, which should be used in most cases.
            species (list of str, optional): Only needs to be changed for multi element fits. Defaults to ["*", "*"].

        Returns:
            [type]: [description]
        """
        s = Spline(identifier, cutoff, derivative_left, derivative_right, species)
        if endpoint_val is False:
            x = np.linspace(
                start=min_distance, stop=cutoff, num=n_nodes, endpoint=False
            )
        else:
            x = np.linspace(start=min_distance, stop=cutoff, num=n_nodes, endpoint=True)
        y = initial_value_func(x)
        s.parameters.create_from_arrays(x, y)
        if endpoint_val is not False and endpoint_val is not None:
            s.parameters[f"node_{cutoff}"].enabled = False
            s.parameters[f"node_{cutoff}"].start_val = endpoint_val

        s.derivative_left.enabled = d_left_enabled
        s.derivative_right.enabled = d_right_enabled
        s.derivative_left.start_val = derivative_left
        s.derivative_right.start_val = derivative_right
        return s

    @staticmethod
    def exp_A_screening(
        identifier, cutoff, species=["*", "*"], is_screening_function=True
    ):
        return ExpA(
            identifier,
            cutoff,
            species=species,
            is_screening_function=is_screening_function,
        )

    @staticmethod
    def exp_B_screening(
        identifier,
        cutoff,
        rc,
        alpha,
        exponent,
        species=["*", "*"],
        is_screening_function=True,
    ):
        return ExpB(
            identifier,
            cutoff,
            rc,
            alpha,
            exponent,
            species=species,
            is_screening_function=is_screening_function,
        )

    @staticmethod
    def exp_gaussian_screening(
        identifier,
        cutoff,
        stddev,
        alpha,
        exponent,
        species=["*", "*"],
        is_screening_function=True,
    ):
        return ExpGaussian(
            identifier,
            cutoff,
            stddev,
            alpha,
            exponent,
            species=species,
            is_screening_function=is_screening_function,
        )

    @staticmethod
    def morse_A(identifier, D0, r0, alpha, species=["*", "*"]):
        return MorseA(identifier, D0, r0, alpha, species=species)

    @staticmethod
    def morse_B(identifier, D0, r0, beta, S, delta, species=["*", "*"]):
        return MorseB(identifier, D0, r0, beta, S, delta, species=species)

    @staticmethod
    def morse_C(identifier, A, B, mu, lambda_val, delta, species=["*", "*"]):
        return MorseC(identifier, A, B, mu, lambda_val, delta, species=species)

    @staticmethod
    def gaussian(identifier, prefactor, eta, mu, species=["*", "*"], cutoff=None):
        return GaussianFunc(identifier, prefactor, eta, mu, species, cutoff)

    @staticmethod
    def x_pow_n_cutoff(
        identifier, cutoff, h=1, N=4, species=["*", "*"], is_screening_function=True
    ):
        return XpowNCutoff(
            identifier=identifier,
            cutoff=cutoff,
            h=h,
            N=N,
            species=species,
            is_screening_function=is_screening_function,
        )

    @staticmethod
    def constant(identifier, constant, species=["*", "*"]):
        return Constant(constant=constant, identifier=identifier, species=species)

    @staticmethod
    def MishinCuV(
        identifier,
        E1,
        E2,
        alpha1,
        alpha2,
        r01,
        r02,
        delta,
        cutoff,
        h,
        S1,
        rs1,
        S2,
        rs2,
        S3,
        rs3,
        species=["*", "*"],
    ):
        product_func = FunctionFactory.product(identifier, species)
        sum_func = FunctionFactory.sum(identifier="MorseSum", species=species)
        morse1 = FunctionFactory.morse_A(
            identifier="Morse1", D0=E1, r0=r01, alpha=alpha1, species=species
        )
        morse2 = FunctionFactory.morse_A(
            identifier="Morse2", D0=E2, r0=r02, alpha=alpha2, species=species
        )
        c = FunctionFactory.constant(
            identifier="delta", constant=delta, species=species
        )
        rep1 = FunctionFactory.RsMinusRPowN(
            identifier="rep1", S=S1, rs=rs1, N=4, species=species
        )
        rep2 = FunctionFactory.RsMinusRPowN(
            identifier="rep2", S=S2, rs=rs2, N=4, species=species
        )
        rep3 = FunctionFactory.RsMinusRPowN(
            identifier="rep3", S=S3, rs=rs3, N=4, species=species
        )
        rep1.parameters.N.enabled = False
        rep2.parameters.N.enabled = False
        rep3.parameters.N.enabled = False
        sum_func.functions[morse1.identifier] = morse1
        sum_func.functions[morse2.identifier] = morse2
        sum_func.functions[c.identifier] = c
        sum_func.functions[rep1.identifier] = rep1
        sum_func.functions[rep2.identifier] = rep2
        sum_func.functions[rep3.identifier] = rep3
        screening = FunctionFactory.x_pow_n_cutoff(
            identifier="screening", cutoff=cutoff, h=h, N=4, species=species
        )
        screening.is_screening_function = False
        screening.screening = None
        product_func.functions[sum_func.identifier] = sum_func
        product_func.functions[screening.identifier] = screening
        return product_func

    @staticmethod
    def MishinCuRho(identifier, a, r1, r2, beta1, beta2, species=["*", "*"]):
        return MishinCuRho(identifier, a, r1, r2, beta1, beta2, species)

    @staticmethod
    def MishinCuF(identifier, F0, F2, q1, q2, q3, q4, Q1, Q2, species=["*"]):
        return MishinCuF(identifier, F0, F2, q1, q2, q3, q4, Q1, Q2, species)

    @staticmethod
    def extendedMishinCuF(
        identifier, F0, F2, f3, f4, f5, f6, a3, a4, a5, a6, d3, d4, d5, species=["*"]
    ):
        return ExtendedMishinCuF(
            identifier=identifier,
            F0=F0,
            F2=F2,
            f3=f3,
            f4=f4,
            f5=f5,
            f6=f6,
            a3=a3,
            a4=a4,
            a5=a5,
            a6=a6,
            d3=d3,
            d4=d4,
            d5=d5,
            species=species,
        )

    @staticmethod
    def RsMinusRPowN(identifier, S, rs, N, species=["*", "*"], cutoff=None):
        return RsMinusRPowN(identifier, S, rs, N, species, cutoff=cutoff)

    @staticmethod
    def sum(identifier, species=["*", "*"]):
        return Sum(identifier=identifier, species=species)

    @staticmethod
    def product(identifier, species=["*", "*"]):
        return Product(identifier=identifier, species=species)

    @staticmethod
    def gaussians_sum(
        n_gaussians,
        eta,
        identifier,
        node_points=None,
        cutoff=None,
        initial_prefactors=None,
        min_prefactors=None,
        max_prefactors=None,
        species=["*", "*"],
    ):
        sum_func = FunctionFactory.sum(identifier=identifier, species=species)
        if node_points is None:
            if cutoff is None:
                raise ValueError(
                    "Specify node points or a cutoff to set them automatically"
                )
            else:
                node_points = np.linspace(0, cutoff, n_gaussians, endpoint=False)
        else:
            if len(node_points) != n_gaussians:
                raise ValueError("Number of node points has to match n_gaussians")

        if initial_prefactors is None:
            initial_prefactors = np.ones(n_gaussians)
        if min_prefactors is not None:
            if len(min_prefactors) != n_gaussians:
                raise ValueError("min_prefactors must have length num_gaussians")
        if max_prefactors is not None:
            if len(max_prefactors) != n_gaussians:
                raise ValueError("max_prefactors must have length num_gaussians")

        for i in range(n_gaussians):
            gauss = FunctionFactory.gaussian(
                identifier=f"gauss_{i}",
                prefactor=initial_prefactors[i],
                eta=eta,
                mu=node_points[i],
                species=species,
                cutoff=cutoff,
            )
            gauss.parameters.mu.enabled = False
            gauss.parameters.eta.enabled = False
            if min_prefactors is not None:
                gauss.parameters.prefactor.min_val = min_prefactors[i]
            if max_prefactors is not None:
                gauss.parameters.prefactor.max_val = max_prefactors[i]

            sum_func.functions[gauss.identifier] = gauss
        return sum_func


class BaseFunctionMixin:
    # Mixin class to implement functionality common in all types of functions
    # Be careful with Spline class because it has params, but also derivatives, requiring some special additions to implementations
    def copy_final_to_initial_params(self, filter_func=None):
        for param in self.parameters.values():
            param.copy_final_to_start_value(filter_func=filter_func)

    def lock_parameters(self, filter_func=None):
        for param in self.parameters.values():
            param.lock_value(filter_func=filter_func)

    def randomize_parameters(self, rng, filter_func=None):
        for param in self.parameters.values():
            param.randomize(rng=rng, filter_func=filter_func)

    def set_max_values(self, constant=None, factor=None, filter_func=None):
        """
        Convenience function so set max values for all parameters at once.
        Can either use a constant value or a factor. If both are given factor is used.

        Args:
            constant ([type], optional): param.max_val = constant. Defaults to None.
            factor ([type], optional): param.max_val = abs(start_val)*factor. Defaults to None.
            filter_func ([type], optional): Optional function to filter params. Should take param as argument and return True or False. Defaults to None.

        Raises:
            ValueError: Raises when constant and factor are None.
        """
        if constant is None and factor is None:
            raise ValueError("constant or factor must be set")

        for param in self.parameters.values():
            param.set_max_val(
                constant=constant,
                factor=factor,
                filter_func=filter_func,
            )

    def set_min_values(self, constant=None, factor=None, filter_func=None):
        """
        Convenience function so set min values for all parameters at once.
        Can either use a constant value or a factor. If both are given factor is used.

        Args:
            constant ([type], optional): param.min_val = constant. Defaults to None.
            factor ([type], optional): param.min_val = -abs(start_val)*factor. Defaults to None.
            filter_func ([type], optional): Optional function to filter params. Should take param as argument and return True or False. Defaults to None.

        Raises:
            ValueError: Raises when constant and factor are None.
        """
        if constant is None and factor is None:
            raise ValueError("constant or factor must be set")

        for param in self.parameters.values():
            param.set_min_val(
                constant=constant,
                factor=factor,
                filter_func=filter_func,
            )

    def count_parameters(self, enabled_only=True):
        parameters = 0
        if enabled_only:
            for param in self.parameters.values():
                if param.enabled:
                    parameters += 1
        else:
            for param in self.parameters.values():
                parameters += 1
        return parameters


class MetaFunctionMixin:
    def copy_final_to_initial_params(self, filter_func=None):
        for f in self.functions.values():
            f.copy_final_to_initial_params(filter_func=filter_func)

    def lock_parameters(self, filter_func=None):
        for f in self.functions.values():
            f.lock_parameters(filter_func=filter_func)

    def randomize_parameters(self, rng, filter_func=None):
        for f in self.functions.values():
            f.randomize_parameters(rng=rng, filter_func=filter_func)

    def set_max_values(self, constant=None, factor=None, filter_func=None):
        """
        Convenience function so set max values for all parameters at once.
        Can either use a constant value or a factor. If both are given factor is used.

        Args:
            constant ([type], optional): param.max_val = constant. Defaults to None.
            factor ([type], optional): param.max_val = abs(start_val)*factor. Defaults to None.
            filter_func ([type], optional): Optional function to filter params. Should take param as argument and return True or False. Defaults to None.

        Raises:
            ValueError: Raises when constant and factor are None.
        """
        for f in self.functions.values():
            f.set_max_values(constant=constant, factor=factor, filter_func=filter_func)

    def set_min_values(self, constant=None, factor=None, filter_func=None):
        """
        Convenience function so set min values for all parameters at once.
        Can either use a constant value or a factor. If both are given factor is used.

        Args:
            constant ([type], optional): param.min_val = constant. Defaults to None.
            factor ([type], optional): param.min_val = -abs(start_val)*factor. Defaults to None.
            filter_func ([type], optional): Optional function to filter params. Should take param as argument and return True or False. Defaults to None.

        Raises:
            ValueError: Raises when constant and factor are None.
        """
        for f in self.functions.values():
            f.set_min_values(constant=constant, factor=factor, filter_func=filter_func)

    def count_parameters(self, enabled_only=True):
        parameters = 0
        for f in self.functions.values():
            parameters += f.count_parameters(enabled_only=enabled_only)
        return parameters


class AbstractMetaFunction(DataContainer, MetaFunctionMixin):
    def __init__(self, identifier=None, species=None, table_name=None):
        super().__init__()
        self.identifier = identifier
        self.functions = DataContainer(table_name=table_name)
        self.species = species

    def _to_xml_element(self, func_name):
        root = ET.Element(func_name)
        root.set("id", self.identifier)
        for k, v in self.functions.items():
            root.append(v._to_xml_element())
        return root

    def _parse_final_parameter(self, leftover, value):
        identifier = leftover[0].split("[")[0]
        leftover = leftover[1:]
        try:
            self.functions[identifier]._parse_final_parameter(leftover, value)
        except KeyError:
            raise KeyError(f"Function {identifier} not found in {self.identifier}")


class Sum(AbstractMetaFunction, MetaFunctionMixin):
    def __init__(self, identifier=None, species=None):
        super().__init__(
            identifier=identifier, species=species, table_name="sum_functions"
        )

    def _to_xml_element(self):
        return super()._to_xml_element(func_name="sum")


class Product(AbstractMetaFunction, MetaFunctionMixin):
    def __init__(self, identifier=None, species=None):
        super().__init__(
            identifier=identifier, species=species, table_name="product_functions"
        )

    def _to_xml_element(self):
        return super()._to_xml_element(func_name="product")


class SpecialFunction(DataContainer, BaseFunctionMixin):
    """
    Analytic functions defined within atomicrex should inherit from this class
    https://atomicrex.org/potentials/functions.html#index-1
    https://atomicrex.org/potentials/functions.html#specialized-functions
    """

    def __init__(
        self, identifier=None, species=["*", "*"], is_screening_function=False
    ):
        super().__init__(table_name=f"special_function_{identifier}")
        self.species = species
        self.parameters = FunctionParameterList()
        self.is_screening_function = is_screening_function
        self.identifier = identifier
        if not is_screening_function:
            self.screening = None

    def _to_xml_element(self, name):
        if self.is_screening_function:
            screening = ET.Element("screening")
            root = ET.SubElement(screening, f"{name}")
        else:
            root = ET.Element(f"{name}")

        root.set("id", f"{self.identifier}")
        for param in self.parameters.values():
            p = ET.SubElement(root, f"{param.param}")
            p.text = f"{param.start_val}"

        # This if condition is to prevent an error with the expA screening function
        if name != "exp-A":
            root.append(self.parameters.fit_dofs_to_xml_element())

        if not self.is_screening_function:
            if self.screening is not None:
                root.append(self.screening._to_xml_element())
            return root
        else:
            return screening

    @property
    def func(self):
        return None

    def plot(self):
        if self.func is None:
            raise NotImplementedError(
                "A func property needs to be defined in the subclass"
            )
        else:
            return plot(self.func)

    def _parse_final_parameter(self, leftover, value):
        param = leftover[0].rstrip(":")
        self.parameters[param].final_value = value


class Poly(DataContainer, BaseFunctionMixin):
    """
    Polynomial interpolation function.
    """

    def __init__(self, identifier=None, cutoff=None, species=["*", "*"]):
        super().__init__(table_name=f"Poly_{identifier}")
        self.identifier = identifier
        self.cutoff = cutoff
        self.species = species
        self.parameters = PolyCoeffList()
        # preparation if poly gets screening function ability
        # self.screening = None

    def _to_xml_element(self):
        poly = ET.Element("poly")
        poly.set("id", self.identifier)
        cutoff = ET.SubElement(poly, "cutoff")
        cutoff.text = f"{self.cutoff}"
        poly.append(self.parameters._to_xml_element())
        # if self.screening is not None:
        #        poly.append(self.screening._to_xml_element())
        return poly


class Spline(DataContainer, BaseFunctionMixin):
    """
    Spline interpolation function
    """

    def __init__(
        self,
        identifier=None,
        cutoff=None,
        derivative_left=0,
        derivative_right=0,
        species=["*", "*"],
    ):
        super().__init__(table_name=f"Spline_{identifier}")
        self.identifier = identifier
        self.cutoff = cutoff
        self.derivative_left = FunctionParameter(
            param="derivative-left", start_val=derivative_left
        )
        self.derivative_right = FunctionParameter(
            param="derivative-right", start_val=derivative_right, enabled=False
        )
        self.species = species
        self.parameters = NodeList()

    def _to_xml_element(self):
        spline = ET.Element("spline")
        spline.set("id", self.identifier)
        if self.cutoff is not None:
            cutoff = ET.SubElement(spline, "cutoff")
            cutoff.text = f"{self.cutoff}"
        der_l = ET.SubElement(spline, "derivative-left")
        der_l.text = f"{self.derivative_left.start_val}"
        der_r = ET.SubElement(spline, "derivative-right")
        der_r.text = f"{self.derivative_right.start_val}"

        fit_dof = ET.SubElement(spline, "fit-dof")
        fit_dof.append(self.derivative_left._to_xml_element())
        fit_dof.append(self.derivative_right._to_xml_element())

        spline.append(self.parameters._to_xml_element())
        return spline

    def _parse_final_parameter(self, leftover, value):
        if "derivative-right" in leftover[-1]:
            self.derivative_right.final_value = value
        elif "derivative-left" in leftover[-1]:
            self.derivative_left.final_value = value
        else:
            param = float(leftover[0].split("[")[1])
            param = f"node_{param:.6g}"
            self.parameters[param].final_value = value

    def copy_final_to_initial_params(self, filter_func=None):
        super().copy_final_to_initial_params(filter_func=filter_func)
        self.derivative_left.copy_final_to_start_value(filter_func=filter_func)
        self.derivative_right.copy_final_to_start_value(filter_func=filter_func)

    def lock_parameters(self, filter_func=None):
        super().lock_parameters(filter_func=filter_func)
        self.derivative_left.lock_value(filter_func=filter_func)
        self.derivative_right.lock_value(filter_func=filter_func)

    def set_max_values(self, constant=None, factor=None, filter_func=None):
        super().set_max_values(constant, factor, filter_func)
        self.derivative_left.set_max_val(
            constant=constant, factor=factor, filter_func=filter_func
        )
        self.derivative_right.set_max_val(
            constant=constant, factor=factor, filter_func=filter_func
        )

    def set_min_values(self, constant=None, factor=None, filter_func=None):
        super().set_min_values(constant, factor, filter_func)
        self.derivative_left.set_min_val(
            constant=constant, factor=factor, filter_func=filter_func
        )
        self.derivative_right.set_min_val(
            constant=constant, factor=factor, filter_func=filter_func
        )

    def count_parameters(self, enabled_only=True):
        parameters = super().count_parameters(enabled_only=enabled_only)
        if enabled_only:
            if self.derivative_left.enabled:
                parameters += 1
            if self.derivative_right.enabled:
                parameters += 1
        else:
            parameters += 2
        return parameters


class ExpA(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        cutoff=None,
        species=["*", "*"],
        is_screening_function=True,
    ):
        super().__init__(
            identifier, species=species, is_screening_function=is_screening_function
        )
        self.parameters.add_parameter(
            "cutoff",
            start_val=cutoff,
            enabled=False,
            fitable=False,
        )

    @property
    def func(self):
        return lambda r: np.exp(1 / (r - self.parameters.cutoff.start_val))

    def _to_xml_element(self):
        return super()._to_xml_element(name="exp-A")


class ExpB(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        cutoff=None,
        rc=None,
        alpha=None,
        exponent=None,
        species=None,
        is_screening_function=True,
    ):
        super().__init__(
            identifier, species=species, is_screening_function=is_screening_function
        )
        self.parameters.add_parameter(
            "cutoff",
            start_val=cutoff,
            enabled=False,
            fitable=False,
        )
        self.parameters.add_parameter(
            "rc",
            start_val=rc,
            enabled=False,
        )
        self.parameters.add_parameter(
            "alpha",
            start_val=alpha,
            enabled=False,
        )
        self.parameters.add_parameter(
            "exponent",
            start_val=exponent,
            enabled=False,
        )

    @property
    def func(self):
        return lambda r: np.exp(
            -np.sign(self.parameters.exponent.start_val)
            * self.parameters.alpha.start_val
            / (
                1
                - (
                    (r - self.parameters.rc.start_val)
                    / self.parameters.cutoff.start_val
                    - self.parameters.rc.start_val
                )
                ** self.parameters.exponent.start_val
            )
        )

    def _to_xml_element(self):
        return super()._to_xml_element(name="exp-B")


class ExpGaussian(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        cutoff=None,
        stddev=None,
        alpha=None,
        exponent=None,
        species=["*", "*"],
        is_screening_function=True,
    ):
        super().__init__(
            identifier, species=species, is_screening_function=is_screening_function
        )
        self.parameters.add_parameter(
            "cutoff",
            start_val=cutoff,
            enabled=False,
            fitable=False,
        )
        self.parameters.add_parameter(
            "stddev",
            start_val=stddev,
            enabled=False,
        )
        self.parameters.add_parameter(
            "alpha",
            start_val=alpha,
            enabled=False,
        )
        self.parameters.add_parameter(
            "exponent",
            start_val=exponent,
            enabled=False,
        )

    @property
    def func(self):
        cutoff = self.parameters["cutoff"].start_val
        stddev = self.parameters["stddev"].start_val
        alpha = self.parameters["alpha"].start_val
        exponent = self.parameters["exponent"].start_val
        return (
            lambda r: np.exp(
                -np.sign(exponent) * alpha / (1 - (r / cutoff) ** exponent)
            )
            * np.exp(-(r**2) / (2 * stddev**2))
            / (stddev * np.sqrt(2 * np.pi))
        )

    def _to_xml_element(self):
        return super()._to_xml_element(name="exp-gaussian")


class XpowNCutoff(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        cutoff=None,
        h=1,
        N=4,
        species=["*", "*"],
        is_screening_function=True,
    ):
        super().__init__(
            identifier, species=species, is_screening_function=is_screening_function
        )
        self.parameters.add_parameter(
            "cutoff",
            start_val=cutoff,
            enabled=False,
            fitable=False,
        )
        self.parameters.add_parameter(
            "h",
            start_val=h,
            enabled=False,
        )
        self.parameters.add_parameter(
            "N",
            start_val=N,
            enabled=False,
        )

    @property
    def func(self):
        rc = self.parameters.cutoff.start_val
        h = self.parameters.h.start_val
        N = self.parameters.N.start_val
        return lambda r: ((r - rc) / h) ** N / (1 + ((r - rc) / h) ** N)

    def _to_xml_element(self):
        return super()._to_xml_element(name="XpowN-cutoff")


class MorseA(SpecialFunction):
    def __init__(
        self, identifier=None, D0=None, r0=None, alpha=None, species=["*", "*"]
    ):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "D0",
            start_val=D0,
            enabled=True,
        )
        self.parameters.add_parameter(
            "r0",
            start_val=r0,
            enabled=True,
        )
        self.parameters.add_parameter(
            "alpha",
            start_val=alpha,
            enabled=True,
        )

    @property
    def func(self):
        return lambda r: self.parameters.D0.start_val * (
            np.exp(
                -2
                * self.parameters.alpha.start_val
                * (r - self.parameters.r0.start_val)
            )
            - 2
            * np.exp(
                -self.parameters.alpha.start_val * (r - self.parameters.r0.start_val)
            )
        )

    def _to_xml_element(self):
        return super()._to_xml_element(name="morse-A")


class MorseB(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        D0=None,
        r0=None,
        beta=None,
        S=None,
        delta=None,
        species=["*", "*"],
    ):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "D0",
            start_val=D0,
            enabled=True,
        )
        self.parameters.add_parameter(
            "r0",
            start_val=r0,
            enabled=True,
        )
        self.parameters.add_parameter(
            "beta",
            start_val=beta,
            enabled=True,
        )
        self.parameters.add_parameter(
            "S",
            start_val=S,
            enabled=True,
        )
        self.parameters.add_parameter(
            "delta",
            start_val=delta,
            enabled=True,
        )

    @property
    def func(self):
        D0 = self.parameters.D0.start_val
        r0 = self.parameters.r0.start_val
        S = self.parameters.S.start_val
        beta = self.parameters.beta.start_val
        delta = self.parameters.delta.start_val
        return lambda r: (
            D0 / (S - 1) * np.exp(-beta * np.sqrt(2 * S) * (r - r0))
            - D0 * S / (S - 1) * np.exp(-beta * np.sqrt(2 / S) * (r - r0))
            + delta
        )

    def _to_xml_element(self):
        return super()._to_xml_element(name="morse-B")


class MorseC(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        A=None,
        B=None,
        mu=None,
        lambda_val=None,
        delta=None,
        species=["*", "*"],
    ):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "A",
            start_val=A,
            enabled=True,
        )

        self.parameters.add_parameter(
            "B",
            start_val=B,
            enabled=True,
        )

        self.parameters.add_parameter(
            "mu",
            start_val=mu,
            enabled=True,
        )

        self.parameters.add_parameter(
            "lambda",
            start_val=lambda_val,
            enabled=True,
        )

        self.parameters.add_parameter(
            "delta",
            start_val=delta,
            enabled=True,
        )

    @property
    def func(self):
        A = self.parameters["A"].start_val
        B = self.parameters["B"].start_val
        mu = self.parameters["mu"].start_val
        param_lambda = self.parameters["lambda"].start_val
        delta = self.parameters["delta"].start_val
        return lambda r: A * np.exp(-param_lambda * r) - B * np.exp(-mu * r) + delta

    def _to_xml_element(self):
        return super()._to_xml_element(name="morse-C")


class RsMinusRPowN(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        S=None,
        rs=None,
        N=None,
        species=None,
        is_screening_function=False,
        cutoff=None,
    ):
        super().__init__(
            identifier, species=species, is_screening_function=is_screening_function
        )
        self.cutoff = cutoff
        self.parameters.add_parameter(
            "S",
            start_val=S,
            enabled=True,
        )
        self.parameters.add_parameter(
            "rs",
            start_val=rs,
            enabled=True,
        )
        self.parameters.add_parameter(
            "N",
            start_val=N,
            enabled=False,
        )

    @property
    def func(self):
        def func(r):
            if r < self.parameters.rs.start_val:
                return (
                    self.parameters.S.start_val
                    * (self.parameters.rs.start_val - r) ** self.parameters.N.start_val
                )
            return 0

        return func

    def _to_xml_element(self):
        xml = super()._to_xml_element(name="RsMinusRPowN")
        if self.cutoff is not None:
            cutoff = ET.SubElement(xml, "cutoff")
            cutoff.text = f"{self.cutoff}"
        return xml


class Constant(SpecialFunction):
    def __init__(self, constant=None, identifier=None, species=["*", "*"]):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "const",
            start_val=constant,
            enabled=True,
        )

    def _to_xml_element(self):
        return super()._to_xml_element(name="constant")


## Renamed GaussianFunc to not mess with the gaussian code when loading from hdf5
class GaussianFunc(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        prefactor=None,
        eta=None,
        mu=None,
        species=None,
        cutoff=None,
    ):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "prefactor",
            start_val=prefactor,
            enabled=True,
        )

        self.parameters.add_parameter(
            "eta",
            start_val=eta,
            enabled=True,
        )

        self.parameters.add_parameter(
            "mu",
            start_val=mu,
            enabled=True,
        )
        self.cutoff = cutoff

    @property
    def func(self):
        prefactor = self.parameters["prefactor"].start_val
        eta = self.parameters["eta"].start_val
        mu = self.parameters["mu"].start_val
        return lambda r: prefactor * np.exp(-eta * (r - mu) ** 2)

    def _to_xml_element(self):
        xml = super()._to_xml_element(name="gaussian")
        # Put this in SpecialFunction or AbstractFunction class when rewriting
        # f.e. using getattr()
        if self.cutoff is not None:
            cutoff = ET.SubElement(xml, "cutoff")
            cutoff.text = f"{self.cutoff}"
        return xml


class MishinCuRho(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        a=None,
        r1=None,
        r2=None,
        beta1=None,
        beta2=None,
        species=["*", "*"],
    ):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "a",
            start_val=a,
            enabled=True,
        )
        self.parameters.add_parameter(
            "r1",
            start_val=r1,
            enabled=True,
        )
        self.parameters.add_parameter(
            "r2",
            start_val=r2,
            enabled=True,
        )
        self.parameters.add_parameter(
            "beta1",
            start_val=beta1,
            enabled=True,
        )
        self.parameters.add_parameter(
            "beta2",
            start_val=beta2,
            enabled=True,
        )

    def _to_xml_element(self):
        return super()._to_xml_element(name="Mishin-Cu-rho")


class MishinCuF(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        F0=None,
        F2=None,
        q1=None,
        q2=None,
        q3=None,
        q4=None,
        Q1=None,
        Q2=None,
        species=["*"],
    ):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "F0",
            start_val=F0,
            enabled=True,
        )
        self.parameters.add_parameter(
            "F2",
            start_val=F2,
            enabled=True,
        )
        self.parameters.add_parameter(
            "q1",
            start_val=q1,
            enabled=True,
        )
        self.parameters.add_parameter(
            "q2",
            start_val=q2,
            enabled=True,
        )
        self.parameters.add_parameter(
            "q3",
            start_val=q3,
            enabled=True,
        )
        self.parameters.add_parameter(
            "q4",
            start_val=q4,
            enabled=True,
        )
        self.parameters.add_parameter(
            "Q1",
            start_val=Q1,
            enabled=True,
        )
        self.parameters.add_parameter(
            "Q2",
            start_val=Q2,
            enabled=True,
        )

    def _to_xml_element(self):
        return super()._to_xml_element(name="Mishin-Cu-F")


class ExtendedMishinCuF(SpecialFunction):
    def __init__(
        self,
        identifier=None,
        F0=None,
        F2=None,
        f3=None,
        f4=None,
        f5=None,
        f6=None,
        a3=None,
        a4=None,
        a5=None,
        a6=None,
        d3=None,
        d4=None,
        d5=None,
        species=["*"],
    ):
        super().__init__(identifier, species=species, is_screening_function=False)
        self.parameters.add_parameter(
            "F0",
            start_val=F0,
            enabled=True,
        )
        self.parameters.add_parameter(
            "F2",
            start_val=F2,
            enabled=True,
        )
        self.parameters.add_parameter(
            "f3",
            start_val=f3,
            enabled=True,
        )
        self.parameters.add_parameter(
            "f4",
            start_val=f4,
            enabled=True,
        )
        self.parameters.add_parameter(
            "f5",
            start_val=f5,
            enabled=True,
        )
        self.parameters.add_parameter(
            "f6",
            start_val=f6,
            enabled=True,
        )
        self.parameters.add_parameter(
            "a3",
            start_val=a3,
            enabled=True,
        )
        self.parameters.add_parameter(
            "a4",
            start_val=a4,
            enabled=True,
        )
        self.parameters.add_parameter(
            "a5",
            start_val=a5,
            enabled=True,
        )
        self.parameters.add_parameter(
            "a6",
            start_val=a6,
            enabled=True,
        )
        self.parameters.add_parameter(
            "d3",
            start_val=d3,
            enabled=True,
        )
        self.parameters.add_parameter(
            "d4",
            start_val=d4,
            enabled=True,
        )
        self.parameters.add_parameter(
            "d5",
            start_val=d5,
            enabled=True,
        )

    def _to_xml_element(self):
        return super()._to_xml_element(name="Extended-Mishin-Cu-F")


class UserFunction(DataContainer, BaseFunctionMixin):
    """
    Analytic functions that are not implemented in atomicrex
    can be provided as user functions.
    All parameters defined in the function should be added using the
    UserFunction.parameters.add_parameter() method.
    """

    def __init__(
        self,
        identifier=None,
        input_variable=None,
        species=["*", "*"],
        is_screening_function=False,
        cutoff=None,
    ):
        super().__init__(table_name=f"user_func_{identifier}")
        self.input_variable = input_variable
        self.identifier = identifier
        self.species = species
        self.parameters = FunctionParameterList()
        self.expression = None
        self.derivative = None
        self.is_screening_function = is_screening_function
        self.cutoff = cutoff
        if not is_screening_function:
            self.screening = None

    def _to_xml_element(self):
        if self.is_screening_function:
            screening = ET.Element("screening")
            root = ET.SubElement(screening, "user-function")
        else:
            root = ET.Element("user-function")
        root.set("id", f"{self.identifier}")
        input_var = ET.SubElement(root, "input-var")
        input_var.text = f"{self.input_variable}"
        expression = ET.SubElement(root, "expression")
        expression.text = f"{self.expression}"
        derivative = ET.SubElement(root, "derivative")
        derivative.text = f"{self.derivative}"

        for param in self.parameters.values():
            p = ET.SubElement(root, "param")
            p.set("name", f"{param.param}")
            p.text = f"{param.start_val:.6g}"  # 6g formatting because atomicrex output is limited to 6 significant digits, prevents some errors

        root.append(self.parameters.fit_dofs_to_xml_element())

        if self.cutoff is not None:
            cutoff = ET.SubElement(root, "cutoff")
            cutoff.text = f"{self.cutoff}"

        if not self.is_screening_function:
            if self.screening is not None:
                root.append(self.screening._to_xml_element())
            return root
        else:
            return screening

    def _parse_final_parameter(self, leftover, value):
        param = leftover[0].rstrip(":")
        self.parameters[param].final_value = value


class FunctionParameter(DataContainer):
    """
    Function parameter. For detailed information
    about the attributes see the atomicrex documentation.
    Objects should only be created using the add_parameter method
    of the FunctionParameterList class.
    """

    def __init__(
        self,
        param=None,
        start_val=None,
        enabled=True,
        reset=False,
        min_val=None,
        max_val=None,
        fitable=True,
        tag=None,
    ):
        super().__init__()
        self.param = param
        self.start_val = start_val
        self.enabled = enabled
        self.reset = reset
        self.min_val = min_val
        self.max_val = max_val
        self.tag = tag
        self.fitable = fitable
        self.final_value = None

    def _to_xml_element(self):
        root = ET.Element(f"{self.param}")
        if self.enabled:
            root.set("enabled", "true")
        else:
            root.set("enabled", "false")

        if self.reset:
            root.set("reset", "true")
        else:
            root.set("reset", "false")

        if self.min_val is not None:
            root.set("min", f"{self.min_val:.6g}")
        if self.max_val is not None:
            root.set("max", f"{self.max_val:.6g}")
        if self.tag is not None:
            root.set("tag", f"{self.tag}")
        return root

    def copy_final_to_start_value(self, filter_func=None):
        """
        Copies the final value to start_val.

        Raises:
            ValueError: Raises if fitting of the parameter is enabled,
                        but the final value is None. This should only be the case
                        if the job aborted or was not run yet.
        """
        if filter_func is not None:
            if not filter_func(self):
                return

        if self.final_value is None:
            if self.enabled:
                raise ValueError(
                    f"Fitting is enabled for {self.param}, but final value is None."
                )
        else:
            self.start_val = copy.copy(self.final_value)

    def set_max_val(self, constant=None, factor=None, filter_func=None):
        if filter_func is not None:
            if not filter_func(self):
                return

        self.max_val = constant
        if factor is not None:
            self.max_val = abs(self.start_val) * factor

    def set_min_val(self, constant=None, factor=None, filter_func=None):
        if filter_func is not None:
            if not filter_func(self):
                return

        self.min_val = constant
        if factor is not None:
            self.min_val = -abs(self.start_val) * factor

    def lock_value(self, filter_func=None):
        if filter_func is not None:
            if not filter_func(self):
                return
        self.enabled = False

    def randomize(
        self,
        rng,
        filter_func=None,
    ):
        if filter_func is not None:
            if not filter_func(self):
                return

        if self.enabled:
            if self.min_val is None or self.max_val is None:
                raise ValueError(
                    f"Min and/or max val not set for {self.param}, can't randomize"
                )

            self.start_val = rng.uniform(self.min_val, self.max_val)


class FunctionParameterList(DataContainer):
    def __init__(self):
        super().__init__(table_name="FunctionParameterList")

    def add_parameter(
        self,
        param,
        start_val,
        enabled=True,
        reset=False,
        min_val=None,
        max_val=None,
        tag=None,
        fitable=True,
    ):
        """
        Add a function parameter named param to a function.
        This needs to be done manually for user functions and
        not for special functions.

        Args:
            param (str): Name of the parameter. Must exactly match the name in the function expression.
            start_val (float): Starting value of the parameter
            enabled (bool, optional): Determines if the paremeter is varied during fitting. Defaults to True.
            reset (bool, optional): Determine if the parameter should be reset every iteration
            Can help with global optimization. Defaults to False.
            min_val (float, optional): Highly recommended for global optimization. Defaults to None.
            max_val (float, optional): Highly recommended for global optimization. Defaults to None.
            tag (str, optional): [description]. Only necessary for ABOP potentials .Defaults to None.
            fitable (bool, optional): [description]. Changing could cause bugs. Defaults to True.
        """
        self[param] = FunctionParameter(
            param,
            start_val,
            enabled=enabled,
            reset=reset,
            min_val=min_val,
            max_val=max_val,
            tag=tag,
            fitable=fitable,
        )

    def fit_dofs_to_xml_element(self):
        """Internal function
        Returns fit dofs as atomicrex xml element.
        """
        fit_dof = ET.Element("fit-dof")
        for param in self.values():
            if param.fitable:
                fit_dof.append(param._to_xml_element())
        return fit_dof


class PolyCoeff(FunctionParameter):
    """
    Function parameter, but for polynomial interpolation.
    """

    def __init__(
        self,
        n: int = None,
        start_val: float = None,
        enabled=True,
        reset=False,
        min_val=None,
        max_val=None,
    ):
        super().__init__(
            param="coeff",
            start_val=start_val,
            enabled=enabled,
            reset=reset,
            min_val=min_val,
            max_val=max_val,
            fitable=True,
            tag=None,
        )
        self.n = n

    def _to_xml_element(self):
        root = super()._to_xml_element()
        root.set("value", f"{self.start_val:.6g}")
        root.set("n", f"{self.n:.6g}")
        return root


class PolyCoeffList(DataContainer):
    def __init__(self):
        super().__init__(table_name="PolyCoeffList")

    def add_coeff(
        self, n, start_val, enabled=True, reset=False, min_val=None, max_val=None
    ):
        """
        Add a term in the form of a*x^n.

        Args:
            n (int): Order n of the coefficient
            start_val (float): Starting value of a.
            enabled (bool, optional): Determines if it should be fitted. Defaults to True.
            reset (bool, optional): Determines if it should be reset after each iteration. Defaults to False.
            min_val (float, optional): Highly recommended for global optimization. Defaults to None.
            max_val (float, optional): Highly recommended for global optimization. Defaults to None.
        """
        self[f"coeff_{n}"] = PolyCoeff(
            n,
            start_val,
            enabled,
            reset,
            min_val,
            max_val,
        )

    def _to_xml_element(self):
        coefficients = ET.Element("coefficients")
        for coeff in self.values():
            coefficients.append(coeff._to_xml_element())
        return coefficients


class Node(FunctionParameter):
    """
    Function parameter, but for spline interpolation.
    """

    def __init__(
        self,
        x=None,
        start_val=None,
        enabled=True,
        reset=False,
        min_val=None,
        max_val=None,
    ):
        super().__init__(
            param="node",
            start_val=start_val,
            enabled=enabled,
            reset=reset,
            min_val=min_val,
            max_val=max_val,
            fitable=True,
            tag=None,
        )
        self.x = x

    def _to_xml_element(self):
        node = super()._to_xml_element()
        node.set("x", f"{self.x:.6g}")
        node.set("y", f"{self.start_val:.6g}")
        return node


class NodeList(DataContainer):
    def __init__(self):
        super().__init__(table_name="NodeList")

    def add_node(
        self, x, start_val, enabled=True, reset=False, min_val=None, max_val=None
    ):
        """
        Add a node to the spline interpolation function.

        Args:
            x (float): x coordinate of the node. Does not change during fitting.
            start_val (float): Initial y coordinate of the node.
            enabled (bool, optional): Determines if y is changed during fitting. Defaults to True.
            reset (bool, optional): Determines if y should be reset every iteration. Defaults to False.
            min_val (float, optional): Highly recommended for global optimization. Defaults to None.
            max_val (float, optional): Highly recommended for global optimization. Defaults to None.
        """
        x = float(x)
        # atomicrex rounds output to 6 digits, so this is done here to prevent issues when reading the output.
        key = f"node_{x:.6g}"
        self[key] = Node(
            x=x,
            start_val=start_val,
            enabled=enabled,
            reset=reset,
            min_val=min_val,
            max_val=max_val,
        )
        return self[key]

    def _to_xml_element(self):
        nodes = ET.Element("nodes")
        for node in self.values():
            nodes.append(node._to_xml_element())
        return nodes

    def create_from_arrays(self, x, y, min_vals=None, max_vals=None):
        """
        Convenience function to create nodes from lists or arrays of values.
        Allows to easily start the fitting process with physically motivated values
        or values taken from previous potentials.
        Creates len(x) nodes at position x with starting values y.
        All given arrays must have the same length.

        Args:
            x (list or array): x values of the nodes
            y (list or array): corresponding y (starting) values
            min_vals ([type], optional): Highly recommended for global optimization. Defaults to None.
            max_vals ([type], optional): Highly recommended for global optimization. Defaults to None.
        """
        for i in range(len(x)):
            node = self.add_node(x[i], y[i])
            if min_vals is not None:
                node.min_val = min_vals[i]
            if max_vals is not None:
                node.max_val = max_vals[i]


def plot(func, x=np.linspace(0.01, 7.0, 351)):
    y = func(x)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, y)
    # These defaults should be fine for most potentials
    ax.set(xlim=[0.0, 7.0], ylim=[-3.0, 3.0], xlabel="r [$\AA$]", ylabel="func(r)")
    return fig, ax
