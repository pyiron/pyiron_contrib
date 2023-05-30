# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Parsers for MTP/Mlip related files.
"""

import pyparsing as pp
import numpy as np

__author__ = "Marvin Poul"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Marvin Poul"
__email__ = "poul@mpie.de"
__status__ = "development"
__date__ = "Aug 18, 2021"

pp.ParserElement.setDefaultWhitespaceChars(" ")


def _make_potential_parser():
    NL = pp.Suppress("\n")
    EQ = pp.Suppress("=")
    LB = pp.Suppress("{")
    RB = pp.Suppress("}")

    def make_keyword(name):
        return pp.Suppress(pp.Keyword(name))

    def make_field(name, expr, key=None, ungroup=True):
        key = name if key is None else key
        field_expr = make_keyword(name) + EQ + expr + NL[0, 1]
        if ungroup:
            field_expr = pp.ungroup(field_expr)
        field_expr = field_expr.set_results_name(key)
        field_expr.set_name(key)
        return field_expr

    def make_list(field_expr, grouped=False):
        list_expr = LB + pp.delimited_list(field_expr) + RB
        if grouped:
            list_expr = pp.Group(list_expr)
        list_expr = list_expr.set_name("list")
        return list_expr

    radial_basis_type = make_field("radial_basis_type", "RBChebyshev", "basis_type")
    radial_info = pp.IndentedBlock(
        make_field("min_dist", pp.pyparsing_common.fnumber)
        + make_field("max_dist", pp.pyparsing_common.fnumber)
        + make_field("radial_basis_size", pp.pyparsing_common.integer, "basis_size")
        + make_field("radial_funcs_count", pp.pyparsing_common.integer, "funcs_count"),
    ).set_results_name("info")
    radial_info.set_parse_action(lambda tk: tk[0].as_dict())

    radial_func_types = pp.Word(pp.nums) + pp.Suppress("-") + pp.Word(pp.nums) + NL
    radial_func_types.set_name("radial function types")
    radial_func_types.set_parse_action(lambda tokens: f"{tokens[0]}-{tokens[1]}")
    radial_func_coeffs = make_list(pp.pyparsing_common.fnumber, grouped=True)
    radial_funcs = pp.IndentedBlock(
        make_keyword("radial_coeffs")
        + NL
        + pp.IndentedBlock(
            radial_func_types + pp.IndentedBlock(radial_func_coeffs + NL)[1, ...]
        )[1, ...]
    )

    radial_funcs = radial_funcs.set_results_name("funcs")
    radial_funcs = pp.ungroup(radial_funcs).set_results_name("funcs")
    radial_funcs.set_parse_action(lambda tokens: {k: v for k, v in tokens.as_list()[0]})

    radial = pp.Group(radial_basis_type + radial_info + radial_funcs)
    radial = radial.set_results_name("radial")

    MTP = make_keyword("MTP") + NL

    parser = (
        NL[0, ...]
        + MTP
        + pp.Each(
            [
                make_field("version", pp.Word(pp.nums + ".")),
                make_field("potential_name", pp.Word(pp.alphanums)),
                make_field("scaling", pp.pyparsing_common.fnumber),
                make_field("species_count", pp.pyparsing_common.integer),
                make_field("potential_tag", pp.Optional(pp.Word(pp.alphanums), "")),
                radial,
                make_field("alpha_moments_count", pp.pyparsing_common.integer),
                make_field("alpha_index_basic_count", pp.pyparsing_common.integer),
                make_field(
                    "alpha_index_basic",
                    make_list(make_list(pp.pyparsing_common.integer, grouped=True)),
                    ungroup=False,
                ),
                make_field("alpha_index_times_count", pp.pyparsing_common.integer),
                make_field(
                    "alpha_index_times",
                    make_list(make_list(pp.pyparsing_common.integer, grouped=True)),
                    ungroup=False,
                ),
                make_field("alpha_scalar_moments", pp.pyparsing_common.integer),
                make_field(
                    "alpha_moment_mapping",
                    make_list(pp.pyparsing_common.integer),
                    ungroup=False,
                ),
                make_field(
                    "species_coeffs",
                    make_list(pp.pyparsing_common.fnumber),
                    ungroup=False,
                ),
                make_field(
                    "moment_coeffs",
                    make_list(pp.pyparsing_common.fnumber),
                    ungroup=False,
                ),
            ]
        )
    )

    return parser


def potential(potential_string):
    """
    Parse an MTP potential for mlip.

    Args:
        potential_string (str): string to parse

    Raises:
        ValueError: failed to parse potential
    """
    try:
        result = _make_potential_parser().parse_string(potential_string).as_dict()
        result["radial"]["basis_type"] = result["radial"]["basis_type"][
            2:
        ]  # strip RB prefix
        # Convert to numpy arrays
        for pair, func in result["radial"]["funcs"].items():
            result["radial"]["funcs"][pair] = np.array(result["radial"]["funcs"][pair])
        result["alpha_index_basic"] = np.array(result["alpha_index_basic"])
        result["alpha_index_times"] = np.array(result["alpha_index_times"])
        result["alpha_moment_mapping"] = np.array(result["alpha_moment_mapping"])
        result["species_coeffs"] = np.array(result["species_coeffs"])
        result["moment_coeffs"] = np.array(result["moment_coeffs"])
        basis_size = result["radial"]["info"]["basis_size"]
        funcs_size = result["radial"]["info"]["funcs_count"]
        for pair, func in result["radial"]["funcs"].items():
            if func.shape != (funcs_size, basis_size):
                raise ValueError(
                    f"Invalid radial basis for pair {pair}, should be {funcs_count}x{ basis_size} not {result['radial']['funcs'].shape}"
                )
        if result["alpha_index_basic"].shape != (result["alpha_index_basic_count"], 4):
            raise ValueError(
                f"Invalid alpha basic indices, length should be {result['alpha_index_basic_count']}"
            )
        if result["alpha_index_times"].shape != (result["alpha_index_times_count"], 4):
            raise ValueError(
                f"Invalid alpha times indices, length should be {result['alpha_index_times_count']}"
            )
        if len(result["alpha_moment_mapping"]) != result["alpha_scalar_moments"]:
            raise ValueError(
                f"Invalid alpha moment mapping, length should be {result['alpha_scalar_moments']}"
            )
        if len(result["moment_coeffs"]) != result["alpha_scalar_moments"]:
            raise ValueError(
                f"Invalid moment coefficients, length should be {result['alpha_scalar_moments']}"
            )
        if len(result["species_coeffs"]) != result["species_count"]:
            raise ValueError(
                f"Invalid species coefficients, length should be {result['species_count']}"
            )

        return result
    except pp.ParseException:
        raise ValueError("failed to parse potential") from None
