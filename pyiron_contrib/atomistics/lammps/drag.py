import numpy as np


def setup_lmp_input(lmp, n_atoms=None, direction=None, fix_id=-1):
    """
    Change input for LAMMPS to run a drag calculation.

    Args:
        lmp (pyiron_atomistics.lammps.lammps.Lammps): LAMMPS job
        n_atoms (int): number of free atoms (default: None, i.e. it is
            determined from the job structure)
        direction (None/numpy.ndarray): direction along which the force is
            cancelled. None if all forces are to be cancelled (default: None)
        fix_id (None/int): id of the atom to be fixed (default: -1, i.e. last
            atom)

    Returns:
        None (input of lmp is changed in-place)

    In the context of this function, a drag calculation is a constraint energy
    minimization, in which one atom is either not allowed to move at all, or
    not allowed to move along a given direction. In order for the system to not
    fall to the energy minimum, the sum of the remaining forces is set to 0.

    Exp: Hydrogen diffusion

    >>> from pyiron_atomistics import Project
    >>> import numpy as np
    >>> pr = Project("DRAG")
    >>> bulk = pr.create.structure.bulk('Ni', cubic=True)
    >>> a_0 = bulk.cell[0, 0]
    >>> x_octa = np.array([0, 0, 0.5 * a_0])
    >>> x_tetra = np.array(3 * [0.25 * a_0])
    >>> dx = x_tetra - x_octa
    >>> transition = np.linspace(0, 1, 101)
    >>> x_lst = transition[:, None] * dx + x_octa
    >>> structure = bulk.repeat(4)
    >>> structure += pr.create.structure.atoms(
    ...     positions=[x_octa],
    ...     elements=['H'],
    ...     cell=structure.cell
    ... )
    >>> lmp = pr.create.job.Lammps('lmp')
    >>> lmp.structure = structure
    >>> lmp.calc_minimize()
    >>> lmp.potential = potential_of_your_choice
    >>> setup_lmp_input(lmp, direction=dx)
    >>> lmp.interactive_open()
    >>> for xx in x_lst:
    >>>     lmp.structure.positions[-1] = xx
    >>>     lmp.run()
    >>> lmp.interactive_close()
    """
    if lmp.input.control["minimize"] is None:
        raise ValueError("set calc_minimize first")
    if n_atoms is None:
        try:
            n_atoms = len(lmp.structure) - 1
        except TypeError:
            raise AssertionError("either `n_atoms` or the structure must be set")
    fix_id = np.arange(n_atoms + 1)[fix_id] + 1
    lmp.input.control["atom_modify"] = "map array"
    lmp.input.control["group___fixed"] = f"id {fix_id}"
    lmp.input.control["group___free"] = "subtract all fixed"
    if direction is None:
        for ii, xx in enumerate(["x", "y", "z"]):
            lmp.input.control[f"variable___f{xx}_free"] = (
                f"equal f{xx}[{fix_id}]/{n_atoms}"
            )
            lmp.input.control[f"variable___f{xx}_fixed"] = f"equal -f{xx}[{fix_id}]"
    else:
        direction = np.array(direction) / np.linalg.norm(direction)
        direction = np.outer(direction, direction)
        direction = np.around(direction, decimals=8)
        for grp, ss in zip(["free", "fixed"], [f"1/{n_atoms}*", "-"]):
            for ii, xx in enumerate(["x", "y", "z"]):
                txt = "+".join(
                    [
                        f"({ss}f{xxx}[{fix_id}]*({direction[ii][iii]}))"
                        for iii, xxx in enumerate(["x", "y", "z"])
                        if not np.isclose(direction[ii][iii], 0)
                    ]
                )
                if txt == "":
                    txt = 0
                lmp.input.control[f"variable___f{xx}_{grp}"] = f" equal {txt}"
    lmp.input.control["variable___energy"] = "atom 0"
    for key in ["free", "fixed"]:
        txt = " ".join([f"v_f{x}_{key}" for x in ["x", "y", "z"]])
        lmp.input.control[f"fix___f_{key}"] = f"{key} addforce {txt} energy v_energy"
    lmp.input.control["min_style"] = "quickmin"
