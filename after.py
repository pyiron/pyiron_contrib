def _minimize(project, name, structure, potential):
    job = project.atomistics.job.Lammps(name)
    job.structure = structure.copy()
    job.potential = potential
    job.calc_minimize()
    job.run()
    return job.output.energy_pot[-1]


def xy_energy(project, name, xy_structure, potential, host):
    return _minimize(project, name, xy_structure, potential)


def x_energy(project, name, xy_structure, potential, host):
    structure = xy_structure.copy()
    structure[1] = host
    return _minimize(project, name, structure, potential)


def y_energy(project, name, xy_structure, potential, host):
    structure = xy_structure.copy()
    structure[0] = host
    return _minimize(project, name, structure, potential)


def bulk_energy(project, name, xy_structure, potential, host):
    structure = xy_structure.copy()
    structure[0] = host
    structure[1] = host
    return _minimize(project, name, structure, potential)


def binding_energy(project, name, xy_structure, potential, host):
    E_xy = xy_energy(project, name + 'xy', xy_structure, potential, host)
    E_x = x_energy(project, name + 'x', xy_structure, potential, host)
    E_y = y_energy(project, name + 'y', xy_structure, potential, host)
    E_bulk = bulk_energy(project, name + 'bulk', xy_structure, potential, host)
    return (E_xy + E_bulk) - (E_x + E_y)