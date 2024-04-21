__version__ = "0.1"
__all__ = []

import warnings

from pyiron_base import Project as BaseProject, JOB_CLASS_DICT
from pyiron_base.project.maintenance import add_module_conversion


class Project(BaseProject):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Importing Project from pyiron_contrib is deprecated. "
            "Import from appropriate pyiron module (e.g. pyiron) instead."
        )
        super().__init__(*args, **kwargs)


# Make classes available for new pyiron version
JOB_CLASS_DICT.update(
    {
        "ProtoMinimGradDes": "pyiron_contrib.protocol.compound.minimize",
        "ProtoMD": "pyiron_contrib.protocol.compound.molecular_dynamics",
        "ProtoConfinedMD": "pyiron_contrib.protocol.compound.molecular_dynamics",
        "ProtoHarmMD": "pyiron_contrib.protocol.compound.molecular_dynamics",
        "ProtoNEBSer": "pyiron_contrib.protocol.compound.nudged_elastic_band",
        "ProtocolQMMM": "pyiron_contrib.protocol.compound.qmmm",
        "ProtoHarmTILDSer": "pyiron_contrib.protocol.compound.thermodynamic_integration",
        "ProtoHarmTILDPar": "pyiron_contrib.protocol.compound.thermodynamic_integration",
        "ProtoVacTILDSer": "pyiron_contrib.protocol.compound.thermodynamic_integration",
        "ProtoVacTILDPar": "pyiron_contrib.protocol.compound.thermodynamic_integration",
        "ProtoVacForm": "pyiron_contrib.protocol.compound.thermodynamic_integration",
        "ProtoFTSEvoSer": "pyiron_contrib.protocol.compound.finite_temperature_string",
        "ProtoFTSEvoPar": "pyiron_contrib.protocol.compound.finite_temperature_string",
        "ImageJob": "pyiron_contrib.image.job",
        "LangevinAse": "pyiron_contrib.atomistics.interactive.langevin",
        "Mixer": "pyiron_contrib.atomistics.interactive.mixer",
        "ParameterMaster": "pyiron_contrib.atomistics.dft.parametermaster",
        "MonteCarloMaster": "pyiron_contrib.atomistics.interactive.montecarlo",
        "RandSpg": "pyiron_contrib.atomistics.randspg.randspg",
        "Fenics": "pyiron_contrib.continuum.fenics.job.generic",
        "FenicsLinearElastic": "pyiron_contrib.continuum.fenics.job.elastic",
        "RandomDisMaster": "pyiron_contrib.atomistics.mlip.masters",
        "RandomMDMaster": "pyiron_contrib.atomistics.mlip.masters",
        "StructureMasterInt": "pyiron_contrib.atomistics.atomistics.job.structurelistmasterinteractive",
        "StorageJob": "pyiron_contrib.RDM.storagejob",
        "Cp2kJob": "pyiron_contrib.atomistics.cp2k.job",
        "PiMD": "pyiron_contrib.atomistics.ipi.ipi_jobs",
        "GleMD": "pyiron_contrib.atomistics.ipi.ipi_jobs",
        "PigletMD": "pyiron_contrib.atomistics.ipi.ipi_jobs",
        "QuasiHarmonicApproximation": "pyiron_contrib.atomistics.atomistics.master.qha",
        "ElasticTensor": "pyiron_contrib.atomistics.elastic.elastic",
    }
)

old_prefix = "pyiron_contrib.atomistics."
new_prefix = "pyiron_potentialfit."
moved_potential_modules = [
    "atomicrex",
    "atomicrex.atomicrex_job",
    "atomicrex.base",
    "atomicrex.fit_properties",
    "atomicrex.function_factory",
    "atomicrex.general_input",
    "atomicrex.interactive",
    "atomicrex.output",
    "atomicrex.parameter_constraints",
    "atomicrex.potential_factory",
    "atomicrex.structure_list",
    "atomicrex.utility_functions",
    "atomistics.job.trainingcontainer",
    # "fitsnap",
    # "fitsnap.common",
    # "fitsnap.job",
    "meamfit.meamfit",
    "ml",
    "ml.potentialfit",
    "mlip",
    "mlip.cfgs",
    "mlip.lammps",
    "mlip.masters",
    "mlip.mlip",
    "mlip.mlipdescriptors",
    "mlip.parser",
    "mlip.potential",
    "pacemaker",
    "pacemaker.job",
    "runner",
    "runner.job",
    "runner.storageclasses",
    "runner.utils",
]
for module in moved_potential_modules:
    add_module_conversion(old_prefix + module, new_prefix + module)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
