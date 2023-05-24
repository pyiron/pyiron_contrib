__version__ = "0.1"
__all__ = []

import warnings

try:
    from pyiron import Project
except:
    warnings.warn("pyiron module not found, importing Project from pyiron_base")
    from pyiron_base import Project

from pyiron_base import JOB_CLASS_DICT
from pyiron_contrib.generic.storage_interface_toolkit import StorageInterfaceFactory

Project.register_tools("storage_interface", StorageInterfaceFactory)

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
        "TrainingContainer": "pyiron_contrib.atomistics.atomistics.job.trainingcontainer",
        "RandomDisMaster": "pyiron_contrib.atomistics.mlip.masters",
        "RandomMDMaster": "pyiron_contrib.atomistics.mlip.masters",
        "RunnerFit": "pyiron_contrib.atomistics.runner.job",
        "MlipSelect": "pyiron_contrib.atomistics.mlip.mlipselect",
        "Mlip": "pyiron_contrib.atomistics.mlip.mlip",
        "LammpsMlip": "pyiron_contrib.atomistics.mlip.lammps",
        "MlipJob": "pyiron_contrib.atomistics.mlip.mlipjob",
        "Atomicrex": "pyiron_contrib.atomistics.atomicrex.atomicrex_job",
        "StructureMasterInt": "pyiron_contrib.atomistics.atomistics.job.structurelistmasterinteractive",
        "StorageJob": "pyiron_contrib.RDM.storagejob",
        "MlipDescriptors": "pyiron_contrib.atomistics.mlip.mlipdescriptors",
        "PacemakerJob": "pyiron_contrib.atomistics.pacemaker.job",
        "MeamFit": "pyiron_contrib.atomistics.meamfit.meamfit",
        "Cp2kJob": "pyiron_contrib.atomistics.cp2k.job",
        "PiMD": "pyiron_contrib.atomistics.ipi.ipi_jobs",
        "GleMD": "pyiron_contrib.atomistics.ipi.ipi_jobs",
        "PigletMD": "pyiron_contrib.atomistics.ipi.ipi_jobs",
        "LammpsInteractiveWithoutOutput": "pyiron_contrib.nofiles.lammps",
        "SQSJobWithoutOutput": "pyiron_contrib.nofiles.sqs",
        "ElasticMatrixJobWithoutFiles": "pyiron_contrib.nofiles.elastic",
        "MurnaghanWithoutFiles": "pyiron_contrib.nofiles.murn",
        "PhonopyJobWithoutFiles": "pyiron_contrib.nofiles.phonopy",
        "SQSMasterMPI": "pyiron_contrib.nofiles.master",
        "LAMMPSMinimizeMPI": "pyiron_contrib.nofiles.master",
        "LAMMPSElasticMPI": "pyiron_contrib.nofiles.master",
        "LAMMPSMinimizeElasticMPI": "pyiron_contrib.nofiles.master",
        "FitsnapJob": "pyiron_contrib.atomistics.fitsnap.job",
        "QuasiHarmonicApproximation": "pyiron_contrib.atomistics.atomistics.master.qha",
    }
)


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
