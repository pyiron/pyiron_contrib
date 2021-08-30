__version__ = "0.1"
__all__ = []

import warnings

try:
	from pyiron import Project
except:
	warnings.warn("pyiron module not found, importing Project from pyiron_base")
	from pyiron_base import Project

from pyiron_base import JOB_CLASS_DICT

# Make classes available for new pyiron version
JOB_CLASS_DICT['ProtoMinimGradDes'] = 'pyiron_contrib.protocol.compound.minimize'
JOB_CLASS_DICT['ProtoMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtoConfinedMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtoHarmMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtoNEBSer'] = 'pyiron_contrib.protocol.compound.nudged_elastic_band'
JOB_CLASS_DICT['ProtocolQMMM'] = 'pyiron_contrib.protocol.compound.qmmm'
JOB_CLASS_DICT['ProtoHarmTILDSer'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtoHarmTILDPar'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtoVacTILDSer'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtoVacTILDPar'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtoVacForm'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtoFTSEvoSer'] = 'pyiron_contrib.protocol.compound.finite_temperature_string'
JOB_CLASS_DICT['ProtoFTSEvoPar'] = 'pyiron_contrib.protocol.compound.finite_temperature_string'
JOB_CLASS_DICT['ImageJob'] = 'pyiron_contrib.image.job'
JOB_CLASS_DICT['LangevinAse'] = 'pyiron_contrib.atomistics.interactive.langevin'
JOB_CLASS_DICT['Mixer'] = 'pyiron_contrib.atomistics.interactive.mixer'
JOB_CLASS_DICT['ParameterMaster'] = 'pyiron_contrib.atomistics.dft.parametermaster'
JOB_CLASS_DICT['MonteCarloMaster'] = 'pyiron_contrib.atomistics.interactive.montecarlo'
JOB_CLASS_DICT['RandSpg'] = 'pyiron_contrib.atomistics.randspg.randspg'
JOB_CLASS_DICT['Fenics'] = 'pyiron_contrib.continuum.fenics.job.generic'
JOB_CLASS_DICT['FenicsLinearElastic'] = 'pyiron_contrib.continuum.fenics.job.elastic'
JOB_CLASS_DICT['TrainingContainer'] = 'pyiron_contrib.atomistics.atomistics.job.trainingcontainer'
JOB_CLASS_DICT['RandomDisMaster'] = 'pyiron_contrib.atomistics.mlip.masters'
JOB_CLASS_DICT['RandomMDMaster'] = 'pyiron_contrib.atomistics.mlip.masters'
JOB_CLASS_DICT['RunnerFit'] = 'pyiron_contrib.atomistics.runner.job'
JOB_CLASS_DICT['MlipSelect'] = 'pyiron_contrib.atomistics.mlip.mlipselect'
JOB_CLASS_DICT['Mlip'] = 'pyiron_contrib.atomistics.mlip.mlip'
JOB_CLASS_DICT['LammpsMlip'] = 'pyiron_contrib.atomistics.mlip.lammps'
JOB_CLASS_DICT['MlipJob'] = 'pyiron_contrib.atomistics.mlip.mlipjob'
JOB_CLASS_DICT['Atomicrex'] = 'pyiron_contrib.atomistics.atomicrex.atomicrex_job'
JOB_CLASS_DICT['StructureMasterInt'] = 'pyiron_contrib.atomistics.atomistics.job.structurelistmasterinteractive'
JOB_CLASS_DICT['StorageJob'] = 'pyiron_contrib.RDM.storagejob'


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
