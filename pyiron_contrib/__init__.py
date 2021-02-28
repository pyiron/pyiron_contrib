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
JOB_CLASS_DICT['ProtocolMinimize'] = 'pyiron_contrib.protocol.compound.minimize'
JOB_CLASS_DICT['ProtocolMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtocolConfinedMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtocolHarmonicMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtocolNEB'] = 'pyiron_contrib.protocol.compound.nudged_elastic_band'
JOB_CLASS_DICT['ProtocolQMMM'] = 'pyiron_contrib.protocol.compound.qmmm'
JOB_CLASS_DICT['ProtocolHarmonicTILD'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtocolHarmonicTILDParallel'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtocolVacancyTILD'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtocolVacancyTILDParallel'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtocolVacancyFormation'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtocolFTSEvolution'] = 'pyiron_contrib.protocol.compound.finite_temperature_string'
JOB_CLASS_DICT['ProtocolFTSEvolutionParallel'] = 'pyiron_contrib.protocol.compound.finite_temperature_string'
JOB_CLASS_DICT['ImageJob'] = 'pyiron_contrib.image.job'
JOB_CLASS_DICT['LangevinAse'] = 'pyiron_contrib.atomistic.interactive.langevin'
JOB_CLASS_DICT['Mixer'] = 'pyiron_contrib.atomistic.interactive.mixer'
JOB_CLASS_DICT['ParameterMaster'] = 'pyiron_contrib.atomistic.dft.parametermaster'
JOB_CLASS_DICT['MonteCarloMaster'] = 'pyiron_contrib.atomistic.interactive.montecarlo'
JOB_CLASS_DICT['RandSpg'] = 'pyiron_contrib.atomistic.atomistics.structures.randspg'
JOB_CLASS_DICT['Fenics'] = 'pyiron_contrib.continuum.fenics.job.generic'
JOB_CLASS_DICT['FenicsLinearElastic'] = 'pyiron_contrib.continuum.fenics.job.elastic'
JOB_CLASS_DICT['TrainingContainer'] = 'pyiron_contrib.atomistic.atomistics.job.trainingcontainer'
JOB_CLASS_DICT['RandomDisMaster'] = 'pyiron_contrib.atomistic.mlip.masters'
JOB_CLASS_DICT['RandomMDMaster'] = 'pyiron_contrib.atomistic.mlip.masters'
JOB_CLASS_DICT['MlipSelect'] = 'pyiron_contrib.atomistic.mlip.mlipselect'
JOB_CLASS_DICT['Mlip'] = 'pyiron_contrib.atomistic.mlip.mlip'
JOB_CLASS_DICT['LammpsMlip'] = 'pyiron_contrib.atomistic.mlip.lammps'
JOB_CLASS_DICT['MlipJob'] = 'pyiron_contrib.atomistic.mlip.mlipjob'


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
