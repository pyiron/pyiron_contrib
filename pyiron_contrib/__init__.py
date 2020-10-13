from pyiron import Project
from pyiron.base.job.jobtype import JOB_CLASS_DICT

# Make classes available for new pyiron version
JOB_CLASS_DICT['ProtocolMinimize'] = 'pyiron_contrib.protocol.compound.minimize'
JOB_CLASS_DICT['ProtocolMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtocolHarmonicMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtocolConfinedMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtocolConfinedHarmonicMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtocolNEB'] = 'pyiron_contrib.protocol.compound.nudged_elastic_band'
JOB_CLASS_DICT['ProtocolQMMM'] = 'pyiron_contrib.protocol.compound.qmmm'
JOB_CLASS_DICT['ProtocolHarmonicTILD'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtocolHarmonicTILDParallel'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtocolVacancyTILD'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtocolVacancyTILDParallel'] = 'pyiron_contrib.protocol.compound.thermodynamic_integration'
JOB_CLASS_DICT['ProtocolFTSEvolution'] = 'pyiron_contrib.protocol.compound.finite_temperature_string'
JOB_CLASS_DICT['ProtocolFTSEvolutionParallel'] = 'pyiron_contrib.protocol.compound.finite_temperature_string'

# Backwards compatibility
JOB_CLASS_DICT['GenericMaster'] = 'pyiron.base.master.generic'
JOB_CLASS_DICT['ListMaster'] = 'pyiron.base.master.list'
JOB_CLASS_DICT['ParallelMaster'] = 'pyiron.base.master.parallel'
JOB_CLASS_DICT['VaspInt'] = 'pyiron_mpie.backwards.back'
JOB_CLASS_DICT['VaspInt2'] = 'pyiron_mpie.backwards.back'
JOB_CLASS_DICT['LammpsInt'] = 'pyiron_mpie.backwards.back'
JOB_CLASS_DICT['LammpsInt2'] = 'pyiron_mpie.backwards.back'
JOB_CLASS_DICT['PhonopyMaster'] = 'pyiron_mpie.backwards.back'
JOB_CLASS_DICT['PhonopyMaster2'] = 'pyiron_mpie.backwards.back'
JOB_CLASS_DICT['MurnaghanInt'] = 'pyiron_mpie.backwards.back'
JOB_CLASS_DICT['SphinxEx'] = 'pyiron_mpie.backwards.back'
JOB_CLASS_DICT['SphinxInt'] = 'pyiron_mpie.backwards.back'
JOB_CLASS_DICT['SphinxInt2'] = 'pyiron_mpie.backwards.back'