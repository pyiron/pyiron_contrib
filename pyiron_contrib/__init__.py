from pyiron import Project
from pyiron_base import JOB_CLASS_DICT

# Make classes available for new pyiron version
JOB_CLASS_DICT['ProtocolMinimize'] = 'pyiron_contrib.protocol.compound.minimize'
JOB_CLASS_DICT['ProtocolMD'] = 'pyiron_contrib.protocol.compound.molecular_dynamics'
JOB_CLASS_DICT['ProtocolNEB'] = 'pyiron_contrib.protocol.compound.nudged_elastic_band'
JOB_CLASS_DICT['ProtocolQMMM'] = 'pyiron_contrib.protocol.compound.qmmm'
JOB_CLASS_DICT['ProtocolHarmonicTILD'] = 'pyiron_contrib.protocol.compound.tild'
JOB_CLASS_DICT['ProtocolHarmonicTILDParallel'] = 'pyiron_contrib.protocol.compound.tild'
JOB_CLASS_DICT['ImageJob'] = 'pyiron_contrib.image.job'
JOB_CLASS_DICT['SQSElasticConstants'] = 'pyiron_contrib.atomistic.atomistics.master.sqselastic'
JOB_CLASS_DICT['SQSElasticConstantsList'] = 'pyiron_contrib.atomistic.atomistics.master.sqselastic'