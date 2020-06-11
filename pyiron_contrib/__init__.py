from pyiron import Project
from pyiron.base.job.jobtype import JOB_CLASS_DICT

# Make classes available for new pyiron version
JOB_CLASS_DICT['ProtoMinimize'] = 'pyiron_contrib.protocol.compound.minimize'
JOB_CLASS_DICT['ProtoMD'] = 'pyiron_contrib.protocol.compound.md'
JOB_CLASS_DICT['ProtoConfinedMD'] = 'pyiron_contrib.protocol.compound.md'
JOB_CLASS_DICT['ProtoConfinedHarmonicMD'] = 'pyiron_contrib.protocol.compound.md'
JOB_CLASS_DICT['ProtoNEB'] = 'pyiron_contrib.protocol.compound.neb'
JOB_CLASS_DICT['ProtoNEBParallel'] = 'pyiron_contrib.protocol.compound.neb'
JOB_CLASS_DICT['ProtoQMMM'] = 'pyiron_contrib.protocol.compound.qmmm'
JOB_CLASS_DICT['ProtoHarmonicTILD'] = 'pyiron_contrib.protocol.compound.tild'
JOB_CLASS_DICT['ProtoHarmonicTILDParallel'] = 'pyiron_contrib.protocol.compound.tild'
JOB_CLASS_DICT['ProtoVacancyTILD'] = 'pyiron_contrib.protocol.compound.tild'
JOB_CLASS_DICT['ProtoStringEvolution'] = 'pyiron_contrib.protocol.compound.fts'
JOB_CLASS_DICT['ProtoStringEvolutionParallel'] = 'pyiron_contrib.protocol.compound.fts'

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