from pyiron_contrib.protocol.generic import CompoundVertex
from pyiron_contrib.protocol.compound.minimize import Minimize
from pyiron_contrib.protocol.compound.molecular_dynamics import MolecularDynamics
from pyiron_contrib.protocol.compound.nudged_elastic_band import NEB, NEBParallel, NEBSerial
from pyiron_contrib.protocol.compound.tild import HarmonicTILD, VacancyTILD
from pyiron_contrib.protocol.compound.finite_temperature_string import StringRelaxation, VirtualWork, \
    Milestoning, VirtualWorkParallel, VirtualWorkSerial, VirtualWorkFullStep
from pyiron_contrib.protocol.compound.qmmm import QMMMProtocol

# protocol is a magic class after this one we have to Register
from pyiron_contrib.protocol.utils.types import PyironJobTypeRegistryMetaType
PyironJobTypeRegistryMetaType.inject_dynamic_types()
__all__ = [
    'CompoundVertex',
    'Minimize',
    'MolecularDynamics',
    'NEB', 'NEBParallel', 'NEBSerial',
    'HarmonicTILD', 'VacancyTILD',
    'StringRelaxation', 'VirtualWork', 'Milestoning', 'VirtualWorkParallel', 'VirtualWorkSerial', 'VirtualWorkFullStep',
    'QMMMProtocol'
]
