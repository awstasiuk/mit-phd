from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "NotInstalledYet"

import nmresearch.fermion
import nmresearch.interface

import nmresearch.fermion.operator
import nmresearch.fermion.majorana
import nmresearch.fermion.math
import nmresearch.fermion.unitary
import nmresearch.interface.experiment
import nmresearch.interface.pp_gen

from nmresearch.fermion.operator import Operator
from nmresearch.fermion.majorana import PauliString, MajoranaString
from nmresearch.fermion.math import Math
from nmresearch.fermion.unitary import Unitary
from nmresearch.interface.experiment import Experiment
from nmresearch.interface.pp_gen import PulseProgram
