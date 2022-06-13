from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "NotInstalledYet"

import nmresearch.fermion.operator
import nmresearch.fermion.majorana
import nmresearch.fermion.math
import nmresearch.fermion.unitary

from nmresearch.fermion.operator import Operator
from nmresearch.fermion.majorana import PauliString, MajoranaString
from nmresearch.fermion.math import Math
from nmresearch.fermion.unitary import Unitary
