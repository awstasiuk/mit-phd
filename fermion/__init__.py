from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "NotInstalledYet"

import fermion.operator
import fermion.majorana
import fermion.math
import fermion.unitary

from fermion.operator import Operator
from fermion.majorana import Majorana, PauliString, MajoranaString
from fermion.math import Math
from fermion.unitary import Unitary
