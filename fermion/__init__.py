from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "NotInstalledYet"

import fermion.operator
import fermion.math

from fermion.operator import Operator
from fermion.math import Math