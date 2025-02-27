from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "NotInstalledYet"

import nmresearch.percolation.generators
import nmresearch.percolation.graph

from nmresearch.percolation.generators import *
from nmresearch.percolation.graph import *
