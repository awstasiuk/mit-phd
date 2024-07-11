from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "NotInstalledYet"

import nmresearch.lanczos.lanczos
import nmresearch.lanczos.op_basis
import nmresearch.lanczos.utils
import nmresearch.lanczos.hamiltonian

from nmresearch.lanczos.lanczos import *
from nmresearch.lanczos.op_basis import *
from nmresearch.lanczos.utils import *
from nmresearch.lanczos.hamiltonian import *
