from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "NotInstalledYet"


import nmresearch.crystal.atom
import nmresearch.crystal.crystal
import nmresearch.crystal.disorder

from nmresearch.crystal.atom import Atom, AtomPos
from nmresearch.crystal.crystal import Crystal
from nmresearch.crystal.disorder import Disorder
