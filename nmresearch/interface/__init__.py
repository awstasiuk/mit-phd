from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "NotInstalledYet"

import nmresearch.interface.experiment
import nmresearch.interface.pp_gen
import nmresearch.interface.pulse_program

from nmresearch.interface.experiment import Experiment
from nmresearch.interface.pp_gen import PulseProgram
from nmresearch.interface.pulse_program import TwoPointCorrelator
