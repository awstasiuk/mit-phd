from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "NotInstalledYet"

import nmresearch.fermion
import nmresearch.interface
import nmresearch.crystal
import nmresearch.percolation

import nmresearch.fermion.operator
import nmresearch.fermion.majorana
import nmresearch.fermion.math
import nmresearch.fermion.unitary
import nmresearch.interface.experiment
import nmresearch.interface.pp_gen
import nmresearch.interface.pulse_program
import nmresearch.crystal.atom
import nmresearch.crystal.crystal
import nmresearch.crystal.disorder
import nmresearch.lanczos.lanczos
import nmresearch.lanczos.op_basis
import nmresearch.lanczos.utils
import nmresearch.lanczos.hamiltonian


from nmresearch.fermion.operator import Operator
from nmresearch.fermion.majorana import PauliString, MajoranaString
from nmresearch.fermion.math import Math
from nmresearch.fermion.unitary import Unitary
from nmresearch.interface.experiment import Experiment
from nmresearch.interface.pp_gen import PulseProgram
from nmresearch.interface.pulse_program import TwoPointCorrelator
from nmresearch.crystal.atom import Atom, AtomPos
from nmresearch.crystal.crystal import Crystal
from nmresearch.crystal.disorder import Disorder
from nmresearch.lanczos.lanczos import Lanczos
from nmresearch.lanczos.hamiltonian import Hamiltonian
from nmresearch.lanczos.op_basis import PauliMatrix, vec, devec
from nmresearch.lanczos.utils import *
