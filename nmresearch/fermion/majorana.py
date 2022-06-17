from nmresearch.fermion.math import Math as fm
from nmresearch.fermion.operator import Operator

import numpy as np
from scipy import linalg as la
from math import e, pi
import itertools
import tensorflow as tf


class PauliString:
    P_CHARS = ["X", "Y", "Z"]

    def __init__(self, paulis=[], sites=[], evo_bools=[]):
        self._paulis = paulis
        self._sites = sites
        self._evo_bools = evo_bools

    @property
    def paulis(self):
        return self._paulis

    @property
    def sites(self):
        return self._sites

    @property
    def evo_bools(self):
        return self._evo_bools

    def add_pauli(self, pauli, site, evolved):
        if (
            pauli in PauliString.P_CHARS
            and isinstance(site, int)
            and isinstance(evolved, bool)
        ):
            self._paulis.append(pauli)
            self._sites.append(site)
            self._evo_bools.append(evolved)

    def __mul__(self, other):
        if isinstance(other, PauliString):
            return PauliString(
                self.paulis + other.paulis,
                self.sites + other.sites,
                self.evo_bools + other.evo_bools,
            )
        else:
            raise ValueError("can't do that!")

    def __str__(self):
        out = []
        for pauli, j, evolved in zip(self.paulis, self.sites, self.evo_bools):
            out.append(pauli)
            out.append(str(j))
            if evolved:
                out.append("(t)")
        return "".join(out)

    def __repr__(self):
        return str(self)


class MajoranaString:
    M_CHARS = ["A", "B"]

    def __init__(self, majoranas=[], sites=[], evo_bools=[], pre=1):
        self._majoranas = majoranas
        self._sites = sites
        self._evo_bools = evo_bools
        self._pre = pre

    @property
    def majoranas(self):
        return self._majoranas

    @property
    def sites(self):
        return self._sites

    @property
    def evo_bools(self):
        return self._evo_bools

    @property
    def pre_factor(self):
        return self._pre

    @staticmethod
    def from_pauli_string(p_string):
        majoranas = []
        sites = []
        evo_bools = []
        pre = 1
        for pauli, j, evo in zip(p_string.paulis, p_string.sites, p_string.evo_bools):
            if pauli == "Z":
                majoranas.extend(["A", "B"])
                sites.extend([j, j])
                evo_bools.extend([evo, evo])
                pre = pre * -2j
            elif pauli in ["X", "Y"]:
                majoranas.extend([c for _ in range(j) for c in ["A", "B"]])
                sites.extend([idx for i in range(j) for idx in [i, i]])
                evo_bools.extend([ev for _ in range(j) for ev in [evo, evo]])

                sites.append(j)
                evo_bools.append(evo)
                pre = pre * (-2j) ** (j) * np.sqrt(2)
                if pauli == "X":
                    majoranas.append("A")
                    pre = -1 * pre
                else:
                    majoranas.append("B")
        return MajoranaString(majoranas, sites, evo_bools, pre)

    def add_majorana(self, majorana, site, evolved):
        if (
            majorana in MajoranaString.P_CHARS
            and isinstance(site, int)
            and isinstance(evolved, bool)
        ):
            self._paulis.append(majorana)
            self._sites.append(site)
            self._evo_bools.append(evolved)

    def __len__(self):
        return len(self.majoranas)

    def __mul__(self, other):
        if isinstance(other, MajoranaString):
            return MajoranaString(
                self.majoranas + other.majoranas,
                self.sites + other.sites,
                self.evo_bools + other.evo_bools,
                self.pre_factor * other.pre_factor,
            )
        else:
            raise ValueError("can't do that!")

    def __str__(self):
        out = [str(self.pre_factor), " * "]
        for maj, j, evolved in zip(self.majoranas, self.sites, self.evo_bools):
            out.append(maj)
            out.append(str(j))
            if evolved:
                out.append("(t)")
        return "".join(out)

    def __repr__(self):
        return str(self)
