from fermion.math import Math as fm
from fermion.operator import Operator

import numpy as np
from scipy import linalg as la
from math import e, pi
import itertools
import tensorflow as tf


class Majorana(Operator):
    r"""
    A class desciribing a quadratic multi-body Majorana-fermionic operator
    """

    def __init__(self, op_or_coef):
        r"""
        initialize the object
        """
        if isinstance(op_or_coef, Operator):
            op = op_or_coef
            T = np.kron([[1, 1], [-1j, 1j]] / np.sqrt(2), np.eye(n))
            coef = {}
            for k in op.components:
                if k == 0:
                    coef[0] = op.coef[0]
                elif k == 1:
                    coef[1] = T @ op.coef[1]
                else:
                    coef[k] = fm.tensor_change_of_basis(op.coef[k], T)
            super().__init__(op.n_fermion, coef)
        else:
            # this step should be sanitized, but probably we will never instantiate directly
            # from coeficients
            super().__init__(len(op_or_coef[2]), op_or_coef)

    def _maj_string(self, idx):
        n = self.n_fermion
        op_list = []
        for i in idx:
            if i < n:
                op_list.append("A" + str(i))
            else:
                op_list.append("B" + str(i % n))
        return "".join(op_list)

    def __str__(self):
        op = []
        for comp, mat in self.coef.items():
            if comp == 0 and mat != 0:
                op.append(str(mat) + "*I")
            else:
                for idx, val in np.ndenumerate(mat):
                    if val != 0:
                        op.append(str(val) + "*" + self._maj_string(idx))
        return "0" if len(op) == 0 else " + ".join(op)


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
                majoranas.extend([c for _ in range(1, j) for c in ["A", "B"]])
                sites.extend([idx for i in range(1, j) for idx in [i, i]])
                evo_bools.extend([ev for _ in range(1, j) for ev in [evo, evo]])

                sites.append(j)
                evo_bools.append(evo)
                pre = pre * (-2j) ** (j - 1) * np.sqrt(2)
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
