import scipy as sp
from random import random, seed
from numpy.random import normal
import numpy as np

from nmresearch.lanczos.op_basis import PauliMatrix, superop2pauli_liouville
from nmresearch.lanczos.utils import super_ham


class Hamiltonian:

    def __init__(self, sites, mat=None):
        self.sites = sites
        self.paulis = PauliMatrix(sites)
        self.ham = mat

    def H_central_spin(self, couplings=None):
        A = couplings
        if A is None or len(A) != self.sites - 1:
            A = [random() for _ in range(self.sites - 1)]

        L = self.sites
        if self.ham is None:
            self.ham = sp.sparse.csr_matrix((2**L, 2**L), dtype=np.complex128)
        for i in range(1, L):
            self.ham = self.ham + A[i - 1] * (
                self.paulis.sigmaX(0, L) @ self.paulis.sigmaX(i, L)
                + self.paulis.sigmaY(0, L) @ self.paulis.sigmaY(i, L)
            )

    def H_cs_scrambling(self, cs_coupling=None, bath_couplings=None):
        A = cs_coupling
        if A is None or len(A) != L - 1:
            A = [random() for _ in range(L - 1)]
        J = bath_couplings
        if J is None:
            J = [[0.1 * (random() - 0.5) for _ in range(i)] for i in range(L - 1)]

        L = self.sites
        self.H_central_spin(A)
        for i in range(1, L):
            for j in range(1, i):
                self.ham = self.ham + J[i - 1][j - 1] * (
                    self.paulis.sigmaZ(i, L) @ self.paulis.sigmaZ(j, L)
                    - 0.5
                    * (
                        self.paulis.sigmaX(i, L) @ self.paulis.sigmaX(j, L)
                        + self.paulis.sigmaY(i, L) @ self.paulis.sigmaY(j, L)
                    )
                )

    def to_super(self):
        return super_ham(self.ham)

    def to_pauli_perm(self):
        return superop2pauli_liouville(self.to_super)

    def clear(self):
        self.ham = None
