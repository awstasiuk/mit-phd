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
        self.dim = 2**sites

    def H_central_spin(self, couplings=None):
        A = couplings
        if A is None or len(A) != self.sites - 1:
            A = [random() for _ in range(self.sites - 1)]

        L = self.sites
        if self.ham is None:
            self.ham = sp.sparse.csr_matrix((2**L, 2**L), dtype=np.complex128)
        for i in range(1, L):
            self.ham = self.ham + A[i - 1] * (
                self.paulis.sigmaX(0) @ self.paulis.sigmaX(i)
                + self.paulis.sigmaY(0) @ self.paulis.sigmaY(i)
            )

    def H_cs_scrambling(self, cs_coupling=None, bath_couplings=None):
        A = cs_coupling
        L = self.sites
        if A is None or len(A) != L - 1:
            A = [random() for _ in range(L - 1)]
        J = bath_couplings
        if J is None:
            J = [[0.1 * (random() - 0.5) for _ in range(i)] for i in range(L - 1)]

        self.H_central_spin(A)
        for i in range(1, L):
            for j in range(1, i):
                self.ham = self.ham + J[i - 1][j - 1] * (
                    self.paulis.sigmaZ(i) @ self.paulis.sigmaZ(j)
                    - 0.5
                    * (
                        self.paulis.sigmaX(i) @ self.paulis.sigmaX(j)
                        + self.paulis.sigmaY(i) @ self.paulis.sigmaY(j)
                    )
                )

    def to_super(self):
        return super_ham(self.ham)

    def to_pauli_perm(self):
        return superop2pauli_liouville(self.to_super())

    def clear(self):
        self.ham = None
