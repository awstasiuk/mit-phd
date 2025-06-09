from scipy.sparse import csr_matrix
from random import random
from numpy import complex128

from nmresearch.lanczos.op_basis import PauliMatrix, superop2pauli_liouville
from nmresearch.lanczos.utils import super_ham, super_ham_alt, random_ball


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
        d = self.dim
        p = self.paulis
        h = self.ham
        if h is None:
            h = csr_matrix((d, d), dtype=complex128)
        for i in range(1, L):
            h = h + A[i - 1] * (p.sigmaX(0) @ p.sigmaX(i) + p.sigmaY(0) @ p.sigmaY(i))
        self.ham = h

    def H_cs_scrambling(self, cs_coupling=None, bath_couplings=None):
        A = cs_coupling
        L = self.sites
        if A is None or len(A) != L - 1:
            A = [random() for _ in range(L - 1)]
        J = bath_couplings
        if J is None:
            J = [[0.1 * (random() - 0.5) for _ in range(i)] for i in range(L - 1)]

        self.H_central_spin(A)
        h = self.ham
        p = self.paulis
        for i in range(1, L):
            for j in range(1, i):
                h = h + J[i - 1][j - 1] * (
                    p.sigmaY(i) @ p.sigmaY(j)
                    - 0.5 * (p.sigmaX(i) @ p.sigmaX(j) + p.sigmaZ(i) @ p.sigmaZ(j))
                )
        self.ham = h
        
    def H_dipolar_chain(self, J=1):
        L = self.sites
        d = self.dim
        p = self.paulis
        h = self.ham
        if h is None:
            h = csr_matrix((d, d), dtype=complex128)
        for i in range(L - 1):
            h = h + J * (
                p.sigmaZ(i) @ p.sigmaZ(i + 1)
                - 0.5 * (p.sigmaX(i) @ p.sigmaX(i + 1) + p.sigmaY(i) @ p.sigmaY(i + 1))
            )
        self.ham = h

    def H_dipolar_sphere(self, scrambling=False):
        L = self.sites
        P1_locations = random_ball(L - 1, 3, radius=1)

    def to_super(self):
        return super_ham(self.ham)

    def to_super_alt(self):
        return super_ham_alt(self.ham)

    def to_pauli_perm(self):
        return superop2pauli_liouville(self.to_super())

    def clear(self):
        self.ham = None
