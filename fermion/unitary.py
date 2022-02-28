import fermion.math as fm
import fermion.operator as op

import numpy as np
from math import e, pi


class Unitary:
    r"""
    A class desciribing unitary time evolution of a fermionic operator. Owns a quadratic
    `Operator` serving as the system Hamiltonian.
    """

    def __init__(self, ham, dt, tmax):
        r"""
        initialize the object, by preparing a time mesh and diagonalizing the hamiltonian
        """
        if not ham.is_quadratic():
            raise ValueError("Invalid Hamiltonian, must be quadratic")
        self._ham = ham

        self._dt = dt
        self._tmax = tmax
        self._n_steps = ceil(tmax / dt)
        self._t = [i * dt for i in range(self._n_steps)]

        diag, T = self.ham.jordan_wigner()
        self._eigen = diag.coef[2].diagonal(0)[0 : ham.n_fermion]
        self._G = T.coef[2][0 : ham.n_fermion, 0 : ham.n_fermion]
        self._H = T.coef[2][0 : ham.n_fermion, ham.n_fermion : 2 * ham.n_fermion]

    def U1(self, t):
        r"""
        the N*N operator representing time evolution of the annihilation operators in the
        original fermionic basis
        """
        return (
            fm.adj(self.G) @ fm.exp_diag(self.eigen, 2 * t) @ G
            + fm.adj(self.H) @ fm.exp_diag(self.eigen, -2 * t) @ self.H
        )

    def U2(self, t):
        r"""
        the N*N operator representing time evolution of the creations operators in the
        original fermionic basis.
        """
        return (
            fm.adj(self.G) @ fm.exp_diag(self.eigen, 2 * t) @ self.H
            + fm.adj(self.H) @ fm.exp_diag(self.eigen, -2 * t) @ self.G
        )

    def U(self, t):
        r"""
        the 2N*2N operator representing time evolution in the original fermionic basis
        """
        u1 = self.U1(t)
        u2 = self.U2(t)
        # TODO: Check if this is right!
        return np.block([[u1, u2], [np.conjugate(u2), np.conjugate(u1)]])

    def local_zz(self, idx1, idx2):
        r"""
        Computes the infinite temperature two point correlator for local Z observables at
        indices `idx1` and `idx2`, which can be written as the expectation value of
        `<Z_idx1(t) Z_idx2>`.

        `idx1` and `idx2` must be integers in the range `[0, n_fermion]`.

        returns a list of points evaluated on this instance's time mesh.
        """
        return [
            np.abs(self.U1(t)[idx1, idx2]) ** 2 - np.abs(self.U2(t)[idx1, idx2]) ** 2
            for t in self.t
        ]

    def global_zz(self):
        r"""
        Computes the infinite temperature two point correlator for local Z observables at
        indices `idx1` and `idx2`, which can be written as the expectation value of
        `<Z(t) Z>`.

        returns a list of points evaluated on this instance's time mesh.
        """
        return [
            np.linalg.norm(self.U1(t)) ** 2 - np.linalg.norm(self.U2(t)) ** 2
            for t in self.t
        ]

    @property
    def hamiltonian(self):
        return self._ham

    @property
    def dt(self):
        return self._dt

    @property
    def tmax(self):
        return self._tmax

    @property
    def t(self):
        return self._t

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def eigen(self):
        r"""
        The eigenvalues of the quadratic form Hamiltonian, which represent fermion energies. If
        the model is non-interacting (quadratic), the spectrum of the Hamiltonian corresponds
        to the sum of all possible subsets of this list.
        """
        return self._eigen

    @property
    def G(self):
        r"""
        The upper left block of the change of basis matrix of the corresponding jordan-wigner
        transformation of this hamiltonian
        """
        return self._G

    @property
    def H(self):
        r"""
        The upper right block of the change of basis matrix of the corresponding jordan-wigner
        transformation of this hamiltonian
        """
        return self._H
