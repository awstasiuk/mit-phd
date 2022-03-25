from fermion.math import Math as fm

import numpy as np
from scipy import linalg as la
from math import e, pi
import itertools
import tensorflow as tf


class Operator:
    r"""
    A class desciribing a quadratic multi-body fermionic operator
    """

    def __init__(self, n_fermion, coef=None):
        r"""
        initialize the object
        """
        self._n_fermion = n_fermion
        if coef is not None:
            self._coef = coef
        else:
            self._coef = {
                0: 0,
                2: np.zeros((2 * n_fermion, 2 * n_fermion)),
            }
        self._components = list(self._coef.keys())
        self._order = max(self._components)

    def fourier_transform(self):
        r"""
        Returns a new operator which has been Fourier transformed. That is, it is
        rewritten in the fermionic momentum basis. This can often block diagonalize
        Hamiltonians with periodic boundary conditions.
        """
        n = self.n_fermion
        isqN = 1 / np.sqrt(n)

        momenta = [(k - (n - 1) / 2) for k in range(n)]
        f = np.array(
            [
                [isqN * e ** (2j * pi * q * (i + 1) / n) for q in momenta]
                for i in range(n)
            ]
        )

        F = np.block([[f, np.zeros((n, n))], [np.zeros((n, n)), np.conjugate(f)]])
        coef = {}
        for k in self.components:
            if k == 0:
                coef[0] = self.coef[0]
            elif k == 1:
                coef[1] = F @ self.coef[1]
            else:
                coef[k] = fm.tensor_change_of_basis(self.coef[k], F)
        return Operator(n, coef)

    def normal_order(self):
        r"""
        Puts the up to the `highest_order` part of the operator into normal order,
        with a default of 2, which only inspects the quadratic part.
        """
        if 0 not in self.components:
            self.set_component(0)
        if 2 not in self.components:
            self.set_component(2)

        if 2 in self.components:
            mat = self.coef[2]
            n = self.n_fermion
            for i in range(n):
                for j in range(n):
                    if mat[i, j + n] != 0:
                        mat[j + n, i] += -mat[i, j + n]
                        self._coef[0] += int(i == j) * mat[i, j + n]
                        mat[i, j + n] = 0
            self.set_component(2, mat)

        if 4 in self.components:
            mat = self.coef[4]
            n = self.n_fermion
            new = np.zeros(mat.shape, dtype=np.complex)
            for idx, val in np.ndenumerate(mat):
                if val != 0 and not Operator._is_zero(idx):
                    # normal order the 4 body term, keeping track of swaps
                    old = list(idx)
                    phase = 1
                    new_idx = ()
                    for i in range(4):
                        m_idx = old.index(max(old))
                        new_idx += (old.pop(m_idx),)
                        phase *= (-1) ** m_idx
                    if not Operator._is_zero(new_idx):
                        new[new_idx] += phase * val

                    # compute all the single contractions
                    for i in range(3):
                        if idx[i] < n:
                            for j in range(i + 1, 4):
                                if idx[j] >= n and idx[i] == (idx[j] - n):
                                    two_body_idx = list(idx)
                                    two_body_idx.pop(j)
                                    two_body_idx.pop(i)
                                    phase = (-1) ** (i - j + 1)
                                    if two_body_idx[0] < two_body_idx[1]:
                                        two_body_idx.reverse()
                                        phase *= -1
                                    if not Operator._is_zero(two_body_idx):
                                        self._coef[2][tuple(two_body_idx)] += (
                                            val * phase
                                        )

                    # compute any double contractions
                    if fm.fermion_weight(idx, n) == 0:
                        if idx[0] < n and idx[1] < n:
                            self._coef[0] += (
                                -1
                                * val
                                * (idx[0] == (idx[2] - n))
                                * (idx[1] == (idx[3] - n))
                            )
                            self._coef[0] += (
                                val
                                * (idx[0] == (idx[3] - n))
                                * (idx[1] == (idx[2] - n))
                            )
                        elif idx[0] < n and idx[2] < n:
                            self._coef[0] += (
                                val
                                * (idx[0] == (idx[1] - n))
                                * (idx[2] == (idx[3] - n))
                            )
            self.set_component(4, new)
        return self

    @staticmethod
    def _is_zero(idx):
        for i in range(1, len(idx)):
            if idx[i - 1] == idx[i]:
                return True
        return False

    def jordan_wigner(self):
        r"""
        Returns a tuple of the diagonalized quadratic fermionic operator and the conjugate
        transpose of the diagonalizing Jordan--Wigner matrix, which is block diagonal of the form
        T = [[G, H], [H.conj, G.conj]], and the blocks satisfy

        G*G.T + H*H.T = Identity
        G*H.T + H*G.T = 0

        WARNING: Puts the operator into normal ordering
        """
        if not self.is_quadratic:
            raise ValueError("Not a quadratic fermionic operator")
        n = self.n_fermion

        # prepare the right matrix to diagonalize
        alpha = 2 * self.coef[2][n : 2 * n, 0:n]
        beta = 2 * self.coef[2][0:n, 0:n]
        mat = (alpha - beta) @ (alpha + beta)

        # diagonalize and extract the correctly ordered operators
        sqr_eig, phi = la.eigh(mat)
        psi = [(1 / np.sqrt(sqr_eig[i])) * (alpha - beta) @ phi[:, i] for i in range(n)]
        psi = np.array(psi)
        phi = phi.T

        # form the diagonalizing matrix, and the resulting diagonal operator.
        G = 0.5 * (phi + psi)
        H = 0.5 * (phi - psi)
        T = np.block([[G, H], [np.conj(H), np.conj(G)]])
        return Operator.disorder_Z(n, -0.5 * np.sqrt(sqr_eig)), T

    def trace(self):
        r"""
        Computes the trace divided by 2^n_fermion, so as to avoid things needlessly
        blowing up. tring to be fast here.
        """
        tr = 0
        if 0 in self.components:
            tr += self.coef[0]

        if 2 in self.components:
            tr += 0.5 * (
                np.sum(np.diagonal(self.coef[2], self.n_fermion))
                + np.sum(np.diagonal(self.coef[2], -self.n_fermion))
            )

        if 4 in self.components:
            idx_list, weights = fm.tw_four_body(self.n_fermion)
            tr += sum(w * self.coef[4][idx] for idx, w in zip(idx_list, weights))

        return tr

    def commutator(self, other, use_speedup=True):
        r"""
        computes the commutator of two operators, returns the resulting operator,
        C = [self, other].
        """
        n = self.n_fermion
        if n != other.n_fermion:
            raise ValueError("invalid shapes")

        if use_speedup and self.is_quadratic and other.is_quadratic:
            self.normal_order()
            other.normal_order()
            comm = np.zeros((2 * self.n_fermion, 2 * self.n_fermion), dtype=np.complex)

            aa1 = tf.constant(self.coef[2][0:n, 0:n], dtype=tf.complex128)
            ca1 = tf.constant(self.coef[2][n : 2 * n, 0:n], dtype=tf.complex128)
            cc1 = tf.constant(self.coef[2][n : 2 * n, n : 2 * n], dtype=tf.complex128)

            aa2 = tf.constant(other.coef[2][0:n, 0:n], dtype=tf.complex128)
            ca2 = tf.constant(other.coef[2][n : 2 * n, 0:n], dtype=tf.complex128)
            cc2 = tf.constant(other.coef[2][n : 2 * n, n : 2 * n], dtype=tf.complex128)

            delta = tf.eye(n, dtype=tf.complex128)

            aa3 = tf.einsum("ij,kl,li->jk", ca1, aa2, delta) - tf.einsum(
                "ij,kl,ki->jl", ca1, aa2, delta
            )
            aa3 += -tf.einsum("kl,ij,li->kj", aa1, ca2, delta) + tf.einsum(
                "kl,ij,ki->lj", aa1, ca2, delta
            )

            cc3 = tf.einsum("ij,kl,li->jk", cc1, ca2, delta) - tf.einsum(
                "ij,kl,lj->ik", cc1, ca2, delta
            )
            cc3 += -tf.einsum("kl,ij,li->kj", ca1, cc2, delta) + tf.einsum(
                "kl,ij,lj->jl", ca1, cc2, delta
            )

            ca3 = (
                -tf.einsum("kl,ij,li", aa1, cc2, delta)
                + tf.einsum("kl,ij,ki", aa1, cc2, delta)
                + tf.einsum("kl,ij,lj", aa1, cc2, delta)
                - tf.einsum("kl,ij,kj", aa1, cc2, delta)
            )
            ca3 += (
                tf.einsum("ij,kl,li", cc1, aa2, delta)
                - tf.einsum("ij,kl,ki", cc1, aa2, delta)
                - tf.einsum("ij,kl,lj", cc1, aa2, delta)
                + tf.einsum("ij,kl,kj", cc1, aa2, delta)
            )
            ca3 += tf.einsum("ij,kl,jk", ca1, ca2, delta) - tf.einsum(
                "ij,kl,li", ca1, ca2, delta
            )

            comm[0:n, 0:n] = aa3.numpy()
            comm[n : 2 * n, 0:n] = ca3.numpy()
            comm[n : 2 * n, n : 2 * n] = cc3.numpy()

            return Operator(self.n_fermion, {2: comm})
        return self * other - other * self

    def anti_commutator(self, other):
        r"""
        computes the anit-commutator of two operators, returns the resulting operator,
        C = {self, other}.
        """
        return self * other + other * self

    @property
    def is_quadratic(self):
        r"""
        Checks if there are non-zero components outside of the identity or quadratic form
        terms.
        """
        for order, mat in self.coef.items():
            if order not in (0, 2) and not np.allclose(mat, 0):
                return False
        return True

    def compactify(self, tol=10**-10):
        ords = self.components
        for n in ords:
            temp = fm.chop(self.coef[n], delta=tol)
            if np.allclose(temp, 0):
                self.remove_component(n)
            else:
                self.set_component(n, temp)
        return self

    def set_component(self, order, tensor=None):
        r"""
        If this operator does not have a component of this order, it adds the `order`--dimensional
        array needed to store it, and updates the highest order stored by this operator, if
        necessary. By specifying `tensor` as a properly sized and shapped array, the stored
        tensor can be specified, and in the case of an existing order, will overwrite the stored
        data.
        """
        if tensor is None:
            self._coef[order] = np.zeros(
                [2 * self.n_fermion for i in range(order)], dtype=np.complex
            )
        elif len(tensor.shape) == order and all(
            [tensor.shape[i] == 2 * self.n_fermion for i in range(order)]
        ):
            self._coef[order] = tensor
        else:
            raise ValueError("Shape of tensor is invalid for desired order")

        if order not in self.components:
            self._components.append(order)

        if order > self.order:
            self._order = order

    def remove_component(self, order):
        if order in self.components:
            del self.coef[order]
            self._components.remove(order)
            self.update_order()
        return self

    def update_order(self):
        if len(self.components) == 0:
            self.set_component(0, 0)
            self._order = 0
        else:
            self._order = max(self.components)

    @property
    def n_fermion(self):
        return self._n_fermion

    @property
    def coef(self):
        return self._coef

    @property
    def order(self):
        return self._order

    @property
    def components(self):
        return self._components

    @staticmethod
    def quadratic_form(A, B):
        n_fermion = len(A)

        if n_fermion != len(B):
            raise ValueError("The matrices must be of the same length")
        shA = A.shape
        shB = B.shape
        if len(shA) != 2 and len(shB) != 2:
            raise ValueError("the matrices must be second order (rank 2)")
        if shA[0] != shA[1] or shB[0] != shB[1]:
            raise ValueError("the matrices must be square")

        q = Operator(n_fermion, {2: np.block([[B, -np.conj(A)], [A, -np.conj(B)]])})
        return q

    @staticmethod
    def creation_op(index, n_spin):
        temp = np.zeros(2 * n_spin)
        temp[index + n_spin] = 1
        return Operator(n_spin, {1: temp})

    @staticmethod
    def annihilation_op(index, n_spin):
        temp = np.zeros(2 * n_spin)
        temp[index] = 1
        return Operator(n_spin, {1: temp})

    @staticmethod
    def number_op(index, n_spin):
        temp = np.zeros((2 * n_spin, 2 * n_spin))
        temp[index + n_spin, index] = 1
        return Operator(n_spin, {2: temp})

    @staticmethod
    def global_Z(n_spin):
        return Operator.disorder_Z(n_spin, np.ones(n_spin))

    @staticmethod
    def disorder_Z(n_spin, field):
        if len(field) is not n_spin:
            raise ValueError("vector of disorder must equal number of spins")
        temp = np.zeros((2 * n_spin, 2 * n_spin))
        for idx, B in enumerate(field):
            temp[idx, idx + n_spin] = B
            temp[idx + n_spin, idx] = -B
        return Operator(n_spin, {2: temp})

    @staticmethod
    def local_Z(index, n_spin):
        Zi = Operator(n_spin)
        temp = np.zeros((2 * n_spin, 2 * n_spin))
        temp[index, index + n_spin] = 1
        temp[index + n_spin, index] = -1
        return Operator(n_spin, {2: temp})

    @staticmethod
    def double_quantum(n_spin, B=0, J=1, periodic=False):
        r"""
        Generate the nearest neighbor double quantum Hamiltonian quadratic form,
        `H=B*Sz + J*Sum(XX+YY)`

        `B` can be specified as a `double` or an `iterable` of length `n_spin` to
        generate a Hamiltonian with disorder. By default, B=0, J=1, and the boundary
        conditions are set to be open. Periodic boundary conditions can be imposed
        by setting `periodic` to `True`.
        """
        if hasattr(B, "__iter__"):
            A = np.diag(B)
        else:
            A = -B * np.identity(n_spin)

        C = J * (np.diag(np.ones(n_spin - 1), 1) - np.diag(np.ones(n_spin - 1), -1))
        if periodic:
            if n_spin % 2 == 0:
                C[0, n_spin - 1] = J
                C[n_spin - 1, 0] = -J
            else:
                C[0, n_spin - 1] = -J
                C[n_spin - 1, 0] = J
        return Operator.quadratic_form(A, C)

    @staticmethod
    def flipflop(n_spin, B=0, J=1, periodic=False):
        r"""
        Generate the nearest neighbor flip-flop Hamiltonian quadratic form,
        `H=B*Sz + J*Sum(XX-YY)`

        `B` can be specified as a `double` or an `iterable` of length `n_spin` to
        generate a Hamiltonian with disorder. By default, B=0, J=1, and the boundary
        conditions are set to be open. Periodic boundary conditions can be imposed
        by setting `periodic` to `True`.
        """
        if hasattr(B, "__iter__"):
            A = np.diag(B)
        else:
            A = -B * np.identity(n_spin)
        A += J * (np.diag(np.ones(n_spin - 1), 1) - np.diag(np.ones(n_spin - 1), -1))
        if periodic:
            if n_spin % 2 == 0:
                A[0, n_spin - 1] = J
                A[n_spin - 1, 0] = -J
            else:
                A[0, n_spin - 1] = -J
                A[n_spin - 1, 0] = J

        return Operator.quadratic_form(A, np.zeros((n_spin, n_spin)))

    @staticmethod
    def generalXY(n_spin, u, v, B=0, J=1, perdioic=False):
        r"""
        Generate the nearest neighbor XY hamiltonian quadratic form, with
        `H=B*Sz + J*Sum(u*XX + v*YY)`, where u and v are arbitrary real numbers.

        `B` can be specified as a `double` or an `iterable` of length `n_spin` to
        generate a Hamiltonian with disorder. By default, B=0, J=1, and the boundary
        conditions are set to be open. Periodic boundary conditions can be imposed
        by setting `periodic` to `True`.
        """
        flip = (u + v) / 2
        dq = (u - v) / 2
        return Operator.double_quantum(
            n_spin, B=B, J=J * dq, periodic=periodic
        ) + Operator.flipflop(n_spin, B=B, J=J * flip, periodic=periodic)

    def __add__(self, other):
        if isinstance(other, Operator) and self.n_fermion == other.n_fermion:
            comps = list(set(self.components).union(other.components))
            coef = {}
            for n in comps:
                if n in self.components and n not in other.components:
                    coef[n] = self.coef[n]
                elif n not in self.components and n in other.components:
                    coef[n] = other.coef[n]
                else:
                    coef[n] = self.coef[n] + other.coef[n]
            return Operator(self.n_fermion, coef)
        else:
            raise ValueError(
                "Operator addition must be between ops with same number of fermions"
            )

    def __sub__(self, other):
        if isinstance(other, Operator) and self.n_fermion == other.n_fermion:
            comps = list(set(self.components).union(other.components))
            coef = {}
            for n in comps:
                if n in self.components and n not in other.components:
                    coef[n] = self.coef[n]
                elif n not in self.components and n in other.components:
                    coef[n] = -1 * other.coef[n]
                else:
                    coef[n] = self.coef[n] - other.coef[n]
            return Operator(self.n_fermion, coef)
        else:
            raise ValueError(
                "Operator addition must be between ops with same number of fermions"
            )

    def __mul__(self, other):
        coef = {}
        if isinstance(other, Operator) and self.n_fermion == other.n_fermion:
            for rS in self.components:
                for rO in other.components:
                    if rS + rO in coef.keys():
                        coef[rS + rO] += fm.tensor_product(
                            self.coef[rS], other.coef[rO]
                        )
                    else:
                        coef[rS + rO] = fm.tensor_product(self.coef[rS], other.coef[rO])
            return Operator(self.n_fermion, coef)
        elif type(other) in (int, float, complex):
            for i in self.components:
                coef[i] = other * self.coef[i]
            return Operator(self.n_fermion, coef)
        else:
            raise ValueError("type not recognized for multiplication")

    def __pow__(self, other):
        if other != 2:
            raise ValueError
        return self * self

    def __eq__(self, other):
        if isinstance(other, Operator) and self.n_fermion == other.n_fermion:
            if self.components == other.components:
                for r in self.components:
                    if not np.allclose(self.coef[r], other.coef[r]):
                        return False
                return True
        return False

    def _ferm_string(self, idx):
        n = self.n_fermion
        op_list = []
        for i in idx:
            if i < n:
                op_list.append("a" + str(i))
            else:
                op_list.append("c" + str(i % n))
        return "".join(op_list)

    def __str__(self):
        op = []
        for comp, mat in self.coef.items():
            if comp == 0 and mat != 0:
                op.append(str(mat) + "*I")
            else:
                for idx, val in np.ndenumerate(mat):
                    if val != 0:
                        op.append(str(val) + "*" + self._ferm_string(idx))
        return "0" if len(op) == 0 else " + ".join(op)
