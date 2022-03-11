from fermion.math import Math as fm

import numpy as np
from math import e, pi


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
        if 0 in self.components:
            coef[0] = self.coef[0]
        if 1 in self.components:
            coef[1] = np.conjugate(F @ self.coef[1])
        if 2 in self.components:
            coef[2] = np.conjugate(F.T) @ self.coef[2] @ F
        return Operator(n, coef)

    def normal_order(self):
        r"""
        Puts the up to the `highest_order` part of the operator into normal order,
        with a default of 2, which only inspects the quadratic part.
        """
        if 0 not in self.components:
            self.set_component(0)
        if 2 in self.components:
            mat = self.coef[2]
            n = self.n_fermion
            for i in range(n):
                for j in range(n):
                    if mat[i, j] != 0:
                        mat[j, i] += -mat[i + n, j + n]
                        self._coef[0] += int(i == j) * mat[i + n, j + n]
                        mat[i + n, j + n] = 0
            self.set_component(2, mat)

    def jordan_wigner(self):
        r"""
        Returns a tuple of the diagonalized quadratic fermionic operator and the diagonalizing
        Jordan--Wigner matrix, which is block diagonal of the form
        T = [[G, H], [H.conj, G.conj]], where T*M*T.dag=D is diagonal, and the blocks
        satisfy
        G*G.T + H*H.T = Identity
        G*H.T + H*G.T = 0

        WARNING: Puts the operator into normal ordering
        """
        if not self.is_quadratic:
            raise ValueError("Not a quadratic fermionic operator")
        self.normal_order()
        n = self.n_fermion

        # prepare the right matrix to diagonalize
        alpha = self.coef[2][0:n, 0:n]
        beta = -2 * np.conj(self.coef[2][0:n, n : 2 * n])
        mat = (alpha - beta) @ (alpha + beta)

        # diagonalize and extract the correctly ordered operators
        sqr_eig, phi = np.linalg.eigh(mat)
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
        blowing up.
        """
        tr = self.coef[0]
        tr += sum(self.coef[2][i, i] / 2 for i in range(2 * self.n_fermion))
        return tr

    def commutator(self, other):
        r"""
        computes the commutator of two operators, returns the resulting operator,
        C = [self, other].
        """
        return

    def anti_commutator(self, other):
        r"""
        computes the anit-commutator of two operators, returns the resulting operator,
        C = {self, other}.
        """
        return

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

    def set_component(self, order, tensor=None):
        r"""
        If this operator does not have a component of this order, it adds the `order`--dimensional
        array needed to store it, and updates the highest order stored by this operator, if
        necessary. By specifying `tensor` as a properly sized and shapped array, the stored
        tensor can be specified, and in the case of an existing order, will overwrite the stored
        data.
        """
        if tensor is None:
            self._coef[order] = np.zeros([2 * self.n_fermion for i in range(order)])
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

    def update_order(self):
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

        q = Operator(
            n_fermion, {2: np.block([[A, -np.conjugate(B)], [B, -np.conjugate(A)]])}
        )
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
        temp[index, index] = 1
        return Operator(n_spin, {2: temp})

    @staticmethod
    def global_Z(n_spin):
        temp = np.zeros((2 * n_spin, 2 * n_spin))
        for i in range(n_spin):
            temp[i, i] = -1
            temp[i + n_spin, i + n_spin] = 1
        return Operator(n_spin, {2: temp})

    @staticmethod
    def disorder_Z(n_spin, field):
        if len(field) is not n_spin:
            raise ValueError("vector of disorder must equal number of spins")
        temp = np.zeros((2 * n_spin, 2 * n_spin))
        for idx, B in enumerate(field):
            temp[idx, idx] = -B
            temp[idx + n_spin, idx + n_spin] = B
        return Operator(n_spin, {2: temp})

    @staticmethod
    def local_Z(index, n_spin):
        Zi = Operator(n_spin)
        temp = np.zeros((2 * n_spin, 2 * n_spin))
        temp[index - 1, index - 1] = -1
        temp[index - 1 + n_spin, index - 1 + n_spin] = 1
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

    def __mult__(self, other):
        if isinstance(other, Operator):
            raise ValueError("non-scalar multiplication not yet implemented")
        elif type(other) in (int, float, complex):
            mult = Operator(self.n_fermion)
            for i in self.components:
                mult._coef[i] = other * self.coef[i]
            return mult
        else:
            raise ValueError("type not recognized for multiplication")
