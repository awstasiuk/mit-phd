import fermion.math as fm

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
            self._coef = {0:0, 1:np.zeros(2*n_fermion), 2:np.zeros((2*n_fermion,2*n_fermion))}

    def fourier_transform(self):
        r"""
        Returns a new operator which has been Fourier transformed. That is, it is
        rewritten in the fermionic momentum basis. This can often block diagonalize
        Hamiltonians with periodic boundary conditions.
        """
        n=self.n_fermion
        isqN = 1/np.sqrt(n)

        momenta = [2*pi*(k-(n-1)/2) for k in range(n)]
        f = np.array([[isqN * e ** (1j*q*i/n) for i in range(n)] for q in momenta])

        F = np.block([[F,np.zeros((2,2))],[np.zeros((2,2)),np.conjuagate(F)]])

        coef = {0:self.coef[0]}
        coef[1] = F @ self.coef[1]
        coef[2] = np.conjugate(F.T) @ self.coef[2] @ F

        return Operator(n,coef)

    def normal_order(self, highest_order=2):
        r"""
        Puts the up to the `highest_order` part of the operator into normal order,
        with a default of 2, which only inspects the quadratic part.
        """
        if highest_order > 2:
            raise ValueError("Not implemented yet, woops")
        elif highest_order < 2:
            return
        mat = self.coef[2]
        n = self.n_fermion
        for i in range(n):
            for j in range(n):
                if mat[i,j] != 0:
                    mat[j,i] += -mat[i+n,j+n]
                    self._coef[0] += int(i==j) * mat[i+n,j+n]
                    mat[i+n,j+n]=0
        self._coef[2]=mat

    @property
    def n_fermion(self):
        return self._n_fermion

    @property
    def coef(self):
        return self._coef

    @staticmethod
    def quadratic_form(A,B):
        n_fermion = len(A)

        if n_fermion != len(B):
            raise ValueError("Test")
        shA=A.shape
        shB=B.shape
        if len(shA)!=2 and len(shB)!=2:
            raise ValueError("test 2")
        if shA[0] != shA[1] or shB[0] != shB[1]:
            raise ValueError("test 3")

        q = Operator(n_fermion)
        q[2] = np.block([[A,-np.conjugate(B)],[B,-np.conjuagate(A)]])
        return q

    @staticmethod
    def global_Z(n_spin):
        Z = Operator(n_spin)
        for i in range(n_spin):
            Z._coef[2][i,i]=-1
            Z._coef[2][i+n_spin,i+n_spin]=1
        return Z

    @staticmethod
    def local_Z(index, n_spin):
        Zi = Operator(n_spin)
        Zi._coef[2][index-1,index-1]=-1
        Zi._coef[2][index-1+n_spin,index-1+n_spin]=1
        return Zi

    @staticmethod
    def double_quantum(n_spin,B=0,J=1,periodic=False):
        r"""
        Generate the nearest neighbor double quantum Hamiltonian quadratic form,
        `H=B*Sz + J*Sum(XX+YY)`

        `B` can be specified as a `double` or an `iterable` of length `n_spin` to
        generate a Hamiltonian with disorder. By default, B=0, J=1, and the boundary
        conditions are set to be open. Periodic boundary conditions can be imposed
        by setting `periodic` to `True`.
        """
        if hasattr(B, '__iter__'):
            A = np.diag(B)
        else:
            A = -B*np.identity(n_spin)

        C = J*(np.diag(np.ones(n_spin-1),1) - np.diag(np.ones(n_spin-1),-1))
        if periodic:
            if n_spin % 2 == 0:
                C[1,n] = J
                C[n,1] = -J
            else:
                C[1,n] = -J
                C[n,1] = J
        return Operator.quadratic_form(A,C)
