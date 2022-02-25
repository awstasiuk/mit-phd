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
        isqN = 1 / np.sqrt(n)

        momenta = [(k-(n-1)/2) for k in range(n)]
        f = np.array([[isqN * e ** (2j*pi*q*(i+1)/n) for q in momenta] for i in range(n)])

        F = np.block([[f,np.zeros((n,n))],[np.zeros((n,n)),np.conjugate(f)]])

        coef = {0:self.coef[0]}
        coef[1] = np.conjugate(F @ self.coef[1])
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
        if not np.allclose(self.coef[1],0):
            raise ValueError("Not a quadratic fermionic operator")
        self.normal_order()
        n=self.n_fermion

        # prepare the right matrix to diagonalize
        alpha = self.coef[2][0:n,0:n]
        beta = -2*np.conj(self.coef[2][0:n,n:2*n])
        mat = (alpha-beta) @ (alpha + beta)

        # diagonalize and extract the correctly ordered operators
        sqr_eig, phi = np.linalg.eigh(mat)
        psi = [(1/np.sqrt(sqr_eig[i]))*(alpha-beta) @ phi[:,i] for i in range(n)]
        psi = np.array(psi)
        phi = phi.T

        # form the diagonalizing matrix, and the resulting diagonal operator.
        G = 0.5*(phi+psi)
        H = 0.5*(phi-psi)
        T = np.block([[G, H], [np.conj(H), np.conj(G)]])
        return Operator.disorder_Z(n,-0.5*np.sqrt(sqr_eig)), T


    def trace(self):
        r"""
        Computes the trace divided by 2^n_fermion, so as to avoid things needlessly
        blowing up.
        """
        tr = self.coef[0]
        tr += sum(self.coef[2][i,i] for i in range(2*self.n_fermion))
        return tr

    def commutator(self,other):
        r"""
        computes the commutator of two operators, returns the resulting operator,
        C = [self, other].
        """
        return

    def anti_commutator(self,other):
        r"""
        computes the anit-commutator of two operators, returns the resulting operator,
        C = {self, other}.
        """
        return

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
        q._coef[2] = np.block([[A,-np.conjugate(B)],[B,-np.conjugate(A)]])
        return q

    @staticmethod
    def creation_op(index, n_spin):
        a = Operator(n_spin)
        a._coef[1][index+n_spin] = 1
        return a

    @staticmethod
    def annihilation_op(index, n_spin):
        adag = Operator(n_spin)
        adag._coef[1][index] = 1
        return adag

    @staticmethod
    def number_op(index, n_spin):
        N = Operator(n_spin)
        N._coef[2][index,index] = 1
        return N

    @staticmethod
    def global_Z(n_spin):
        Z = Operator(n_spin)
        for i in range(n_spin):
            Z._coef[2][i,i]=-1
            Z._coef[2][i+n_spin,i+n_spin]=1
        return Z

    @staticmethod
    def disorder_Z(n_spin, field):
        if len(field) is not n_spin:
            raise ValueError("vector of disorder must equal number of spins")
        dZ = Operator(n_spin)
        for idx,B in enumerate(field):
            dZ._coef[2][idx,idx] = -B
            dZ._coef[2][idx+n_spin,idx+n_spin] = B
        return dZ

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
                C[0,n_spin-1] = J
                C[n_spin-1,0] = -J
            else:
                C[0,n_spin-1] = -J
                C[n_spin-1,0] = J
        return Operator.quadratic_form(A,C)

    def __add__(self, other):
        if isinstance(other,Operator) and self.n_fermion == other.n_fermion:
            add = Operator(self.n_fermion)
            for i in range(3):
                add._coef[i] = self.coef[i] + other.coef[i]
            return add
        else:
            raise ValueError("Operator addition must be between ops with same number of fermions")

    def __mult__(self, other):
        if isinstance(other,Operator):
            raise ValueError("non-scalar multiplication not yet implemented")
        elif type(other) in (int, float, complex):
            mult = Operator(self.n_fermion)
            for i in range(3):
                mult._coef[i] = other * self.coef[i]
            return mult
        else:
            raise ValueError("type not recognized for multiplication")
