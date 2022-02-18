import numpy as np
import fermion.math as fm

class Operator:
    r"""
    A class desciribing a quadratic multi-body fermionic operator
    """

    # __slots__ = ("ticker", "data", "close")

    def __init__(self, n_fermion):
        r"""
        initialize the object
        """
        self._n_fermion = n_fermion
        self._coef = {0:0, 1:np.zeros(2*n_fermion), 2:np.zeros((2*n_fermion,2*n_fermion))}

    def fourier_transform(self):
        # TODO: Mutate the coefficient matrices for all orders to use momentum basis
        # f_k = (1/sqrt(N)) * sum_j a_j exp(i*k*q[j])
        return

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
    def double_quantum(B,J,n_spin):
        # generate the NN DQ Hamiltonian
        return
