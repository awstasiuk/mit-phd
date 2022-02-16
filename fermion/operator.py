import numpy as np

class Operator:
    r"""
    A class desciribing a multi-body fermionic operator
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
        # f_k = sum_j a_j exp(i*k*j)
        return
        
    @property
    def n_fermion(self):
        return self._n_fermion

    @property
    def coef(self):
        return self._coef

    @staticmethod
    def quadratic_form(A,B):
        if len(A) != len(B):
            raise ValueError("Test")
        # TODO: check if A and B are square matrices
        n_fermion = len(A)
        q = Operator(n_fermion)
        # TODO: set the elements of q in the block matrix.
        return q

    @staticmethod
    def global_Z(n_spin):
        Z = Operator(n_spin)
        # TODO: set elements
        return Z

    @staticmethod
    def local_Z(index, n_spin)
        Zi = Operator(n_spin)
        # TODO: set elements
        return Zi
