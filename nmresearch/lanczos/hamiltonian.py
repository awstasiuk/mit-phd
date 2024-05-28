import scipy as sp
from random import random, seed
from numpy.random import normal
import numpy as np

from nmresearch.lanczos.op_basis import PauliMatrix

class Hamiltonian:

    def __init__(self, sites, mat=None):
        self.sites = sites
        self.paulis = PauliMatrix(sites)
        self.ham = mat

    def H_central_spin(self, couplings=None):
        A = couplings
        if A is None or len(A) != self.sites -1:
            A = [random() for _ in range(self.sites - 1)]
            
        L = self.sites
        interaction = sp.sparse.csr_matrix((2**L,2**L),dtype=np.complex128)
        for i in range(1,L):
            interaction = interaction + A[i-1] * (self.paulis.sigmaX(0,L) @ self.paulis.sigmaX(i,L) + self.paulis.sigmaY(0,L) @ self.paulis.sigmaY(i,L))
                
        return interaction.tocsr()
    
    def H_cs_scrambling(self,cs_coupling=None,bath_couplings=None):
       
        A = cs_coupling
        if A is None or len(A) != L-1:
            A = [random() for _ in range(L-1)]
        J=bath_couplings
        if J is None:
            J= [[0.1*(random()-0.5) for _ in range(i)] for i in range(L-1)]
            
        L = self.sites
        interaction = self.H_central_spin(A)
        for i in range(1,L):
            for j in range(1,i):
                interaction = interaction + J[i-1][j-1]*(self.paulis.sigmaZ(i,L)@self.paulis.sigmaZ(j,L) - 0.5*(self.paulis.sigmaX(i,L)@self.paulis.sigmaX(j,L) + self.paulis.sigmaY(i,L)@self.paulis.sigmaY(j,L)))
                
        return interaction