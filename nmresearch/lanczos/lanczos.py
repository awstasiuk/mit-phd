import numpy as np
import scipy as sp
from timeit import default_timer as timer

from nmresearch.lanczos.utils import basis_vec

class Lanczos:

    def __init__(self, liouville, op):
        self.liouv = liouville
        self.op = op
        self.e0 = None
        self.L = None
        self.krylov_basis = None
        self.lanczos_coef = None

    def compute_lanczos_fast(self,max_iter=50,tol=1e-10):
        r"""
        Computes lanczos method up to `max_iter` steps or until the produced vector is null to within
        tolerance `tol`. This method uses Gram-Schmidt to orthogonalize, and as such is numerically unstable
        after a handful of iterations. The produced lanczos coefficients are still a good proxy for the
        liovillian's action at time not too long, or operators which scramble too much. All orthogonalized
        krylov vectors are stored, check their inner products for loss of orthogonality if interested in 
        checking on the numerical stability of the outcome.
        """
        start = timer()
        end=0
        O0 = self.op
        A1 = self.liouv @ O0
        b1 = np.sqrt( A1.conj() @ A1)
        O1 = (1/b1)*A1
        self.krylov_basis = np.array([O0, O1])
        self.lanczos_coef = np.array([b1])

        def lanczos_iteration(liouv, On2, On1, bn1, tol=1e-10):
            An = liouv @ On1 - bn1 * On2
            bn = np.sqrt( An.conj() @ An )
            if np.round(bn,16) < tol:
                return None, 0
            return (1/bn) * An, bn

        def generate_next():
            On,bn = lanczos_iteration(self.liouv, self.krylov_basis[-2], self.krylov_basis[-1], self.lanczos_coef[-1])
            if On is not None:
                self.krylov_basis.append(On)
                self.lanczos_coef.append(bn)
                return True
            else:
                end = timer()
                return False
        counter = 1
        while counter < max_iter and generate_next():
            counter+=1

        print("Computation took " + str(end-start) + " sec, after " + str(counter) + " iterations.")
        
    def compute_lanczos_FRO(self, max_iter=50,tol=1e-10):
        r"""
        Computes lanczos method up to `max_iter` steps or until the produced vector is null to within
        tolerance `tol`. This method uses modified Gram-Schmidt to orthogonalize. This method is stable,
        but it can be slow and memory intensive for large system sizes, since all krylov vectors must be
        stored and we repeatedly re-orthogonalize
        """
        start = timer()
        end = 0
        O0 = self.op
        A1 = self.Liouv @ O0
        b1 = np.sqrt( A1.conj() @ A1)
        O1 = (1/b1)*A1
        self.krylov_basis = np.array([O0, O1])
        self.lanczos_coef = np.array([b1])

        for j in range(2, max_iter, 1):
            Aj = self.Liouv @ self.krylov_basis[-1] - self.lanczos_coef[-1] * self.krylov_basis[-2]
            for i in range(j):
                Aj = Aj - (self.krylov_basis[i].conj() @ Aj) * self.krylov_basis[i]
            bj = np.round(np.sqrt( Aj.conj() @ Aj ), 16)
            if bj < tol:
                print("Lanczos Algorithm terminated at a 0-vector")
                break
        
            Oj = (1/bj) * Aj
            self.krylov_basis.append(Oj)
            self.lanczos_coef.append(bj)
        end = timer()
        print("Computation took " + str(end-start) + " sec")

    def auto_correlation(self, times):
        if self.lanczos_coef is None:
            print("Lanczos coefficients must be computed!")
            return
        if self.L is None:
            self.L = sp.sparse.diags([self.lanczos_coef,self.lanczos_coef], [1,-1])
            self.e0 = basis_vec(self.L.shape[0],0)
        return np.array([self.e0 @ (sp.sparse.linalg.expm_multiply(1j*self.L*t,self.e0)) for t in times])
    
    def auto_correlation_ED(self, times):
        r"""
        Uses Scipy sparse matrix techniques to compute the autocorrelation. Recomputes the exponential
        for each timestep desired, minimizing integration error, but increasing computation time.
        """
        return np.array([self.op @ sp.sparse.linalg.expm_multiply(-1j*self.liouv*t, self.op) for t in times])
    
    def auto_correlation_ED_fast(self, times):
        r"""
        Uses Scipy sparse matrix exponential matrix solver to computer the autocorrelation.
        This method appears to only compute a single propegator, and so its accuracy is dependent on
        the timestep, for some reason.
        """
        return sp.sparse.linalg.expm_multiply(-1j*self.liouv, self.op, start=times[0], stop=times[-1], num=len(times), endpoint=True) @ self.op