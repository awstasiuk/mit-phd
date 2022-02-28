import numpy as np


class Math:
    r"""
    A class which does math stuff
    """

    @staticmethod
    def kron_delta(i, j):
        if int(i) != i or int(j) != j:
            raise ValueError("arguments should be integers")
        return int(i == j)

    @staticmethod
    def chop(expr, delta=10**-10):
        if hasattr(expr, "__iter__"):
            return [Math.chop(x) for x in expr]
        else:
            return 0 if -delta <= abs(expr) <= delta else expr

    @staticmethod
    def adj(mat):
        return np.conjugate(mat.T)

    @staticmethod
    def exp_diag(vec, t):
        diag = np.exp(-1j * vec * t)
        return np.diag(diag)
