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
            return np.array([Math.chop(x) for x in expr])
        else:
            return 0 if -delta <= abs(expr) <= delta else expr

    @staticmethod
    def adj(mat):
        return np.conjugate(mat.T)

    @staticmethod
    def exp_diag(vec, t):
        diag = np.exp(-1j * vec * t)
        return np.diag(diag)

    @staticmethod
    def tensor_change_of_basis(tensor, matrix):
        new = np.zeros(tensor.shape)
        rank = len(tensor.shape)
        for idx, _ in np.ndenumerate(new):
            temp = 0
            for dummy, val in np.ndenumerate(tensor):
                temp += val * np.prod([matrix[dummy[i], idx[i]] for i in range(rank)])
            new[idx] = temp
        return new
