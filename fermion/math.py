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
        for idx1, _ in np.ndenumerate(new):
            temp = 0
            for idx2, val in np.ndenumerate(tensor):
                if val != 0:
                    temp += val * np.prod(
                        [matrix[idx2[i], idx1[i]] for i in range(rank)]
                    )
            new[idx1] = temp
        return new
