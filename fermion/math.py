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
    def tensor_product(A, B):
        A = np.array(A)
        B = np.array(B)
        shape = A.shape + B.shape
        rA = len(A.shape)
        rB = len(B.shape)
        new = np.zeros(shape, dtype=np.complex)
        for idx, _ in np.ndenumerate(new):
            new[idx] = A[idx[0:rA]] * B[idx[rA : (rA + rB)]]
        return new

    @staticmethod
    def tensor_change_of_basis(tensor, matrix):
        new = np.zeros(tensor.shape, dtype=np.complex)
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

    @staticmethod
    def fermion_weight(idx_str, n):
        return sum([1 if idx_str[i] >= n else -1 for i in range(len(idx_str))])

    @staticmethod
    def trace_weight(idx_str, n):
        r"""
        works for quadratic and quartic operators.
        """
        for idx in idx_str:
            if (idx + n) % (2 * n) not in idx_str:
                return 0
        for i in range(len(idx_str) - 1):
            if idx_str[i] == idx_str[i + 1]:
                return 0
        if len(idx_str) == 2:
            return 0.5
        elif len(idx_str) == 4:
            f = idx_str[0]
            if idx_str[0] in idx_str[1 : len(idx_str)]:
                return 0.5
            else:
                return 0.25

        return 0
