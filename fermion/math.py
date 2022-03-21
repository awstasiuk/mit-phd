import numpy as np
import tensorflow as tf
import cmath


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
        if hasattr(expr, "__iter__") and len(expr.shape) > 0:
            return np.array([Math.chop(x) for x in expr])
        else:
            return 0 if -delta <= abs(expr) <= delta else expr

    @staticmethod
    def adj(mat):
        return np.conj(mat.T)

    @staticmethod
    def exp_diag(vec, t):
        angles = (vec * t) % (2 * np.pi)
        diag = [cmath.exp(-1j * angle) for angle in angles]
        return np.diag(diag)

    @staticmethod
    def tensor_product(A, B):
        A = tf.constant(A, dtype=tf.complex128)
        B = tf.constant(B, dtype=tf.complex128)
        return tf.tensordot(A, B, axes=0).numpy()

    @staticmethod
    def tensor_change_of_basis(tensor, matrix):
        a = 97
        A = 65
        chars = [str(chr(a + i)) for i in range(len(tensor.shape))]
        chars_upper = [str(chr(A + i)) for i in range(len(tensor.shape))]
        lhs = []
        lhs.append("".join(chars))
        for low, up in zip(chars, chars_upper):
            lhs.append(low + up)

        ein = ",".join(lhs) + " -> " + "".join(chars_upper)
        mats = [tf.constant(tensor, dtype=tf.complex128)]
        mats.extend([tf.constant(matrix) for _ in range(len(chars))])
        return tf.einsum(ein, *mats).numpy()

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
        for i in range(len(idx_str)):
            if idx_str[i] == idx_str[(i + 1) % len(idx_str)]:
                return 0
        if len(idx_str) == 2:
            return 0.5
        elif len(idx_str) == 4:
            f = idx_str[0]
            if idx_str[0] in idx_str[1 : len(idx_str)]:
                return 0.5
            else:
                for j in range(3):
                    if idx_str[j] == (idx_str[j + 1] + n) % (2 * n):
                        return 0.25
                else:
                    return -0.25

        return 0
