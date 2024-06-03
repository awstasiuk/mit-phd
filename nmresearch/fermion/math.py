from numpy import array, conj, pi, diag
from cmath import exp
from tensorflow import constant, tensordot, complex128, einsum
from functools import lru_cache


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
            return array([Math.chop(x) for x in expr])
        else:
            return 0 if -delta <= abs(expr) <= delta else expr

    @staticmethod
    def adj(mat):
        return conj(mat.T)

    @staticmethod
    def exp_diag(vec, t):
        angles = (vec * t) % (2 * pi)
        diag = [exp(-1j * angle) for angle in angles]
        return diag(diag)

    @staticmethod
    def tensor_product(A, B):
        r"""
        compute the tensor outer product. tensorflow does this much faster than we can
        natively in python, so we wrap around their existing functionality.
        """
        A = constant(A, dtype=complex128)
        B = constant(B, dtype=complex128)
        return tensordot(A, B, axes=0).numpy()

    @staticmethod
    def tensor_change_of_basis(tensor, matrix, reversed=False):
        r"""
        Performs a change of basis computation for a general tensor, given a rank 2 change
        of basis matrix, which corresponds to the following operation, as an einsum.

        F_{ab...c} = U_a^i U_b^j ... U_c^k T_{ij...k}

        Here, {F} are the coordinates of the tensor in the new basis, and {T} are the coordinates
        of the tensor in the old basis, and {U} represents the change of basis matrix.
        """
        a = 97
        A = 65
        chars = [str(chr(a + i)) for i in range(len(tensor.shape))]
        chars_upper = [str(chr(A + i)) for i in range(len(tensor.shape))]
        lhs = []
        lhs.append("".join(chars))
        for low, up in zip(chars, chars_upper):
            lhs.append(low + up)

        if reversed:
            chars_upper.reverse()

        ein = ",".join(lhs) + " -> " + "".join(chars_upper)
        mats = [constant(tensor, dtype=complex128)]
        mats.extend([constant(matrix, dtype=complex128) for _ in range(len(chars))])
        return einsum(ein, *mats).numpy()

    @staticmethod
    @lru_cache
    def tw_four_body(n_spin):
        r"""
        Computes the indices which have a non-zero trace, and their trace weighting, and return
        them as a pair of a list of tuples and a list of weights. Further, the returned values
        are globally cached, so that repeat calls to this function will not need to regenerate
        the lists of indices and weights, for a given number of spins.
        """
        n = 2 * n_spin

        idx_list = []
        vals = []

        for i in range(n):
            for j in range(n):
                if i != (j + n_spin) % n and i != j:
                    idx = (i, j, (i + n_spin) % n, (j + n_spin) % n)
                    idx_list.append(idx)
                    vals.append(Math.trace_weight(idx, n_spin))

                    idx = (i, j, (j + n_spin) % n, (i + n_spin) % n)
                    idx_list.append(idx)
                    vals.append(Math.trace_weight(idx, n_spin))
                elif i == (j + n_spin) % n and i != j:
                    for k in range(n):
                        if k != j:
                            idx = (i, j, k, (k + n_spin) % n)
                            idx_list.append(idx)
                            vals.append(Math.trace_weight(idx, n_spin))
        return idx_list, vals

    @staticmethod
    def fermion_weight(idx_str, n):
        return sum([1 if idx_str[i] >= n else -1 for i in range(len(idx_str))])

    @staticmethod
    def trace_weight(idx_str, n):
        r"""
        works for up to quartic operators, as linear and cubic terms have 0 trace for all
        terms.
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
