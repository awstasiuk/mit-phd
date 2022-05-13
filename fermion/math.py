from __future__ import division
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import math, cmath
import tensorflow as tf
from functools import lru_cache


"""A package for computing Pfaffians (see pfapack for citation details)"""


def householder_real(x):
    """(v, tau, alpha) = householder_real(x)

    Compute a Householder transformation such that
    (1-tau v v^T) x = alpha e_1
    where x and v a real vectors, tau is 0 or 2, and
    alpha a real number (e_1 is the first unit vector)
    """

    assert x.shape[0] > 0

    sigma = np.dot(x[1:], x[1:])

    if sigma == 0:
        return (np.zeros(x.shape[0]), 0, x[0])
    else:
        norm_x = math.sqrt(x[0] ** 2 + sigma)

        v = x.copy()

        # depending on whether x[0] is positive or negatvie
        # choose the sign
        if x[0] <= 0:
            v[0] -= norm_x
            alpha = +norm_x
        else:
            v[0] += norm_x
            alpha = -norm_x

        v /= np.linalg.norm(v)

        return (v, 2, alpha)


def householder_complex(x):
    """(v, tau, alpha) = householder_real(x)

    Compute a Householder transformation such that
    (1-tau v v^T) x = alpha e_1
    where x and v a complex vectors, tau is 0 or 2, and
    alpha a complex number (e_1 is the first unit vector)
    """
    assert x.shape[0] > 0

    sigma = np.dot(np.conj(x[1:]), x[1:])

    if sigma == 0:
        return (np.zeros(x.shape[0]), 0, x[0])
    else:
        norm_x = cmath.sqrt(x[0].conjugate() * x[0] + sigma)

        v = x.copy()

        phase = cmath.exp(1j * math.atan2(x[0].imag, x[0].real))

        v[0] += phase * norm_x

        v /= np.linalg.norm(v)

    return (v, 2, -phase * norm_x)


def skew_tridiagonalize(A, overwrite_a=False, calc_q=True):
    """T, Q = skew_tridiagonalize(A, overwrite_a, calc_q=True)

    or

    T = skew_tridiagonalize(A, overwrite_a, calc_q=False)

    Bring a real or complex skew-symmetric matrix (A=-A^T) into
    tridiagonal form T (with zero diagonal) with a orthogonal
    (real case) or unitary (complex case) matrix U such that
    A = Q T Q^T
    (Note that Q^T and *not* Q^dagger also in the complex case)

    A is overwritten if overwrite_a=True (default: False), and
    Q only calculated if calc_q=True (default: True)
    """

    # Check if matrix is square
    assert A.shape[0] == A.shape[1] > 0
    # Check if it's skew-symmetric
    assert abs((A + A.T).max()) < 1e-14

    n = A.shape[0]
    A = np.asarray(A)  # the slice views work only properly for arrays

    # Check if we have a complex data type
    if np.issubdtype(A.dtype, np.complexfloating):
        householder = householder_complex
    elif not np.issubdtype(A.dtype, np.number):
        raise TypeError("pfaffian() can only work on numeric input")
    else:
        householder = householder_real

    if not overwrite_a:
        A = A.copy()

    if calc_q:
        Q = np.eye(A.shape[0], dtype=A.dtype)

    for i in xrange(A.shape[0] - 2):
        # Find a Householder vector to eliminate the i-th column
        v, tau, alpha = householder(A[i + 1 :, i])
        A[i + 1, i] = alpha
        A[i, i + 1] = -alpha
        A[i + 2 :, i] = 0
        A[i, i + 2 :] = 0

        # Update the matrix block A(i+1:N,i+1:N)
        w = tau * np.dot(A[i + 1 :, i + 1 :], v.conj())
        A[i + 1 :, i + 1 :] += np.outer(v, w) - np.outer(w, v)

        if calc_q:
            # Accumulate the individual Householder reflections
            # Accumulate them in the form P_1*P_2*..., which is
            # (..*P_2*P_1)^dagger
            y = tau * np.dot(Q[:, i + 1 :], v)
            Q[:, i + 1 :] -= np.outer(y, v.conj())

    if calc_q:
        return (np.asmatrix(A), np.asmatrix(Q))
    else:
        return np.asmatrix(A)


def skew_LTL(A, overwrite_a=False, calc_L=True, calc_P=True):
    """T, L, P = skew_LTL(A, overwrite_a, calc_q=True)

    Bring a real or complex skew-symmetric matrix (A=-A^T) into
    tridiagonal form T (with zero diagonal) with a lower unit
    triangular matrix L such that
    P A P^T= L T L^T

    A is overwritten if overwrite_a=True (default: False),
    L and P only calculated if calc_L=True or calc_P=True,
    respectively (default: True).
    """

    # Check if matrix is square
    assert A.shape[0] == A.shape[1] > 0
    # Check if it's skew-symmetric
    assert abs((A + A.T).max()) < 1e-14

    n = A.shape[0]
    A = np.asarray(A)  # the slice views work only properly for arrays

    if not overwrite_a:
        A = A.copy()

    if calc_L:
        L = np.eye(n, dtype=A.dtype)

    if calc_P:
        Pv = np.arange(n)

    for k in xrange(n - 2):
        # First, find the largest entry in A[k+1:,k] and
        # permute it to A[k+1,k]
        kp = k + 1 + np.abs(A[k + 1 :, k]).argmax()

        # Check if we need to pivot
        if kp != k + 1:
            # interchange rows k+1 and kp
            temp = A[k + 1, k:].copy()
            A[k + 1, k:] = A[kp, k:]
            A[kp, k:] = temp

            # Then interchange columns k+1 and kp
            temp = A[k:, k + 1].copy()
            A[k:, k + 1] = A[k:, kp]
            A[k:, kp] = temp

            if calc_L:
                # permute L accordingly
                temp = L[k + 1, 1 : k + 1].copy()
                L[k + 1, 1 : k + 1] = L[kp, 1 : k + 1]
                L[kp, 1 : k + 1] = temp

            if calc_P:
                # accumulate the permutation matrix
                temp = Pv[k + 1]
                Pv[k + 1] = Pv[kp]
                Pv[kp] = temp

        # Now form the Gauss vector
        if A[k + 1, k] != 0.0:
            tau = A[k + 2 :, k].copy()
            tau /= A[k + 1, k]

            # clear eliminated row and column
            A[k + 2 :, k] = 0.0
            A[k, k + 2 :] = 0.0

            # Update the matrix block A(k+2:,k+2)
            A[k + 2 :, k + 2 :] += np.outer(tau, A[k + 2 :, k + 1])
            A[k + 2 :, k + 2 :] -= np.outer(A[k + 2 :, k + 1], tau)

            if calc_L:
                L[k + 2 :, k + 1] = tau

    if calc_P:
        # form the permutation matrix as a sparse matrix
        P = sp.csr_matrix((np.ones(n), (np.arange(n), Pv)))

    if calc_L:
        if calc_P:
            return (np.asmatrix(A), np.asmatrix(L), P)
        else:
            return (np.asmatrix(A), np.asmatrix(L))
    else:
        if calc_P:
            return (np.asmatrix(A), P)
        else:
            return np.asmatrix(A)


def pfaffian(A, overwrite_a=False, method="P"):
    """pfaffian(A, overwrite_a=False, method='P')

    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses
    either the Parlett-Reid algorithm (method='P', default),
    or the Householder tridiagonalization (method='H')
    """
    # Check if matrix is square
    assert A.shape[0] == A.shape[1] > 0
    # Check if it's skew-symmetric
    assert abs((A + A.T).max()) < 1e-14
    # Check that the method variable is appropriately set
    assert method == "P" or method == "H"

    # Make sure the matrix is using floating point algebra
    if not np.issubdtype(A.dtype, np.inexact):
        A = 1.0 * A

    if method == "P":
        return pfaffian_LTL(A, overwrite_a)
    else:
        return pfaffian_householder(A, overwrite_a)


def pfaffian_LTL(A, overwrite_a=False):
    """pfaffian_LTL(A, overwrite_a=False)

    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses
    the Parlett-Reid algorithm.
    """
    # Check if matrix is square
    assert A.shape[0] == A.shape[1] > 0
    # Check if it's skew-symmetric
    assert abs((A + A.T).max()) < 1e-14

    n = A.shape[0]
    A = np.asarray(A)  # the slice views work only properly for arrays

    # Quick return if possible
    if n % 2 == 1:
        return 0

    if not overwrite_a:
        A = A.copy()

    pfaffian_val = 1.0

    for k in xrange(0, n - 1, 2):
        # First, find the largest entry in A[k+1:,k] and
        # permute it to A[k+1,k]
        kp = k + 1 + np.abs(A[k + 1 :, k]).argmax()

        # Check if we need to pivot
        if kp != k + 1:
            # interchange rows k+1 and kp
            temp = A[k + 1, k:].copy()
            A[k + 1, k:] = A[kp, k:]
            A[kp, k:] = temp

            # Then interchange columns k+1 and kp
            temp = A[k:, k + 1].copy()
            A[k:, k + 1] = A[k:, kp]
            A[k:, kp] = temp

            # every interchange corresponds to a "-" in det(P)
            pfaffian_val *= -1

        # Now form the Gauss vector
        if A[k + 1, k] != 0.0:
            tau = A[k, k + 2 :].copy()
            tau /= A[k, k + 1]

            pfaffian_val *= A[k, k + 1]

            if k + 2 < n:
                # Update the matrix block A(k+2:,k+2)
                A[k + 2 :, k + 2 :] += np.outer(tau, A[k + 2 :, k + 1])
                A[k + 2 :, k + 2 :] -= np.outer(A[k + 2 :, k + 1], tau)
        else:
            # if we encounter a zero on the super/subdiagonal, the
            # Pfaffian is 0
            return 0.0

    return pfaffian_val


def pfaffian_householder(A, overwrite_a=False):
    """pfaffian(A, overwrite_a=False)

    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses the
    Householder tridiagonalization.

    Note that the function pfaffian_schur() can also be used in the
    real case. That function does not make use of the skew-symmetry
    and is only slightly slower than pfaffian_householder().
    """

    # Check if matrix is square
    assert A.shape[0] == A.shape[1] > 0
    # Check if it's skew-symmetric
    assert abs((A + A.T).max()) < 1e-14

    n = A.shape[0]

    # Quick return if possible
    if n % 2 == 1:
        return 0

    # Check if we have a complex data type
    if np.issubdtype(A.dtype, np.complexfloating):
        householder = householder_complex
    elif not np.issubdtype(A.dtype, np.number):
        raise TypeError("pfaffian() can only work on numeric input")
    else:
        householder = householder_real

    A = np.asarray(A)  # the slice views work only properly for arrays

    if not overwrite_a:
        A = A.copy()

    pfaffian_val = 1.0

    for i in xrange(A.shape[0] - 2):
        # Find a Householder vector to eliminate the i-th column
        v, tau, alpha = householder(A[i + 1 :, i])
        A[i + 1, i] = alpha
        A[i, i + 1] = -alpha
        A[i + 2 :, i] = 0
        A[i, i + 2 :] = 0

        # Update the matrix block A(i+1:N,i+1:N)
        w = tau * np.dot(A[i + 1 :, i + 1 :], v.conj())
        A[i + 1 :, i + 1 :] += np.outer(v, w) - np.outer(w, v)

        if tau != 0:
            pfaffian_val *= 1 - tau
        if i % 2 == 0:
            pfaffian_val *= -alpha

    pfaffian_val *= A[n - 2, n - 1]

    return pfaffian_val


def pfaffian_schur(A, overwrite_a=False):
    """Calculate Pfaffian of a real antisymmetric matrix using
    the Schur decomposition. (Hessenberg would in principle be faster,
    but scipy-0.8 messed up the performance for scipy.linalg.hessenberg()).

    This function does not make use of the skew-symmetry of the matrix A,
    but uses a LAPACK routine that is coded in FORTRAN and hence faster
    than python. As a consequence, pfaffian_schur is only slightly slower
    than pfaffian().
    """

    assert np.issubdtype(A.dtype, np.number) and not np.issubdtype(
        A.dtype, np.complexfloating
    )

    assert A.shape[0] == A.shape[1] > 0

    assert abs(A + A.T).max() < 1e-14

    # Quick return if possible
    if A.shape[0] % 2 == 1:
        return 0

    (t, z) = la.schur(A, output="real", overwrite_a=overwrite_a)
    l = np.diag(t, 1)
    return np.prod(l[::2]) * la.det(z)


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
        r"""
        compute the tensor outer product. tensorflow does this much faster than we can
        natively in python, so we wrap around their existing functionality.
        """
        A = tf.constant(A, dtype=tf.complex128)
        B = tf.constant(B, dtype=tf.complex128)
        return tf.tensordot(A, B, axes=0).numpy()

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
        mats = [tf.constant(tensor, dtype=tf.complex128)]
        mats.extend(
            [tf.constant(matrix, dtype=tf.complex128) for _ in range(len(chars))]
        )
        return tf.einsum(ein, *mats).numpy()

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
