from itertools import product
from collections import OrderedDict
from functools import lru_cache
from numpy import array, eye, kron, log2, complex128, ndarray, sqrt
from scipy.sparse import csr_matrix, lil_matrix, kron

from nmresearch.lanczos.utils import generate_binary_strings, binary_to_index

X = array([[0, 1], [1, 0]])
Y = array([[0, -1j], [1j, 0]])
Z = array([[1, 0], [0, -1]])

pauli_label_ops = [("I", eye(2)), ("X", X), ("Y", Y), ("Z", Z)]


class OperatorBasis(object):
    """
    Encapsulate a complete set of basis operators.
    """

    def __init__(self, labels_ops):
        """
        Encapsulates a set of linearly independent operators.

        :param (list|tuple) labels_ops: Sequence of tuples (label, operator) where label is a string
            and operator is a numpy.ndarray/
        """
        self.ops_by_label = OrderedDict(labels_ops)
        self.labels = list(self.ops_by_label.keys())
        self.ops = list(self.ops_by_label.values())
        self.dim = len(self.ops)

    def product(self, *bases):
        """
        Compute the tensor product with another basis.

        :param bases: One or more additional bases to form the product with.
        :return (OperatorBasis): The tensor product basis as an OperatorBasis object.
        """
        if len(bases) > 1:
            basis_rest = bases[0].product(*bases[1:])
        else:
            assert len(bases) == 1
            basis_rest = bases[0]

        labels_ops = [
            (b1l + b2l, kron(b1, b2))
            for (b1l, b1), (b2l, b2) in product(self, basis_rest)
        ]

        return OperatorBasis(labels_ops)

    def __iter__(self):
        """
        Iterate over tuples of (label, basis_op)

        :return: Yields the labels and qutip operators corresponding to the vectors in this basis.
        :rtype: tuple (str, qutip.qobj.Qobj)
        """
        for l, op in zip(self.labels, self.ops):
            yield l, op

    def __pow__(self, n):
        """
        Create the n-fold tensor product basis.

        :param int n: The number of identical tensor factors.
        :return: The product basis.
        :rtype: OperatorBasis
        """
        if not isinstance(n, int):
            raise TypeError("Can only accept an integer number of factors")
        if n < 1:
            raise ValueError("Need positive number of factors")
        if n == 1:
            return self
        return self.product(*([self] * (n - 1)))

    def __repr__(self):
        return "<span[{}]>".format(",".join(self.labels))


PAULI_BASIS = OperatorBasis(pauli_label_ops)


class PauliMatrix:

    def __init__(self, sites):
        self.L = sites
        self.strings = generate_binary_strings(sites)

    @lru_cache
    def sigmaX(self, site):
        r"""
        site is a number from 0 to L-1
        """
        sp_mat = lil_matrix((2**self.L, 2**self.L))
        for bin in self.strings:
            amp = 1
            if bin[site] == "0":
                target = bin[:site] + "1" + bin[site + 1 :]
            else:
                target = bin[:site] + "0" + bin[site + 1 :]
            sp_mat[binary_to_index(bin), binary_to_index(target)] = amp
        return sp_mat.tocsr()

    @lru_cache
    def sigmaY(self, site):
        r"""
        site is a number from 0 to L-1
        """
        sp_mat = lil_matrix((2**self.L, 2**self.L), dtype=complex128)
        for bin in self.strings:
            if bin[site] == "0":
                target = bin[:site] + "1" + bin[site + 1 :]
                amp = 1j
            else:
                target = bin[:site] + "0" + bin[site + 1 :]
                amp = -1j
            sp_mat[binary_to_index(bin), binary_to_index(target)] = amp
        return sp_mat.tocsr()

    @lru_cache
    def sigmaZ(self, site):
        r"""
        site is a number from 0 to L-1
        """
        sp_mat = lil_matrix((2**self.L, 2**self.L))
        for bin in self.strings:
            target = bin
            if bin[site] == "0":
                amp = 1
            else:
                amp = -1
            sp_mat[binary_to_index(bin), binary_to_index(target)] = amp
        return sp_mat.tocsr()


@lru_cache
def computational2pauli_basis_matrix(dim) -> ndarray:
    r"""
    Produces a basis transform matrix that converts from a computational basis to the unnormalized
    pauli basis.

    This is the conjugate transpose of pauli2computational_basis_matrix with an extra dimensional
    factor.

    .. math::

        \rm{c2p\_transform(dim)}  = \frac{1}{dim} sum_{k=1}^{dim^2}  | k > << \sigma_k |

    For example

    .. math::

        vec(\sigma_z) = | \sigma_z >> = [1, 0, 0, -1].T

    in the computational basis, so

    .. math::

        c2p * | \sigma_z >> = [0, 0, 0, 1].T

    :param dim: dimension of the hilbert space on which the operators act.
    :return: A dim**2 by dim**2 basis transform matrix
    """
    return pauli2computational_basis_matrix(dim).conj().T / dim


@lru_cache
def pauli2computational_basis_matrix(dim) -> ndarray:
    r"""
    Produces a basis transform matrix that converts from the unnormalized pauli basis to the
    computational basis.

    .. math::

        \rm{p2c\_transform(dim)} = \sum_{k=1}^{dim^2}  | \sigma_k >> <k|

    For example

    .. math::

        \sigma_x = [0, 1, 0, 0].T

    in the 'pauli basis', so

    .. math::

        p2c * \sigma_x = vec(\sigma_x) = | \sigma_x >>

    :param dim: dimension of the hilbert space on which the operators act.
    :return: A dim**2 by dim**2 basis transform matrix
    """
    n_qubits = int(log2(dim))

    # conversion_mat = zeros((dim ** 2, dim ** 2), dtype=complex)
    conversion_mat = csr_matrix((dim**2, dim**2), dtype=complex128)
    for i, pauli in enumerate(n_qubit_pauli_basis(n_qubits)):
        pauli_label = lil_matrix((dim**2, 1), dtype=complex128)
        pauli_label[i] = 1.0
        pauli_mat = pauli[1]
        conversion_mat += kron(vec(pauli_mat), pauli_label.T)

    return conversion_mat


def n_qubit_pauli_basis(n):
    """
    Construct the tensor product operator basis of `n` PAULI_BASIS's.

    :param int n: The number of qubits.
    :return: The product Pauli operator basis of `n` qubits
    :rtype: OperatorBasis
    """
    if n >= 1:
        return PAULI_BASIS**n
    else:
        raise ValueError("n = {} should be at least 1.".format(n))


def vec(matrix: ndarray) -> ndarray:
    """
    Vectorize, or "vec", a matrix by column stacking.

    For example the 2 by 2 matrix A::

        A = [[a, b]
             [c, d]]

    becomes::

      |A>> := vec(A) = (a, c, b, d)^T

    where `|A>>` denotes the vec'ed version of A and :math:`^T` denotes transpose.

    :param matrix: A N (rows) by M (columns) numpy array.
    :return: Returns a column vector with  N by M rows.
    """
    return matrix.T.reshape((-1, 1))


def superop2pauli_liouville(superop: ndarray) -> ndarray:
    """
    Converts a superoperator into a pauli_liouville matrix.

    This is achieved by a linear change of basis.

    :param superop: a dim**2 by dim**2 superoperator
    :return: dim**2 by dim**2 Pauli-Liouville matrix
    """
    dim = int(sqrt(superop.shape[0]))
    c2p_basis_transform = computational2pauli_basis_matrix(dim)
    return c2p_basis_transform @ superop @ c2p_basis_transform.conj().T * dim
