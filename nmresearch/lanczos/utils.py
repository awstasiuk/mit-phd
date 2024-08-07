from numpy import log, zeros
from numpy.random import normal, random
from numpy.linalg import norm
from scipy.sparse import kron, identity, kronsum


def base4_to_index(bin_str):
    n = len(bin_str)
    return sum([(int(val) % 4) * 4 ** (n - idx - 1) for idx, val in enumerate(bin_str)])


def base4_to_pauli(bin_str):
    return (
        bin_str.replace("0", "I").replace("1", "X").replace("2", "Y").replace("3", "Z")
    )


def pauli_to_base4(bin_str):
    return (
        bin_str.replace("I", "0").replace("X", "1").replace("Y", "2").replace("Z", "3")
    )


def pauli_to_index(pauli_str):
    return base4_to_index(pauli_to_base4(pauli_str))


def paulivec_to_str(pauli, tol=1e-10):
    op_list = []
    dim = len(pauli)
    sys = log(dim) / log(4)
    for idx, val in enumerate(pauli):
        if abs(val) > tol:
            op_list.append(str(val) + "*" + index_to_pauli(idx, str_len=sys))
    return " + ".join(op_list)


def index_to_base(n, b=4, str_len=4):
    if n == 0:
        return "0" * str_len
    digits = ""
    while n:
        digits += str(n % b)
        n //= b
    k = len(digits)
    return ("0" * (str_len - k)) + digits[::-1]


def index_to_pauli(idx, str_len=4):
    return base4_to_pauli(index_to_base(idx, b=4, str_len=4))


def generate_base4_strings(bit_count):
    b_strings = []

    def genbin(n, bs=""):
        if len(bs) == n:
            b_strings.append(bs)
        else:
            genbin(n, bs + "0")
            genbin(n, bs + "1")
            genbin(n, bs + "2")
            genbin(n, bs + "3")

    genbin(bit_count)
    return b_strings


def generate_binary_strings(bit_count):
    binary_strings = []

    def genbin(n, bs=""):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + "0")
            genbin(n, bs + "1")

    genbin(bit_count)
    return binary_strings


def binary_to_index(bin_str):
    n = len(bin_str)
    return sum([int(val) * 2 ** (n - idx - 1) for idx, val in enumerate(bin_str)])


def to_super(opA, opB):
    return kron(opB, opA.T, format="csr")


def super_ham(ham):
    r"""
    This used to be `eye_array` but this version of scipy
    """
    dim = ham.shape[0]
    id = identity(dim, format="coo")
    temp = ham.tocoo()
    return to_super(temp, id) - to_super(id, temp)


def super_ham_alt(ham):
    return kronsum(-1 * ham, ham.T, format="csr")


def basis_vec(dim, idx):
    vec = zeros(dim)
    vec[idx] = 1.0
    return vec


def random_ball(num_points, dimension, radius=1):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = normal(size=(dimension, num_points))
    random_directions /= norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = random(num_points) ** (1 / dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T
