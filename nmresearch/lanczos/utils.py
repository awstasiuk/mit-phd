import numpy as np
import scipy.sparse as sp
import math, cmath
from functools import lru_cache


def base4_to_index(bin_str):
    n = len(bin_str)
    return sum([ (int(val) % 4) * 4 ** (n-idx-1) for idx,val in enumerate(bin_str)])
    
def base4_to_pauli(bin_str):
    return bin_str.replace("0","I").replace("1","X").replace("2","Y").replace("3","Z")

def pauli_to_base4(bin_str):
    return bin_str.replace("I","0").replace("X","1").replace("Y","2").replace("Z","3")

def pauli_to_index(pauli_str):
    return base4_to_index(pauli_to_base4(pauli_str))

def paulivec_to_str(pauli, tol=1e-10):
    op_list = []
    dim = len(pauli)
    sys = np.log(dim)/np.log(4)
    for idx,val in enumerate(pauli):
        if abs(val) > tol:
            op_list.append(str(val) + "*" + index_to_pauli(idx,str_len=sys))
    return " + ".join(op_list)

def index_to_base(n, b=4, str_len=4):
    if n == 0:
        return "0"*str_len
    digits = ""
    while n:
        digits += str(n % b)
        n //= b
    k=len(digits)
    return ("0"*(str_len-k)) + digits[::-1]

def index_to_pauli(idx,str_len=4):
    return base4_to_pauli(index_to_base(idx,b=4,str_len=4))

def generate_base4_strings(bit_count):
    b_strings = []
    def genbin(n, bs=''):
        if len(bs) == n:
            b_strings.append(bs)
        else:
            genbin(n, bs + '0')
            genbin(n, bs + '1')
            genbin(n, bs + '2')
            genbin(n, bs + '3')

    genbin(bit_count)
    return b_strings

def generate_binary_strings(bit_count):
    binary_strings = []
    def genbin(n, bs=''):
        if len(bs) == n:
            binary_strings.append(bs)
        else:
            genbin(n, bs + '0')
            genbin(n, bs + '1')

    genbin(bit_count)
    return binary_strings

def binary_to_index(bin_str):
    n = len(bin_str)
    return sum([int(val)*2**(n-idx-1) for idx,val in enumerate(bin_str)])

def to_super(opA,opB):
    return sp.sparse.kron(opB,opA.T)

def super_ham(ham):
    r"""
    This used to be `eye_array` but this version of scipy
    """
    dim = ham.shape[0]
    return (to_super(ham,sp.sparse.eye(dim)) - to_super(sp.sparse.eye(dim),ham))