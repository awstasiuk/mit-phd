from fermion.math import Math as fm
from fermion.operator import Operator

import numpy as np
from scipy import linalg as la
from math import e, pi
import itertools
import tensorflow as tf


class Majorana(Operator):
    r"""
    A class desciribing a quadratic multi-body Majorana-fermionic operator
    """

    def __init__(self, op_or_coef):
        r"""
        initialize the object
        """
        if isinstance(op_or_coef, Operator):
            op = op_or_coef
            T = np.kron([[1, 1], [-1, 1]] / np.sqrt(2), np.eye(n))
            coef = {}
            for k in op.components:
                if k == 0:
                    coef[0] = op.coef[0]
                elif k == 1:
                    coef[1] = self.U(t) @ op.coef[1]
                else:
                    coef[k] = fm.tensor_change_of_basis(op.coef[k], self.U(t))
            super().__init__(op.n_fermion, coef)
        else:
            # this step should be sanitized, but probably we will never instantiate directly
            # from coeficients
            super().__init__(len(op_or_coef[2]), op_or_coef)
