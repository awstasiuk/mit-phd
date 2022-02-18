import numpy as np

class Math:
    r"""
    A class which does math stuff
    """
    @staticmethod
    def kron_delta(i,j):
        if int(i) != i or int(j) !=j:
            raise ValueError("arguments should be integers")
        return int(i==j)
