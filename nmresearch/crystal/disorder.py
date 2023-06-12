from sympy import inverse_fourier_transform, cos, exp, sqrt, pi
from sympy.abc import x, k

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.linalg import expm

import pickle


class Disorder:
    r"""
    This class should accept a :type:`Crystal`, and be able to compute the mean field at a given
    site.


    """

    def __init__(self, crystal):
        self._crystal = crystal
