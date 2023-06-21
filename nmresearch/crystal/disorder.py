from sympy import inverse_fourier_transform, cos, exp, sqrt, pi
from sympy.abc import x, k
import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.linalg import expm

# from sklearn.mixture import GaussianMixture
from timeit import default_timer as timer

import crystal
import atom

import pickle


class Disorder:
    """
    This class should accept a :type:`Crystal`, and be able to compute the mean field at a given
    site.


    """

    def __init__(self, crystal):
        self._crystal = crystal

    @staticmethod
    def ij_coupling(atpt1, atpt2):

        """
        compute coupling coefficients in terms of p1, p2 in cell basis
        coupling returned in units of rad/s
        """
        p1 = atpt1.pos()
        p2 = atpt2.pos()
        hbar = 1.05457 * 10 ** (-34)  # J s / rad
        r = (p2 - p1) * 10**-10
        dx = np.linalg.norm(r)
        rhat = r / dx
        cos = rhat[2]
        return (
            (10**-7)
            * hbar
            * atpt1.gamma
            * atpt2.gamma
            * (1 - 3 * cos**2)
            / dx**3
        )

    def cpl_xtal(self, orig_atom, atomlis, shell_rad=1):
        r"""
        generates crystal, should be bottleneck
        specify mean field point as an ordered pair (atom, position), crystal, and list of atoms with which we want to couple
        """
        xtal = self._crystal.generate_crystal(shell_rad)
        new_xtal = []
        for atompos in xtal:
            if atompos.name in atomlis:
                atompos.set_cpl(Disorder.ij_coupling(orig_atom, atompos))
            new_xtal.append(atompos)
        return new_xtal

    @staticmethod
    def get_rand_config(cpl_xtal):
        """
        get a random configuration of n_spins
        each with spin-dimension s, so spin-1/2 is s=2,
        spin-2 is s=5, etc (j=(s-1)/2)
        """
        rng = np.random.default_rng()
        config = []
        for atompos in cpl_xtal:
            s = atompos.dim_s
            config.append(rng.integers(low=0, high=s) - (s - 1) / 2)
        return config

    # @staticmethod
    def mean_field_calc(self, orig_atom, atomlis, shell_rad=1):
        crystal = self.cpl_xtal(orig_atom, atomlis, shell_rad)
        config = Disorder.get_rand_config(crystal)
        return sum(
            [spin.cpl() * orien for spin, orien in zip(crystal, config)]
        )

    @staticmethod
    def mean_field_helper(crystal):
        config = Disorder.get_rand_config(crystal)
        return sum(
            [spin.cpl() * orien for spin, orien in zip(crystal, config)]
        )

    def variance_estimate(self, orig_atom, atomlis, shell_rad=1):
        """
        Gives variance of mean field calculated at origin according to uniformly distributed nuclear spins over spin dimension
        Computes variance only for one fixed isotope position
        (which is generated randomly from probabilities of each isotope at lattice site)
        """
        crystal = self.cpl_xtal(orig_atom, atomlis, shell_rad)
        return sum(
            [
                ((spin.dim_s**2 - 1) / 12) * (spin.cpl() ** 2)
                for spin in crystal
            ]
        )

    def kurtosis_estimate(self, orig_atom, atomlis, shell_rad=1):
        """
        Same conditions as previous, computes kurtosis (related to 4th moment)
        """
        crystal = self.cpl_xtal(orig_atom, atomlis, shell_rad)
        fm = sum(
            [
                (spin.dim_s**2 - 1)
                * (
                    (3 * spin.dim_s**2 - 7) / 240
                    - (spin.dim_s**2 - 1) / 48
                )
                * (spin.cpl() ** 4)
                for spin in crystal
            ]
        )
        return (
            3 + fm / self.variance_estimate(orig_atom, atomlis, shell_rad) ** 2
        )

    def simulation(self, orig_atom, atomlis, trials, filename, shell_rad=1):
        """
        Monte-carlo simulation of mean field at orig_atom
        given contributing atoms in atomlis according to ij coupling
        Note this runs all trials for only one fixing of isotopes
        """
        try:
            my_distro = pickle.load(open(filename, "rb"))
        except (OSError, IOError) as e:
            if time:
                start = timer()
            my_distro = np.zeros(trials)
            crystal = self.cpl_xtal(orig_atom, atomlis, shell_rad)
            for idx in range(trials):
                my_distro[idx] = Disorder.mean_field_helper(crystal)
            with open(filename, "wb") as fi:
                pickle.dump(my_distro, fi)
        return my_distro

    def abundance_simulation(
        self, orig_atom, atomlis, trials, filename, shell_rad=1
    ):
        """
        Runs simulation accounting for changing isotopes/atoms at lattice sites
        Slower than simulation function
        """
        try:
            my_distro = pickle.load(open(filename, "rb"))
        except (OSError, IOError) as e:
            my_distro = np.zeros(trials)
            for idx in range(trials):
                my_distro[idx] = self.mean_field_calc(
                    orig_atom, atomlis, shell_rad
                )
            with open(filename, "wb") as fi:
                pickle.dump(my_distro, fi)
        return my_distro
