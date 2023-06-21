from crystal import Crystal
from atom import Atom

import numpy as np
import matplotlib.pyplot as plt
import pickle


class Disorder:
    """
    This class should accept a :type:`Crystal`, and be able to compute the mean field at a given
    site.
    """

    def __init__(self, crystal, shell_rad=1):
        self._crystal = crystal
        self._network = {}
        self._shell_rad = shell_rad

    @staticmethod
    def heteronuclear_coupling(origin, source):
        """
        Compute the heteronuclear dipolar coupling strength for a pair of `Atom`s,
        `origin` and `source`.
        """
        p1 = origin.pos()
        p2 = source.pos()
        hbar = 1.05457 * 10 ** (-34)  # J s / rad
        r = (p2 - p1) * 10 ** (-10)
        dx = np.linalg.norm(r)
        rhat = r / dx
        cos = rhat[2]
        return (
            (10**-7)
            * hbar
            * origin.gamma
            * source.gamma
            * (1 - 3 * cos**2)
            / dx**3
        )

    def get_network(self, origin):
        if origin in self._network.keys():
            return self._network[origin]
        return self.generate_network(origin)

    def generate_network(self, origin):
        r"""
        Generates the network of heteronuclear couplings for the origin atom and the
        surrounding crystal structure which was used to instantiate this instance.
        """
        lattice = self._crystal.generate_lattice(self.shell_rad)
        network = []
        for atompos in xtal:
            if atompos.name in atomlis:
                atompos.set_cpl(Disorder.heteronuclear_coupling(orig_atom, atompos))
            network.append(atompos)

        self._network[origin] = network
        return network

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

    def mean_field_calc(self, orig_atom, atomlis):
        """
        describe me
        """
        crystal = self.cpl_xtal(orig_atom, atomlis)
        config = Disorder.get_rand_config(crystal)
        return sum([spin.cpl() * orien for spin, orien in zip(crystal, config)])

    @staticmethod
    def mean_field_helper(crystal):
        """
        describe me
        """
        config = Disorder.get_rand_config(crystal)
        return sum([spin.cpl() * orien for spin, orien in zip(crystal, config)])

    def variance_estimate(self, orig_atom, atomlis):
        """
        Gives variance of mean field calculated at origin according to uniformly distributed nuclear spins over spin dimension
        Computes variance only for one fixed isotope position
        (which is generated randomly from probabilities of each isotope at lattice site)
        """
        crystal = self.cpl_xtal(orig_atom, atomlis)
        return sum(
            [((spin.dim_s**2 - 1) / 12) * (spin.cpl() ** 2) for spin in crystal]
        )

    def kurtosis_estimate(self, orig_atom, atomlis):
        """
        Same conditions as previous, computes kurtosis (related to 4th moment)
        """
        crystal = self.cpl_xtal(orig_atom, atomlis)
        fm = sum(
            [
                (spin.dim_s**2 - 1)
                * ((3 * spin.dim_s**2 - 7) / 240 - (spin.dim_s**2 - 1) / 48)
                * (spin.cpl() ** 4)
                for spin in crystal
            ]
        )
        return 3 + fm / self.variance_estimate(orig_atom, atomlis) ** 2

    def simulation(self, orig_atom, atomlis, trials, filename="disorder_sim.dat"):
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
            crystal = self.cpl_xtal(orig_atom, atomlis)
            for idx in range(trials):
                my_distro[idx] = Disorder.mean_field_helper(crystal)
            with open(filename, "wb") as fi:
                pickle.dump(my_distro, fi)
        return my_distro

    def abundance_simulation(self, orig_atom, atomlis, trials, filename):
        """
        Runs simulation accounting for changing isotopes/atoms at lattice sites
        Slower than simulation function
        """
        try:
            my_distro = pickle.load(open(filename, "rb"))
        except (OSError, IOError) as e:
            my_distro = np.zeros(trials)
            for idx in range(trials):
                my_distro[idx] = self.mean_field_calc(orig_atom, atomlis)
            with open(filename, "wb") as fi:
                pickle.dump(my_distro, fi)
        return my_distro

    @property
    def crystal(self):
        return self._crystal

    @property
    def shell_rad(self):
        return self._shell_rad
