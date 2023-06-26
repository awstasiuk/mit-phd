from nmresearch.crystal.disorder import Disorder

from numpy import zeros
from numpy.linalg import norm
from numpy.random import default_rng
from pickle import load, dump


class Disorder:
    """
    This class should accept a :type:`Crystal`, and be able to compute the mean field at a given
    site.
    """

    def __init__(self, crystal, shell_radius=1):
        self._crystal = crystal
        self._network = {}
        self._shell_radius = shell_radius

    @staticmethod
    def heteronuclear_coupling(origin, source):
        """
        Compute the heteronuclear dipolar coupling strength for a pair of `Atom`s,
        `origin` and `source`.
        """
        p1 = origin.position
        p2 = source.position
        hbar = 1.05457 * 10 ** (-34)  # J s / rad
        r = (p2 - p1) * 10 ** (-10)
        dx = norm(r)
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
        lattice = self.crystal.generate_lattice(self.shell_radius)
        network = []
        for atompos in lattice:
            if not atompos.name == origin.name:
                atompos.coupling = Disorder.heteronuclear_coupling(
                    origin, atompos
                )
            network.append(atompos)

        self._network[origin] = network
        return network

    @staticmethod
    def get_rand_config(network):
        """
        get a random configuration of n_spins
        each with spin-dimension s, so spin-1/2 is s=2,
        spin-2 is s=5, etc (j=(s-1)/2)
        """
        rng = default_rng()
        config = []
        for atompos in network:
            s = atompos.dim_s
            config.append(rng.integers(low=0, high=s) - (s - 1) / 2)
        return config

    def mean_field_calc(self, origin, regen=False):
        """
        Calculate disorder contribution to the local field at a specific atom
        Randomized sum of discretized spin orientations times heteronuclear coupling strengths
        """
        if regen:
            crystal = self.generate_network(origin)
        else:
            crystal = self.get_network(origin)
        config = Disorder.get_rand_config(crystal)
        return sum(
            [spin.coupling * orien for spin, orien in zip(crystal, config)]
        )

    def variance_estimate(self, origin):
        """
        Gives variance of mean field calculated at origin according to uniformly distributed nuclear spins over spin dimension
        Computes variance only for one fixed isotope position
        (which is generated randomly from probabilities of each isotope at lattice site)
        """
        crystal = self.get_network(origin)
        return sum(
            [
                ((spin.dim_s**2 - 1) / 12) * (spin.coupling**2)
                for spin in crystal
            ]
        )

    def kurtosis_estimate(self, origin):
        """
        Same conditions as previous, computes kurtosis (related to 4th moment)
        """
        crystal = self.get_network(origin)
        fm = sum(
            [
                (spin.dim_s**2 - 1)
                * (
                    (3 * spin.dim_s**2 - 7) / 240
                    - (spin.dim_s**2 - 1) / 48
                )
                * (spin.coupling**4)
                for spin in crystal
            ]
        )
        return 3 + fm / self.variance_estimate(origin) ** 2

    def simulation(
        self, origin, trials, filename="disorder_sim.dat", regen=False
    ):
        """
        Monte-carlo simulation of mean field at orig_atom
        given contributing atoms in atomlis according to ij coupling
        Regen boolean says whether all trials have one fixing of isotopes (faster)
        or vary isotopes/atoms at lattice sites (slower)
        """
        try:
            my_distro = load(open(filename, "rb"))
        except (OSError, IOError) as e:
            my_distro = zeros(trials)
            crystal = self.get_network(origin)
            for idx in range(trials):
                my_distro[idx] = self.mean_field_calc(origin, regen)
            with open(filename, "wb") as fi:
                dump(my_distro, fi)
        return my_distro

    @property
    def crystal(self):
        return self._crystal

    @property
    def shell_radius(self):
        return self._shell_radius
