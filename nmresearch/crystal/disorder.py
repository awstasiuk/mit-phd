from nmresearch.crystal.crystal import Crystal

from numpy import zeros
from numpy import pi
from numpy.linalg import norm
from numpy.random import default_rng
from numpy import inner
from pickle import load, dump


class Disorder:
    """
    This accepts an instance of a :type:`Crystal`, and is able compute statistical
    quantities relevant to the `crystal`, using Monte Carlo techniques or otherwise.
    """

    def __init__(self, crystal, shell_radius=1):
        self._crystal = crystal
        self._network = {}
        self._double_network = {}
        self._shell_radius = shell_radius

    @staticmethod
    def heteronuclear_coupling(origin, source):
        """
        Compute the heteronuclear dipolar coupling strength for a pair of `Atom`s,
        `origin` and `source` - result in
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

    def nuclear_coupling(origin, source, bdir=[0, 0, 1]):
        """
        Compute the dipolar coupling strength for a pair of `Atom`s,
        `origin` and `source` where the direction of magnetic field is not necessarily aligned with z
        """
        bdir = bdir / norm(bdir)
        p1 = origin.position
        p2 = source.position
        hbar = 1.05457 * 10 ** (-34)  # J s / rad
        r = (p2 - p1) * 10 ** (-10)
        dx = norm(r)
        rhat = r / dx
        cos = inner(rhat, bdir)
        return (
            (10**-7)
            * hbar
            * origin.gamma
            * source.gamma
            * (1 - 3 * cos**2)
            / dx**3
        )

    def spin_diffusion_coeff_parallel(
        self, origin, bdir=[0, 0, 1]
    ):  # Based on formula given in Khutsishvilli, 1970ish?
        bdir = bdir / norm(bdir)
        lattice = self.crystal.generate_lattice(self.shell_radius)
        big_S = 0
        numerator = 0
        for atompos in lattice:
            if atompos.name == origin.name:
                if (
                    not norm(atompos.position - origin.position) <= 0.1
                ):  # maybe should be less than some small cutoff, hard to compare floats?
                    big_S += (
                        1
                        / 3
                        * (origin.dim_s**2 - 1)
                        * Disorder.nuclear_coupling(origin, atompos, bdir) ** 2
                    )
                    numerator += (
                        Disorder.nuclear_coupling(origin, atompos, bdir) ** 2
                        * inner(
                            (origin.position - atompos.position) * 10**-10,
                            bdir,
                        )
                        ** 2
                    )
            else:
                big_S += (
                    4
                    / 27
                    * (atompos.dim_s**2 - 1)
                    * Disorder.nuclear_coupling(origin, atompos, bdir) ** 2
                )
        spectral_lambda = (3 / 2**0.5) / pi**0.5
        pre_const = spectral_lambda * pi**0.5 / 72  # old constant
        pre_const = 1 / 8 * (pi / 5) ** 0.5
        D = pre_const * numerator / (big_S) ** 0.5
        return D * 10**4  # in cm^2/s

    @staticmethod
    def perpendicular_vec(a):
        if a[0] < a[2]:
            return [0, a[2], -a[1]]
        else:
            return [-a[1], a[0], 0]

    def spin_diffusion_coeff_perpendicular(
        self, origin, bdir=[0, 0, 1]
    ):  # Based on formula given in Khutsishvilli, 1970ish?, with different constants
        perp_b = Disorder.perpendicular_vec(bdir)
        perp_b = perp_b / norm(bdir)
        lattice = self.crystal.generate_lattice(self.shell_radius)
        big_S = 0
        numerator = 0
        for atompos in lattice:
            if atompos.name == origin.name:
                if (
                    not norm(atompos.position - origin.position) <= 0.1
                ):  # maybe should be less than some small cutoff, hard to compare floats?
                    big_S += (
                        1
                        / 3
                        * (origin.dim_s**2 - 1)
                        * Disorder.nuclear_coupling(origin, atompos, perp_b)
                        ** 2
                    )
                    numerator += (
                        Disorder.nuclear_coupling(origin, atompos, perp_b) ** 2
                        * norm(
                            (origin.position - atompos.position) * 10**-10
                        )
                        ** 2
                    )
            else:
                big_S += (
                    4
                    / 27
                    * (atompos.dim_s**2 - 1)
                    * Disorder.nuclear_coupling(origin, atompos, bdir) ** 2
                )
        spectral_lambda = (3 / 2**0.5) / pi**0.5
        pre_const = spectral_lambda * pi**0.5 / 72
        pre_const = 1 / 24 * (pi / 5) ** 0.5
        D = pre_const * numerator / (big_S) ** 0.5
        return D * 10**4  # in cm^2/s

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

    def double_get_network(self, origin_1, origin_2, plus=False):
        if (origin_1, origin_2, plus) in self._network.keys():
            return self._double_network[(origin_1, origin_2, plus)]
        return self.double_generate_network(origin_1, origin_2, plus)

    def double_generate_network(self, origin_1, origin_2, plus=False):
        r"""
        Note that origin_1, origin_2 need to be the same atom for this to work.
        """
        assert origin_1.name == origin_2.name
        if plus == True:
            a = 1
        else:
            a = -1
        lattice = self.crystal.generate_lattice(self.shell_radius)
        network = []
        for atompos in lattice:
            if not atompos.name == origin_1.name:
                atompos.coupling = Disorder.heteronuclear_coupling(
                    origin_1, atompos
                ) + a * Disorder.heteronuclear_coupling(origin_2, atompos)
            network.append(atompos)

        self._double_network[(origin_1, origin_2, plus)] = network
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

    def double_mean_field_calc(
        self, origin_1, origin_2, plus=False, regen=False
    ):
        """
        Calculate sum or difference of disorder contribution to the local field at two specific atoms
        Randomized sum of discretized spin orientations times heteronuclear coupling strengths
        """

        if regen:
            double_crystal = self.double_generate_network(
                origin_1, origin_2, plus
            )
        else:
            double_crystal = self.double_get_network(origin_1, origin_2, plus)

        config = Disorder.get_rand_config(double_crystal)
        return sum(
            [
                spin.coupling * orien
                for spin, orien in zip(double_crystal, config)
            ]
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

    def double_variance_estimate(self, origin_1, origin_2, plus=False):
        """
        Gives variance of mean field calculated at origin according to uniformly distributed nuclear spins over spin dimension
        Computes variance only for one fixed isotope position
        (which is generated randomly from probabilities of each isotope at lattice site)
        """
        crystal = self.double_generate_network(origin_1, origin_2, plus)
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

    def double_simulation(
        self,
        origin_1,
        origin_2,
        trials,
        filename=(
            "double_disorder_sim_plus.dat",
            "double_disorder_sim_minus.dat",
        ),
        regen=False,
    ):
        """
        Monte-carlo simulation of mean field at orig_atom
        given contributing atoms in atomlis according to ij coupling
        Regen boolean says whether all trials have one fixing of isotopes (faster)
        or vary isotopes/atoms at lattice sites (slower)
        """
        try:
            my_distro_plus = load(open(filename[0], "rb"))
            my_distro_minus = load(open(filename[1], "rb"))
        except (OSError, IOError) as e:
            my_distro_plus = zeros(trials)
            my_distro_minus = zeros(trials)
            for idx in range(trials):
                my_distro_plus[idx] = self.double_mean_field_calc(
                    origin_1, origin_2, True, regen
                )
                my_distro_minus[idx] = self.double_mean_field_calc(
                    origin_1, origin_2, False, regen
                )
            with open(filename[0], "wb") as fi:
                dump(my_distro_plus, fi)
            with open(filename[1], "wb") as fil:
                dump(my_distro_minus, fil)
        return (my_distro_plus, my_distro_minus)

    @property
    def crystal(self):
        return self._crystal

    @property
    def shell_radius(self):
        return self._shell_radius
