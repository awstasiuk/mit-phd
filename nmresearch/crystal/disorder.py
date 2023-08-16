from nmresearch.crystal.crystal import Crystal

from numpy import zeros
from numpy import pi
from numpy.linalg import norm
from numpy.random import default_rng
from numpy import dot
from numpy import array
from pickle import load, dump
from numpy import sinc


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
        self._homonuclear_double_network = {}

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
            0.5
            * (10**-7)
            * hbar
            * origin.gamma
            * source.gamma
            * (1 - 3 * cos**2)
            / dx**3
        )

    @staticmethod
    def dot_product(avec, bvec):  # helper method, necessary for below
        dot = 0
        for i in range(3):
            dot += avec[i] * bvec[i]
        return dot

    @staticmethod
    def homonuclear_coupling(origin, source, bdir=[0, 0, 1]):
        """
        Compute the dipolar coupling strength for a pair of `Atom`s in rad/s,
        `origin` and `source` where the direction of magnetic field is not necessarily aligned with z
        """

        bdir = bdir / norm(bdir)
        p1 = origin.position
        p2 = source.position
        hbar = 1.05457 * 10 ** (-34)  # J s / rad
        r = (p2 - p1) * 10 ** (-10)
        dx = norm(r)
        rhat = r / dx
        cos = Disorder.dot_product(-rhat, bdir)
        return (
            0.5
            * (1e-7)
            * hbar
            * origin.gamma
            * source.gamma
            * (1 - 3 * cos**2)
            / dx**3
        )  # based on 1.34 in master thesis, eqn 7 in Cory

    def spin_diffusion_coeff(self, origin, bdir, a, shell_radius=-1):
        """
        Based on a formula (eqn 29) by Cory 2005 for 0th order magnetic spin diffusion constant-
        Further assumes that the distribution in homonuclear spin diffusion is roughly Gaussian (Cory eqn 38)
        Note a is lattice size
        """
        bdir = bdir / norm(bdir)
        D = 0
        F = (
            0.5
            * (
                2
                * pi
                / self.homonuclear_single_variance_estimate(origin, bdir, 5)
            )
            ** 0.5
        )  # uniform F estimation
        if shell_radius == -1:
            shell_radius = self.shell_radius
        lattice = self.crystal.generate_lattice(shell_radius)
        for atompos in lattice:
            d = norm(atompos.position - origin.position)
            if atompos.name == origin.name and not d <= 0.1:
                bijsquared = (
                    Disorder.homonuclear_coupling(origin, atompos, bdir) ** 2
                )

                zijsquared = (
                    Disorder.dot_product(
                        (atompos.position - origin.position) * 1e-10, bdir
                    )
                    ** 2  # Dot product gives transverse component lying along b field
                )
                if d <= 1.5 * a:
                    # Inside 1.5 shells, calculate F with eqn 38 of Cory
                    Fij = (
                        0.5
                        * (
                            2
                            * pi
                            / self.homonuclear_double_variance_estimate(
                                origin, atompos, bdir, 3
                            )
                        )
                        ** 0.5
                    )
                    # print(Fij)
                else:
                    Fij = F  # Outside 1.5 shells, use uniform F

                # Fij = self.diffusion_fij(origin, atompos, bdir)
                D += bijsquared * zijsquared * Fij

        return (
            1 / 4 * D * 1e4
        )  # in cm^2/s (need to divide by 2pi for radians?)

    def spin_diffusion_second_order(self, origin, bdir, a, shell_radius=-1):
        r"""
        Calculate second order contribution to spin diffusion coeff as given in Cory eqn 31
        Assumes uniform F throughout
        """
        bdir = bdir / norm(bdir)
        D2 = 0
        term1 = 0
        term2 = 0
        term3 = 0
        F = (
            0.5
            * (
                2
                * pi
                / self.homonuclear_single_variance_estimate(origin, bdir, 5)
            )
            ** 0.5
        )
        if shell_radius == -1:
            shell_radius = self.shell_radius
        lattice = self.crystal.generate_lattice(shell_radius)
        for atompos in lattice:
            d = norm(atompos.position - origin.position)
            if atompos.name == origin.name and not d <= 0.1:
                bijsquared = (
                    Disorder.homonuclear_coupling(origin, atompos, bdir) ** 2
                )

                zijsquared = (
                    Disorder.dot_product(
                        (atompos.position - origin.position) * 1e-10, bdir
                    )
                    ** 2
                )
                term1 += bijsquared**2 * zijsquared
                term2 += bijsquared
                term3 += bijsquared * zijsquared

                # Fij = self.diffusion_fij(origin, atompos, bdir)
        D2 = F**3 / 48 * (2 * term1 - term2 * term3)

        return D2 * 1e4  # (need to divide by 2pi for radians?)

    def max_remainder(self, n, origin, bdir, a):
        r"""
        Upper bound on error to add to spin diffusion coefficient estimation (0th order) within n shells
        """
        bdir = bdir / norm(bdir)
        F = (
            0.5
            * (
                2
                * pi
                / self.homonuclear_single_variance_estimate(origin, bdir, 5)
            )
            ** 0.5
        )
        # print(F)
        hbar = 1.05457 * 10 ** (-34)
        return (
            8
            * 24
            / 4
            * 1e4
            * (
                1
                / (2 * pi)
                * 1
                / n
                * F
                * hbar**2
                * origin.gamma**4
                * 4
                * a**-4
                * 1e40
                * 1e-14
                / (4)
            )
        )

    def average_remainder(self, n, origin, bdir, a):
        r"""
        Averaged bound on error to add to spin diffusion coefficient estimation (0th order) within n shells using integration
        """
        bdir = bdir / norm(bdir)
        F = (
            0.5
            * (
                2
                * pi
                / self.homonuclear_single_variance_estimate(origin, bdir, 5)
            )
            ** 0.5
        )
        # print(F)
        hbar = 1.05457 * 10 ** (-34)
        return (
            8
            * 8
            * 3
            / 4
            * 1e4
            * (
                1
                / n
                * F
                * hbar**2
                * origin.gamma**4
                * 17
                / 16  # 17/16 = <cos^2(1-3*cos^2)^2> (*35/8 for cos^1)
                * a**-4
                * 1e40
                * 1e-14
                / (4)  # (mu0/8pi)^2
            )
        )

    def tot_estimated_spin_diffusion_coeff(self, origin, bdir, a):
        # 3 level estimation on 0th order (variable Fij within 2 shells, constant F within 6 shells, integrated 7 shells and out)
        # Added 2nd order contribution within 6 shells
        return (
            self.spin_diffusion_coeff(origin, bdir, a, 6)
            + self.average_remainder(7, origin, bdir, a)
            + self.spin_diffusion_second_order(origin, bdir, a, 6)
        )

    r"""
    def OLD_spin_diffusion_coeff_parallel(
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
                        * Disorder.dot_product(
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

    def OLD_spin_diffusion_coeff_perpendicular(
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
        """

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
        if (origin_1, origin_2, plus) in self._double_network.keys():
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

    def homonuclear_double_get_network(
        self, origin_1, origin_2, bdir=[0, 0, 1], plus=False
    ):
        bdir = bdir / norm(bdir)
        if (origin_1, origin_2, bdir, plus) in self._network.keys():
            return self._homonuclear_double_network[
                (origin_1, origin_2, bdir, plus)
            ]
        return self.homonuclear_double_generate_network(
            origin_1, origin_2, bdir, plus
        )

    def homonuclear_double_generate_network(
        self, origin_1, origin_2, bdir=[0, 0, 1], plus=False, shell_radius=-1
    ):
        r"""
        Note that origin_1, origin_2 need to be the same atom for this to work.
        Assign coupling coefficients of b_1j - b_2j to each j not equal to 1 or 2
        """
        assert origin_1.name == origin_2.name
        bdir = bdir / norm(bdir)
        if shell_radius == -1:
            shell_radius = self.shell_radius
        if plus == True:
            a = 1
        else:
            a = -1
        lattice = self.crystal.generate_lattice(shell_radius)
        network = []
        for atompos in lattice:
            if atompos.name == origin_1.name:
                if (
                    not norm(atompos.position - origin_1.position) <= 0.1
                    and not norm(atompos.position - origin_2.position) <= 0.1
                ):
                    atompos.coupling = Disorder.homonuclear_coupling(
                        origin_1, atompos, bdir
                    ) + a * Disorder.homonuclear_coupling(
                        origin_2, atompos, bdir
                    )
                    network.append(atompos)
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

    def homonuclear_double_mean_field_calc(
        self, origin_1, origin_2, bdir=[0, 0, 1], regen=False
    ):
        """
        Calculate sum or difference of disorder contribution to the local field at two specific atoms
        Randomized sum of discretized spin orientations times heteronuclear coupling strengths
        """

        if regen:
            double_crystal = self.homonuclear_double_generate_network(
                origin_1, origin_2, bdir
            )
        else:
            double_crystal = self.homonuclear_double_get_network(
                origin_1, origin_2, bdir
            )

        config = Disorder.get_rand_config(double_crystal)
        return sum(
            [
                spin.coupling * orien * 2
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

    def homonuclear_double_variance_estimate(
        self, origin_1, origin_2, bdir=[0, 0, 1], plus=False, shell_radius=-1
    ):
        """
        Gives variance of mean field calculated at origin
        Assuming Gaussian distribution of mean field used in calculating F_ij
        """
        crystal = self.homonuclear_double_generate_network(
            origin_1, origin_2, bdir, plus, shell_radius
        )
        return sum([(spin.coupling**2) for spin in crystal])

    def homonuclear_single_variance_estimate(
        self, origin, bdir=[0, 0, 1], shell_radius=-1
    ):
        r"""
        Used in estimating a uniform F_ij constant as defined in Cory 2005
        Multiplied by two at the end because we really want sum (b_i - b_j)^2 where b_i, b_j have non-overlapping support
        """
        variance = 0
        if shell_radius == -1:
            shell_radius = self.shell_radius
        lattice = self.crystal.generate_lattice(shell_radius)
        for atompos in lattice:
            if atompos.name == origin.name:
                if not norm(atompos.position - origin.position) <= 0.1:
                    variance += (
                        Disorder.homonuclear_coupling(origin, atompos, bdir)
                        ** 2
                    )
        return 2 * variance

    """
    def diffusion_fij(self, origin_1, origin_2, bdir=[0, 0, 1]):
        crystal = self.homonuclear_double_generate_network(
            origin_1, origin_2, bdir
        )
        return sum([sinc(spin.coupling) for spin in crystal])
    """

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

    def homonuclear_double_simulation(
        self,
        origin_1,
        origin_2,
        trials,
        bdir=[0, 0, 1],
        filename="spin_diffusion_bij_distro.dat",
        regen=False,
    ):
        """
        Monte-carlo simulation of mean field at orig_atom
        given contributing atoms in atomlis according to ij coupling
        Regen boolean says whether all trials have one fixing of isotopes (faster)
        or vary isotopes/atoms at lattice sites (slower)
        """
        try:
            my_distro_minus = load(open(filename, "rb"))
        except (OSError, IOError) as e:
            my_distro_minus = zeros(trials)
            for idx in range(trials):
                my_distro_minus[idx] = self.homonuclear_double_mean_field_calc(
                    origin_1, origin_2, bdir, regen
                )
            with open(filename, "wb") as fil:
                dump(my_distro_minus, fil)
        return my_distro_minus

    @property
    def crystal(self):
        return self._crystal

    @property
    def shell_radius(self):
        return self._shell_radius
