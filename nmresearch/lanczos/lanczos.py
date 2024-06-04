from numpy import sqrt, round, array, real, arange, sum, zeros
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from timeit import default_timer as timer
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from nmresearch.lanczos.utils import basis_vec
from nmresearch.lanczos.op_basis import vec, devec


class Lanczos:

    def __init__(self, propegator, op):
        self.prop = propegator
        self.op = op
        self.krylov_basis = None
        self.lanczos_coef = None
        self.e0 = None
        self.L = None
        self.B = None
        self.tbc_vals = None
        self.k_complexity = None

    def compute_lanczos_fast(self, use_ham=True, max_iter=50, tol=1e-10):
        if use_ham:
            self._compute_lanczos_fast_ham(max_iter=max_iter, tol=tol)
        else:
            self._compute_lanczos_fast_liouv(max_iter=max_iter, tol=tol)

    def _compute_lanczos_fast_ham(self, max_iter, tol):
        start = timer()
        end = 0

        krylov_basis = []
        lanczos_coef = []
        O0 = self.op
        ham = self.prop
        A1 = ham @ O0 - O0 @ ham
        b1 = sqrt((vec(A1).T.conj() @ vec(A1)).data[0])
        O1 = (1 / b1) * A1
        krylov_basis.append(O0)
        krylov_basis.append(O1)
        lanczos_coef.append(b1)

        for j in range(2, max_iter, 1):
            Aj = (ham @ krylov_basis[j - 1] - krylov_basis[j - 1] @ ham) - lanczos_coef[
                j - 2
            ] * krylov_basis[j - 2]
            bj = round(sqrt((vec(Aj).T.conj() @ vec(Aj)).data[0]), 16)
            if bj < tol:
                print("Lanczos Algorithm terminated at a 0-vector")
                break

            Oj = (1 / bj) * Aj
            krylov_basis.append(Oj)
            lanczos_coef.append(bj)

        self.lanczos_coef = lanczos_coef
        self.krylov_basis = krylov_basis
        end = timer()
        print("Computation took " + str(end - start) + " sec")

    def _compute_lanczos_fast_liouv(self, max_iter=50, tol=1e-10):
        r"""
        Computes lanczos method up to `max_iter` steps or until the produced vector is null to within
        tolerance `tol`. This method uses modified Gram-Schmidt to orthogonalize. This method is stable,
        but it can be slow and memory intensive for large system sizes, since all krylov vectors must be
        stored and we repeatedly re-orthogonalize
        """
        start = timer()
        end = 0

        krylov_basis = []
        lanczos_coef = []
        O0 = self.op.todense()
        L = self.prop
        A1 = L @ O0
        b1 = sqrt((A1.T.conj() @ A1)[0, 0])
        O1 = (1 / b1) * A1
        krylov_basis.append(O0)
        krylov_basis.append(O1)
        lanczos_coef.append(b1)

        for j in range(2, max_iter, 1):
            Aj = L @ krylov_basis[j - 1] - lanczos_coef[j - 2] * krylov_basis[j - 2]
            bj = round(sqrt((Aj.T.conj() @ Aj)[0, 0]), 16)
            if bj < tol:
                print("Lanczos Algorithm terminated at a 0-vector")
                break

            Oj = (1 / bj) * Aj
            krylov_basis.append(Oj)
            lanczos_coef.append(bj)

        self.lanczos_coef = lanczos_coef
        self.krylov_basis = krylov_basis
        end = timer()
        print("Computation took " + str(end - start) + " sec")

    def compute_lanczos_FRO(self, max_iter=50, tol=1e-10):
        r"""
        Computes lanczos method up to `max_iter` steps or until the produced vector is null to within
        tolerance `tol`. This method uses modified Gram-Schmidt to orthogonalize. This method is stable,
        but it can be slow and memory intensive for large system sizes, since all krylov vectors must be
        stored and we repeatedly re-orthogonalize
        """
        start = timer()
        end = 0

        krylov_basis = []
        lanczos_coef = []
        O0 = self.op.todense()
        L = self.prop
        A1 = L @ O0
        b1 = sqrt((A1.T.conj() @ A1)[0, 0])
        O1 = (1 / b1) * A1
        krylov_basis.append(O0)
        krylov_basis.append(O1)
        lanczos_coef.append(b1)

        for j in range(2, max_iter, 1):
            Aj = L @ krylov_basis[j - 1] - lanczos_coef[j - 2] * krylov_basis[j - 2]
            for i in range(j):
                Aj = Aj - (krylov_basis[i].T.conj() @ Aj)[0, 0] * krylov_basis[i]
            bj = round(sqrt((Aj.T.conj() @ Aj)[0, 0]), 16)
            if bj < tol:
                print("Lanczos Algorithm terminated at a 0-vector")
                break

            Oj = (1 / bj) * Aj
            krylov_basis.append(Oj)
            lanczos_coef.append(bj)

        self.lanczos_coef = lanczos_coef
        self.krylov_basis = krylov_basis
        end = timer()
        print("Computation took " + str(end - start) + " sec")

    def auto_correlation(self, times):
        r"""
        Uses the lanczos coefficients to compute the autocorrelation with help of Scipy to perform
        the matrix exponentials in the Krylov space. This method appears to only compute a single propegator,
        and so its accuracy is dependent on the timestep, which can be pretty small since L should be fairly
        low dimension compared to the entire Hilbert space.
        """
        if self.lanczos_coef is None:
            print("Lanczos coefficients must be computed!")
            return
        if self.L is None:
            self.L = diags([self.lanczos_coef, self.lanczos_coef], [1, -1])
        if self.e0 is None:
            self.e0 = basis_vec(self.L.shape[0], 0)

        return (
            expm_multiply(
                -1j * self.L,
                self.e0,
                start=times[0],
                stop=times[-1],
                num=len(times),
                endpoint=True,
            )
            @ self.e0
        )

    def auto_correlation_ED(self, times):
        r"""
        Uses Scipy sparse matrix techniques to compute the autocorrelation. Recomputes the exponential
        for each timestep desired, minimizing integration error, but increasing computation time.
        """
        return array(
            [self.op @ expm_multiply(-1j * self.prop * t, self.op) for t in times]
        )

    def auto_correlation_ED_fast(self, times):
        r"""
        Uses Scipy sparse matrix exponential matrix solver to computer the autocorrelation.
        This method appears to only compute a single propegator, and so its accuracy is dependent on
        the timestep, for some reason.
        """
        return (
            expm_multiply(
                -1j * self.prop,
                self.op,
                start=times[0],
                stop=times[-1],
                num=len(times),
                endpoint=True,
            )
            @ self.op
        )

    def orthogonality_test(self):
        r"""
        check the orthogonality of the krylov basis and generate a heatmap visualization of the resulting
        matrix.

        If the vectors have no overlapping indices, this method will fail since it will produce a zero
        dimensional scalar, which is annoying to deal with.
        """
        if self.krylov_basis is None:
            print("Lanczos coefficients must be computed!")
            return
        n = len(self.krylov_basis)
        orth_test = array(
            [
                [
                    (self.krylov_basis[i].T.conj() @ self.krylov_basis[j]).data[0]
                    for i in range(n)
                ]
                for j in range(n)
            ]
        )
        ax = sb.heatmap(real(orth_test))
        ax.invert_yaxis()
        plt.xlabel("index 1")
        plt.ylabel("index 2")
        plt.title("Krylov Orthogonality test")
        plt.show()

    def tight_binding_complexity(
        self, start, end, max_step=0.05, upper_limit=20, plot=True
    ):
        r"""
        Solves the tight binding model seeded from the lanczos coefficients. This should be
        equivalent to computing the evolution of the initial operator in the Krylov basis,
        up to a power of the imaginary unit.
        """
        if self.lanczos_coef is None:
            print("Lanczos coefficients must be computed!")
            return
        if self.B is None:
            self.B = diags(
                [-1 * array(self.lanczos_coef), array(self.lanczos_coef)], [1, -1]
            )
        if self.e0 is None:
            self.e0 = basis_vec(self.L.shape[0], 0)

        B = self.B

        def f(t, y):
            return B @ y

        self.tbc_vals = solve_ivp(f, (start, end), self.e0, max_step=max_step)

        if plot:
            fig, ax = plt.subplots()
            c = ax.pcolormesh(
                self.tbc_vals["t"],
                list(range(0, self.B.shape[0], 1)),
                abs(self.tbc_vals["y"]) ** 2,
            )
            ax.set_title("Tight-Binding Evolution")
            # set the limits of the plot to the limits of the data
            ax.axis([0, self.tbc_vals["t"][-1], 0, upper_limit])
            ax.set_ylabel("Complexity")
            ax.set_xlabel("Time")
            fig.colorbar(c, ax=ax)
            plt.show()

    def compute_krylov_complexity(self, plot=True):
        r"""
        Uses the results from the tight binding simulation to compute the Krylov complexity, which
        is equal to the expected site occupation of the virtual tight-binding particle. For a generically
        scrambling thermodynamic Hamiltonian, this is expected to grow exponentially at a rate which bounds
        the Lyapunov exponent.
        """
        if self.tbc_vals is None:
            print("Solve the semi-infinite tight-binding model first")
            return
        self.k_complexity = [
            sum((arange(0, self.B.shape[0], 1)) * abs(self.tbc_vals["y"][:, idx]) ** 2)
            for idx in range(len(self.tbc_vals["t"]))
        ]

        if plot:
            plt.plot(self.tbc_vals["t"], self.k_complexity)
            plt.xlabel("Time")
            plt.ylabel("K-Complexity")
            plt.show()
