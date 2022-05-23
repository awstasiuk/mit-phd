from fermion.math import Math as fm
from fermion.operator import Operator
from fermion.majorana import PauliString, MajoranaString

import numpy as np
from pfapack import pfaffian as pf
from math import e, pi, ceil


class Unitary:
    r"""
    A class desciribing unitary time evolution of a fermionic operator. Owns a quadratic
    `Operator` serving as the system Hamiltonian.
    """

    def __init__(self, ham, dt, tmax):
        r"""
        initialize the object, by preparing a time mesh and diagonalizing the hamiltonian
        """
        if not ham.is_quadratic:
            raise ValueError("Invalid Hamiltonian, must be quadratic")
        self._ham = ham

        self._dt = dt
        self._tmax = tmax
        self._n_steps = ceil(tmax / dt)
        self._t = [i * dt for i in range(self._n_steps)]

        diag, T = ham.jordan_wigner()
        self._eigen = 2 * diag.coef[2].diagonal(-ham.n_fermion)
        self._T = T
        self._G = T[0 : ham.n_fermion, 0 : ham.n_fermion]
        self._H = T[0 : ham.n_fermion, ham.n_fermion : 2 * ham.n_fermion]

        otoc_cache = {}
        for mu in PauliString.P_CHARS:
            for nu in PauliString.P_CHARS:
                otoc_cache[mu + nu] = {}

        self._cache = {"U1": {}, "U2": {}, "Cm": {}, "OTOC": otoc_cache}

    def U1(self, t):
        r"""
        the N*N operator representing time evolution of the annihilation operators in the
        original fermionic basis
        """
        u1 = self._cache["U1"].get(t)
        if u1 is not None:
            return u1
        u1 = (
            fm.adj(self.G) @ fm.exp_diag(self.eigen, t) @ self.G
            + fm.adj(self.H) @ fm.exp_diag(self.eigen, -t) @ self.H
        )
        self._cache["U1"][t] = u1
        return u1

    def U2(self, t):
        r"""
        the N*N operator representing time evolution of the creations operators in the
        original fermionic basis.
        """
        u2 = self._cache["U2"].get(t)
        if u2 is not None:
            return u2
        u2 = (
            fm.adj(self.G) @ fm.exp_diag(self.eigen, t) @ self.H
            + fm.adj(self.H) @ fm.exp_diag(self.eigen, -t) @ self.G
        )
        self._cache["U2"][t] = u2
        return u2

    def U(self, t):
        r"""
        the 2N*2N operator representing time evolution in the original fermionic basis
        a(t) = u1(t)*a + u2(t)*adag
        adag(t) = u1(t).conj * adag + u2(t).conj * a
        """
        u1 = self.U1(t)
        u2 = self.U2(t)
        return np.block([[u1, u2], [np.conj(u2), np.conj(u1)]])

    def local_zz(self, idx1, idx2):
        r"""
        Computes the infinite temperature two point correlator for local Z observables at
        indices `idx1` and `idx2`, which can be written as the expectation value of
        `<Z_idx1(t) Z_idx2>`.

        `idx1` and `idx2` must be integers in the range `[0, n_fermion-1]`.

        returns a list of points evaluated on this instance's time mesh.
        """
        return [
            (np.abs(self.U1(t)[idx1, idx2]) ** 2 - np.abs(self.U2(t)[idx1, idx2]) ** 2)
            for t in self.t
        ]

    def global_zz(self):
        r"""
        Computes the infinite temperature two point correlator for local Z observables at
        indices `idx1` and `idx2`, which can be written as the expectation value of
        `<Z(t) Z>`.

        returns a list of points evaluated on this instance's time mesh.
        """
        return [
            (
                np.linalg.norm(self.U1(t), "fro") ** 2
                - np.linalg.norm(self.U2(t), "fro") ** 2
            )
            for t in self.t
        ]

    def global_OTOC(self, mu, nu, use_true=False):
        r"""
        Computes the infinite temperature OTOC for global magnetization operators,

        F_{mu nu}(t) = - < [J_mu(t), J_nu]^2 >

        on this instance's time mesh.
        """
        if use_true:
            return [
                np.sum(self.true_OTOC_tensor(mu, nu, t)) / self.n_fermion
                for t in self.t
            ]
        return [np.sum(self.centered_OTOC_tensor(mu, nu, t)) for t in self.t]

    def OTOC_tensor_elem(self, mu, nu, a, b, c, d, t):
        r"""
        Returns the tensory element of the truly local OTOC computed under the assumption of
        translational symmetry

        T^{abcd} = - < [S_mu^{(a)}(t), S_nu^{(b)}] * [S_mu^{(c)}(t), S_nu^{(d)}] >,
        """
        center = int(self.n_fermion / 2)
        delta = a - center
        return self.centered_OTOC_tensor(mu, nu, t)[b + delta, c + delta, d + delta]

    def centered_OTOC_tensor(self, mu, nu, t, use_mirror=False):
        r"""
        Returns the rank 3 centered OTOC tensor for two magnetization directions such that
        mu, nu \in {"X", "Y", "Z"}, corresponding to the relevant local paulis in the OTOC
        tensor,

        T^{abc} = - < [S_mu^{(N/2)}(t), S_nu^{(a)}] * [S_mu^{(b)}(t), S_nu^{(c)}] >,

        where a,b,c run over all spin sites, 1,2,...,N. If `use_mirror is `True`, a will only
        run over one half of the chain, and we use mirror symmetry to get the other half, reducing
        the computational cost by a factor of two.

        Encoded within contractions of this tensor are global, local, and semi-local OTOCs commonly
        seen in the literature, so long as the underlying hamiltonian is translationally invariant.

        non-translationally invariant hamiltonians should use an uncentered OTOC tensor, which
        is rank 4.
        """
        if t in self._cache["OTOC"][mu + nu]:
            return self._cache["OTOC"][mu + nu][t]
        n = self.n_fermion
        center = int(n / 2)
        otoc = np.zeros((n, n, n))

        for a in range(self.n_fermion):
            for b in range(self.n_fermion):
                for c in range(self.n_fermion):
                    term1 = PauliString(
                        [mu, nu, mu, nu],
                        [center, a, b, c],
                        [True, False, True, False],
                    )

                    term2 = PauliString(
                        [nu, mu, mu, nu],
                        [a, center, b, c],
                        [False, True, True, False],
                    )
                    otoc[a, b, c] = -2 * np.real(
                        self._pfaffian_helper(term1, t)
                        - self._pfaffian_helper(term2, t)
                    )
                    # if all(n - idx > 0 for idx in [a - 1, b - 1, c - 1]):
                    #    otoc[n - a - 1, n - b - 1, n - c - 1] = otoc_elem

        self._cache["OTOC"][mu + nu][t] = otoc
        return otoc

    def true_OTOC_tensor(self, mu, nu, t):
        r"""
        Returns the rank 4 OTOC tensor for two magnetization directions such that
        mu, nu \in {"X", "Y", "Z"}, corresponding to the relevant local paulis in the OTOC
        tensor,

        T^{abc} = - < [S_mu^{(a)}(t), S_nu^{(b)}] * [S_mu^{(c)}(t), S_nu^{(d)}] >,

        where a,b,c,d run over all spin sites, 1,2,...,N.

        Encoded within contractions of this tensor are global, local, and semi-local OTOCs commonly
        seen in the literature, so long as the underlying hamiltonian is translationally invariant.

        This is for non-translationally invariant hamiltonians, and is not very efficient
        """
        n = self.n_fermion
        center = int(n / 2)
        otoc = np.zeros((n, n, n, n))

        for a in range(self.n_fermion):
            for b in range(self.n_fermion):
                for c in range(self.n_fermion):
                    for d in range(self.n_fermion):
                        term1 = PauliString(
                            [mu, nu, mu, nu],
                            [a, b, c, d],
                            [True, False, True, False],
                        )

                        term2 = PauliString(
                            [nu, mu, mu, nu],
                            [b, a, c, d],
                            [False, True, True, False],
                        )
                        otoc[a, b, c, d] = -2 * np.real(
                            self._pfaffian_helper(term1, t)
                            - self._pfaffian_helper(term2, t)
                        )

        return otoc

    def pauli_string_expectation(self, pauli_string):
        r"""
        Compute the infinite temperature expectation value for the (possibly) out of time
        order local pauli string of operators in ``pauli_string'', on this instance's time mesh.
        We do this by  writting the pauli string as a product of majorana fermion terms,
        and computing its expectation value using Wick's theorem and the pfaffian matrix method,
        so only a submatrix of two body correlations within the many-body string need to be computed.
        """
        return [self._pfaffian_helper(pauli_string, t) for t in self.t]

    def _pfaffian_helper(self, pauli_string, t):
        r"""
        Actually do the numerical work of computing the reduced 2-body correlation matrix and
        taking its pfaffian to compute the pauli string expectation value.
        """
        # transform pauli string into majorana fermion representation
        maj_str = MajoranaString.from_pauli_string(pauli_string)

        n = len(maj_str)
        if n % 2 == 1:
            return 0

        # compute all 2-body majorana expectation values ahead of time
        corr = self.majorana_two_body(t)

        # generate the skew-symmetric matrix encoding the total expecation value to be
        # extracted via pfaffian
        pf_mat = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            for j in range(i + 1, n):
                idx1 = maj_str.sites[i] + self.n_fermion * int(
                    maj_str.majoranas[i] == "B"
                )
                idx2 = maj_str.sites[j] + self.n_fermion * int(
                    maj_str.majoranas[j] == "B"
                )

                if maj_str.evo_bools[i] == maj_str.evo_bools[j]:
                    # Case where both operators are evolved or neither are evolved (static case)
                    pf_mat[i, j] = 0.5 * int(idx1 == idx2)
                else:
                    # case where one only one operator is evolved (dynamic case)
                    if maj_str.evo_bools[i]:
                        # subcase where the first operator is evolved
                        pf_mat[i, j] = corr[idx1, idx2]
                    else:
                        # subcase where the second operator is evolved
                        pre = 1
                        if (idx1 < self.n_fermion and idx2 >= self.n_fermion) or (
                            idx1 >= self.n_fermion and idx2 < self.n_fermion
                        ):
                            # fix ordering phase for upper right and lower left blocks
                            pre = -1
                        pf_mat[i, j] = pre * corr[idx1, idx2]

        # anti-symmetrize
        pf_mat = pf_mat - pf_mat.transpose()

        # we need to chop here because the pfaffian package is unstable with small values,
        # it checks float equality to 0 instead of `allclose`
        return maj_str.pre_factor * pf.pfaffian(fm.chop(pf_mat))

    def majorana_two_body(self, t):
        r"""
        Computes and returns the two body correlation matrix for a set of 2n Majorana
        fermions evolving under system hamiltonian at time t. Ai = isqrt2*(ai+ci),
        Bi = -1j*isqrt2*(ai-ci)
        """
        if t in self._cache["Cm"]:
            return self._cache["Cm"][t]
        T = np.kron([[1, 1], [-1j, 1j]] / np.sqrt(2), np.eye(self.n_fermion))
        Ct = 0.5 * (T @ self.U(t) @ fm.adj(T))
        self._cache["Cm"][t] = Ct
        return Ct

    def evolve_op(self, op, t):
        r"""
        Propagates the fermionic operator `op` to a future time `t` via the Hamiltonian.

        returns a :py:class:`fermion.Operator`
        """
        if self.hamiltonian.n_fermion != op.n_fermion:
            raise ValueError("Dimension mismatch!")

        coef = {}
        for k in op.components:
            if k == 0:
                coef[0] = op.coef[0]
            elif k == 1:
                coef[1] = self.U(t) @ op.coef[1]
            else:
                coef[k] = fm.tensor_change_of_basis(op.coef[k], self.U(t))

        return Operator(op.n_fermion, coef)

    def populate_cache(self):
        r"""
        pre-computes the time evolution blocks U1 and U2 and caches the results for each
        value in this instance's time mesh. This may be useful if these matrices are going
        to be used often and take a while to compute, assuming memory is cheaper than clocks
        and you don't want to populate the cache on the fly.
        """
        if len(self._cache) == 0:
            for t in self.t:
                self.u1(t)
                self.u2(t)
        else:
            print("Cache is already populated with at least one element.")

    def clear_cache(self):
        r"""
        clears the cache, possibly freeing up memory if needed.
        """
        for key in self._cache.keys():
            self._cache[key] = {}

    @property
    def hamiltonian(self):
        return self._ham

    @property
    def n_fermion(self):
        return self._ham.n_fermion

    @property
    def dt(self):
        return self._dt

    @property
    def tmax(self):
        return self._tmax

    @property
    def t(self):
        return self._t

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def eigen(self):
        r"""
        The eigenvalues of the quadratic form Hamiltonian, which represent fermion energies. If
        the model is non-interacting (quadratic), the spectrum of the Hamiltonian corresponds
        to the sum of all possible subsets of this list.
        """
        return self._eigen

    @property
    def G(self):
        r"""
        The upper left block of the change of basis matrix of the corresponding jordan-wigner
        transformation of this hamiltonian
        """
        return self._G

    @property
    def H(self):
        r"""
        The upper right block of the change of basis matrix of the corresponding jordan-wigner
        transformation of this hamiltonian
        """
        return self._H

    @property
    def T(self):
        r"""
        The change of basis matrix of the corresponding jordan-wigner transformation of this
        hamiltonian
        """
        return self._T
