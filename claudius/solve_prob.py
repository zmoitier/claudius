from numpy import arange, expand_dims, pi, size, sqrt, zeros
from numpy.linalg import solve
from scipy.special import jv, jvp, spherical_jn

from claudius import Solution


def _plane_wave(dim, pde, k):
    if dim == 2:
        return (lambda m, r: jv(m, k * r), lambda m, r: k * jvp(m, k * r))

    if dim == 3:
        if pde.startswith("H"):
            return (
                lambda l, r: 1j ** l
                * sqrt(4 * pi * (2 * l + 1))
                * spherical_jn(l, k * r),
                lambda l, r: k
                * 1j ** l
                * sqrt(4 * pi * (2 * l + 1))
                * spherical_jn(l, k * r, derivative=True),
            )

        if pde.startswith("M"):
            return None

    return None


def _calc_mat(shape, m, ρ, εμ, fun, fun_der, shift):
    A = zeros(shape, dtype=complex)

    for n, (
        ρn,
        (εl, _),
        (εr, _),
        (fl0, gl0),
        (fl1, gl1),
        (fr0, gr0),
        (fr1, gr1),
    ) in enumerate(
        zip(
            ρ[1:-1],
            εμ[shift:-1],
            εμ[(shift + 1) :],
            fun[shift:-2],
            fun_der[shift:-2],
            fun[(shift + 1) : -1],
            fun_der[(shift + 1) : -1],
        )
    ):
        A[:, 2 * n, 2 * n] = fl0(m, ρn)
        A[:, 2 * n, 2 * n + 1] = gl0(m, ρn)
        A[:, 2 * n, 2 * n + 2] = -fr0(m, ρn)
        A[:, 2 * n, 2 * n + 3] = -gr0(m, ρn)

        A[:, 2 * n + 1, 2 * n] = fl1(m, ρn) / εl(ρn)
        A[:, 2 * n + 1, 2 * n + 1] = gl1(m, ρn) / εl(ρn)
        A[:, 2 * n + 1, 2 * n + 2] = -fr1(m, ρn) / εr(ρn)
        A[:, 2 * n + 1, 2 * n + 3] = -gr1(m, ρn) / εr(ρn)

    A[:, -2, -3] = fun[-2][0](m, ρ[-1])
    A[:, -2, -2] = fun[-2][1](m, ρ[-1])
    A[:, -2, -1] = -fun[-1](m, ρ[-1])

    A[:, -1, -3] = fun_der[-2][0](m, ρ[-1]) / εμ[-1][0](ρ[-1])
    A[:, -1, -2] = fun_der[-2][1](m, ρ[-1]) / εμ[-1][0](ρ[-1])
    A[:, -1, -1] = -fun_der[-1](m, ρ[-1])

    return A


def solve_prob(prob, M):
    ρ = prob.radii
    εμ = prob.eps_mu
    k = prob.wavenum
    fun = prob.fun
    fun_der = prob.fun_der

    fj0, fj1 = _plane_wave(prob.dim, prob.pde, k)

    N = len(prob.radii) - 1  # Number of layers
    m = arange(M + 1)

    if prob.inn_bdy.startswith("P"):
        if N == 0:
            A = zeros((M + 1, 2, 2), dtype=complex)

            A[:, 0, 0] = fun[0](m, ρ[0])
            A[:, 0, 1] = -fun[1](m, ρ[0])

            A[:, 1, 0] = fun_der[0](m, ρ[0]) / εμ[0][0](ρ[0])
            A[:, 1, 1] = -fun_der[1](m, ρ[0])

        else:
            nbi = 2 * len(ρ)
            A = zeros((M + 1, nbi, nbi), dtype=complex)

            A[:, 0, 0] = fun[0](m, ρ[0])
            A[:, 0, 1] = -fun[1][0](m, ρ[0])
            A[:, 0, 2] = -fun[1][1](m, ρ[0])

            A[:, 1, 0] = fun_der[0](m, ρ[0]) / εμ[0][0](ρ[0])
            A[:, 1, 1] = -fun_der[1][0](m, ρ[0]) / εμ[1][0](ρ[0])
            A[:, 1, 2] = -fun_der[1][1](m, ρ[0]) / εμ[1][0](ρ[0])

            A[:, 2:, 1:] = _calc_mat(
                (M + 1, nbi - 2, nbi - 1), m, ρ, εμ, fun, fun_der, 1
            )

    else:
        if N == 0:
            if prob.inn_bdy.startswith("D"):
                return Solution(
                    *prob, expand_dims(-fj0(m, ρ[0]) / fun[0](m, ρ[0]), axis=1)
                )
            else:
                return Solution(
                    *prob, expand_dims(-fj1(m, ρ[0]) / fun_der[0](m, ρ[0]), axis=1)
                )

        else:
            nbi = 2 * len(ρ) - 1
            A = zeros((M + 1, nbi, nbi), dtype=complex)

            if prob.inn_bdy.startswith("D"):
                A[:, 0, 0] = fun[0][0](m, ρ[0])
                A[:, 0, 1] = fun[0][1](m, ρ[0])
            else:
                A[:, 0, 0] = fun_der[0][0](m, ρ[0])
                A[:, 0, 1] = fun_der[0][1](m, ρ[0])

            A[:, 1:, :] = _calc_mat((M + 1, nbi - 1, nbi), m, ρ, εμ, fun, fun_der, 0)

    F = zeros((M + 1, size(A, 1)), dtype=complex)
    F[:, -2] = fj0(m, ρ[-1])
    F[:, -1] = fj1(m, ρ[-1])

    return Solution(*prob, solve(A, F))
