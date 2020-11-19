from context import claudius
from numpy import (arange, linspace, meshgrid, ones_like, pi, size, sqrt,
                   zeros_like)
from numpy.random import uniform
from numpy.testing import assert_allclose

from claudius.Helmholtz_3d import create_problem_cst


def coeff_serie(l):
    return 1j ** l * sqrt(4 * pi * (2 * l + 1))


def _sol(inn_bdy, radii_list, εμ_list, k):
    pb_list = [
        create_problem_cst(inn_bdy, radii, εμ, k)
        for radii, εμ in zip(radii_list, εμ_list)
    ]

    M = claudius.trunc_H3d(k, 3)

    return [claudius.solve_prob(pb, M) for pb in pb_list]


class TestImpenetrable:
    def test_dirichlet(self):
        for k in uniform(0.5, 2, 8):
            sol0, sol1 = _sol("Dirichlet", [(1,), (1, 2)], [(), ((1, 1),)], k)
            c0, c1 = sol0.coeff, sol1.coeff

            cl = coeff_serie(arange(size(c0, 0)))

            assert assert_allclose(c1[:, 2], c0[:, 0], rtol=1e-6) is None
            assert assert_allclose(c1[:, 0], cl + c0[:, 0], rtol=1e-6) is None
            assert assert_allclose(c1[:, 1], 1j * c0[:, 0], rtol=1e-6) is None

            r, t, p = (
                linspace(1, 3, num=8),
                linspace(0, 2 * pi, num=8, endpoint=False),
                linspace(0, pi, num=6)[1:-1],
            )
            R, T, P = meshgrid(r, t, p)
            us0, ut0 = claudius.sc_field(sol0, R, T, P), claudius.tt_field(
                sol0, R, T, P
            )
            us1, ut1 = claudius.sc_field(sol1, R, T, P), claudius.tt_field(
                sol0, R, T, P
            )

            assert assert_allclose(us0, us1, atol=1e-6) is None
            assert assert_allclose(ut0, ut1, atol=1e-6) is None

    def test_neumann(self):
        for k in uniform(0.5, 2, 8):
            sol0, sol1 = _sol("Neumann", [(1,), (1, 2)], [(), ((1, 1),)], k)
            c0, c1 = sol0.coeff, sol1.coeff

            cl = coeff_serie(arange(size(c0, 0)))

            assert assert_allclose(c1[:, 2], c0[:, 0], rtol=1e-6) is None
            assert assert_allclose(c1[:, 0], cl + c0[:, 0], rtol=1e-6) is None
            assert assert_allclose(c1[:, 1], 1j * c0[:, 0], rtol=1e-6) is None

            r, t, p = (
                linspace(1, 3, num=8),
                linspace(0, 2 * pi, num=8, endpoint=False),
                linspace(0, pi, num=6)[1:-1],
            )
            R, T, P = meshgrid(r, t, p)
            us0, ut0 = claudius.sc_field(sol0, R, T, P), claudius.tt_field(
                sol0, R, T, P
            )
            us1, ut1 = claudius.sc_field(sol1, R, T, P), claudius.tt_field(
                sol0, R, T, P
            )

            assert assert_allclose(us0, us1, rtol=1e-6) is None
            assert assert_allclose(ut0, ut1, rtol=1e-6) is None


class TestPenetrable:
    def test_penetrable_coeff(self):
        for k in uniform(0.5, 2, 8):
            sol0, sol1 = _sol(
                "Penetrable", [(1,), (1, 2)], [((1, 1),), ((1, 1), (1, 1))], k
            )
            c0, c1 = sol0.coeff, sol1.coeff

            cl = coeff_serie(arange(size(c0, 0)))

            assert assert_allclose(c0[:, 0], cl, rtol=1e-6) is None
            assert assert_allclose(c0[:, 1], zeros_like(c0[:, 1]), atol=1e-15) is None

            assert assert_allclose(c1[:, 0], cl, rtol=1e-6) is None
            assert assert_allclose(c1[:, 1], cl, rtol=1e-6) is None
            assert assert_allclose(c1[:, 2], zeros_like(c1[:, 2]), atol=1e-15) is None
            assert assert_allclose(c1[:, 3], zeros_like(c1[:, 3]), atol=1e-15) is None

    def test_penetrable_field(self):
        for k in uniform(0.5, 2, 8):
            sol0, sol1 = _sol(
                "Penetrable", [(1,), (1, 2)], [((1, 1),), ((1, 1), (1, 1))], k
            )
            c0, c1 = sol0.coeff, sol1.coeff

            r, t, p = (
                linspace(1, 3, num=8),
                linspace(0, 2 * pi, num=8, endpoint=False),
                linspace(0, pi, num=6)[1:-1],
            )
            R, T, P = meshgrid(r, t, p)
            us0, ut0 = claudius.sc_field(sol0, R, T, P), claudius.tt_field(
                sol0, R, T, P
            )
            us1, ut1 = claudius.sc_field(sol1, R, T, P), claudius.tt_field(
                sol0, R, T, P
            )
            ui = claudius.Helmholtz_3d.incident_field(k, R, T, P, "spherical")

            assert assert_allclose(ui + us0, ut0, rtol=1e-6) is None
            assert assert_allclose(ui + us1, ut1, rtol=1e-6) is None

            assert assert_allclose(us0, us1, atol=1e-6) is None
            assert assert_allclose(ut0, ut1, atol=1e-6) is None
