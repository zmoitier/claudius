from numpy import arange, linspace, meshgrid, pi, size, sqrt, zeros_like
from numpy.random import uniform
from numpy.testing import assert_allclose

import claudius
from claudius.Helmholtz_3d import (
    create_problem_cst,
    incident_field,
    scattered_field,
    total_field,
)


def coeff_serie(l):
    return 1j ** l * sqrt(4 * pi * (2 * l + 1))


def _sol(inn_bdy, radii_list, εμ_list, k):
    pb_list = [
        create_problem_cst(inn_bdy, radii, εμ, k)
        for radii, εμ in zip(radii_list, εμ_list)
    ]

    return [claudius.solve_prob(pb, radius_max=3) for pb in pb_list]


class TestSolve:
    def test_solve_D(self):
        pb = create_problem_cst("Dirichlet", (0.5, 1, 1.5), ((-2, 1), (1, 2)), 1)
        assert claudius.solve_prob(pb, trunc_series=1)

    def test_solve_N(self):
        pb = create_problem_cst("Neumann", (0.5, 1, 1.5), ((-2, 1), (1, 2)), 1)
        assert claudius.solve_prob(pb, trunc_series=1)

    def test_solve_P(self):
        pb = create_problem_cst("Penetrable", (0.5, 1), ((-2, 1), (1, 2)), 1)
        assert claudius.solve_prob(pb, trunc_series=1)


class TestImpenetrable:
    def test_dirichlet(self):
        for k in uniform(0.5, 2, 8):
            sol0, sol1 = _sol("Dirichlet", [(1,), (1, 2)], [(), ((1, 1),)], k)
            c0, c1 = sol0.coeff, sol1.coeff

            cl = coeff_serie(arange(size(c0, 0)))

            assert assert_allclose(c1[:, 2], c0[:, 0], rtol=5e-6) is None
            assert assert_allclose(c1[:, 0], cl + c0[:, 0], rtol=5e-6) is None
            assert assert_allclose(c1[:, 1], 1j * c0[:, 0], rtol=5e-6) is None

            r, t, p = (
                linspace(1, 3, num=8),
                linspace(0, 2 * pi, num=8, endpoint=False),
                linspace(0, pi, num=6)[1:-1],
            )
            R, T, P = meshgrid(r, t, p)
            us0, ut0 = scattered_field(sol0, R, T, P), total_field(sol0, R, T, P)
            us1, ut1 = scattered_field(sol1, R, T, P), total_field(sol0, R, T, P)

            assert assert_allclose(us0, us1, atol=5e-6) is None
            assert assert_allclose(ut0, ut1, atol=5e-6) is None

    def test_neumann(self):
        for k in uniform(0.5, 2, 8):
            sol0, sol1 = _sol("Neumann", [(1,), (1, 2)], [(), ((1, 1),)], k)
            c0, c1 = sol0.coeff, sol1.coeff

            cl = coeff_serie(arange(size(c0, 0)))

            assert assert_allclose(c1[:, 2], c0[:, 0], rtol=5e-6) is None
            assert assert_allclose(c1[:, 0], cl + c0[:, 0], rtol=5e-6) is None
            assert assert_allclose(c1[:, 1], 1j * c0[:, 0], rtol=5e-6) is None

            r, t, p = (
                linspace(1, 3, num=8),
                linspace(0, 2 * pi, num=8, endpoint=False),
                linspace(0, pi, num=6)[1:-1],
            )
            R, T, P = meshgrid(r, t, p)
            us0, ut0 = scattered_field(sol0, R, T, P), total_field(sol0, R, T, P)
            us1, ut1 = scattered_field(sol1, R, T, P), total_field(sol0, R, T, P)

            assert assert_allclose(us0, us1, rtol=5e-6) is None
            assert assert_allclose(ut0, ut1, rtol=5e-6) is None


class TestPenetrable:
    def test_penetrable_coeff(self):
        for k in uniform(0.5, 2, 8):
            sol0, sol1 = _sol(
                "Penetrable", [(1,), (1, 2)], [((1, 1),), ((1, 1), (1, 1))], k
            )
            c0, c1 = sol0.coeff, sol1.coeff

            cl = coeff_serie(arange(size(c0, 0)))

            assert assert_allclose(c0[:, 0], cl, rtol=5e-6) is None
            assert assert_allclose(c0[:, 1], zeros_like(c0[:, 1]), atol=2e-15) is None

            assert assert_allclose(c1[:, 0], cl, rtol=5e-6) is None
            assert assert_allclose(c1[:, 1], cl, rtol=5e-6) is None
            assert assert_allclose(c1[:, 2], zeros_like(c1[:, 2]), atol=2e-15) is None
            assert assert_allclose(c1[:, 3], zeros_like(c1[:, 3]), atol=2e-15) is None

    def test_penetrable_field(self):
        for k in uniform(0.5, 2, 8):
            sol0, sol1 = _sol(
                "Penetrable", [(1,), (1, 2)], [((1, 1),), ((1, 1), (1, 1))], k
            )

            r, t, p = (
                linspace(1, 3, num=8),
                linspace(0, 2 * pi, num=8, endpoint=False),
                linspace(0, pi, num=6)[1:-1],
            )
            R, T, P = meshgrid(r, t, p)
            us0, ut0 = scattered_field(sol0, R, T, P), total_field(sol0, R, T, P)
            us1, ut1 = scattered_field(sol1, R, T, P), total_field(sol0, R, T, P)
            ui = incident_field(k, R, T, P)

            assert assert_allclose(ui + us0, ut0, rtol=5e-6) is None
            assert assert_allclose(ui + us1, ut1, rtol=5e-6) is None

            assert assert_allclose(us0, us1, atol=5e-6) is None
            assert assert_allclose(ut0, ut1, atol=5e-6) is None
