from context import claudius as acs
from numpy import linspace, ones_like, zeros_like
from numpy.random import uniform
from numpy.testing import assert_allclose

from claudius.Helmholtz_2d import fun_cst, fun_cst_der


def _define_prob(inn_bdy, k, N):
    dim = 2
    pde = "Helmholtz"

    if N == 0:
        radii = (1,)
    else:
        radii = tuple(linspace(1, 2, num=N + 1))

    if inn_bdy.startswith("P"):
        εμc = tuple((ones_like, ones_like) for n in range(N + 1))
        fun = (fun_cst(1, 1, k)[0], *(fun_cst(1, 1, k) for n in range(N)))
        fun_der = (fun_cst_der(1, 1, k)[0], *(fun_cst_der(1, 1, k) for n in range(N)))
    else:
        εμc = tuple((ones_like, ones_like) for n in range(N))
        fun = tuple(fun_cst(1, 1, k) for n in range(N))
        fun_der = tuple(fun_cst_der(1, 1, k) for n in range(N))

    return acs.create_probem(dim, pde, inn_bdy, radii, εμc, k, fun, fun_der)


def _coeff_sol(inn_bdy, k, N_list):
    pb_list = [_define_prob(inn_bdy, k, N) for N in N_list]

    M = acs.trunc_H2d(k, 3)

    sol_list = [acs.solve_prob(pb, M) for pb in pb_list]

    return tuple(sol.coeff for sol in sol_list)


class TestImpenetrable:
    def test_dirichlet(self):
        vk = uniform(0.5, 2, 8)
        for k in vk:
            c0, c1 = _coeff_sol("Dirichlet", k, [0, 1])

            assert assert_allclose(c1[:, 2], c0[:, 0]) is None
            assert assert_allclose(c1[:, 0], 1 + c0[:, 0]) is None
            assert assert_allclose(c1[:, 1], 1j * c0[:, 0]) is None

    def test_neumann(self):
        vk = uniform(0.5, 2, 8)
        for k in vk:
            c0, c1 = _coeff_sol("Neumann", k, [0, 1])

            assert assert_allclose(c1[:, 2], c0[:, 0]) is None
            assert assert_allclose(c1[:, 0], 1 + c0[:, 0]) is None
            assert assert_allclose(c1[:, 1], 1j * c0[:, 0]) is None


class TestPenetrable:
    def test_penetrable_0(self):
        vk = uniform(0.5, 2, 8)
        for k in vk:
            (c,) = _coeff_sol("Penetrable", k, [0])

            assert assert_allclose(c[:, 0], ones_like(c[:, 0])) is None
            assert assert_allclose(c[:, 1], zeros_like(c[:, 1]), atol=1e-15) is None

    def test_penetrable_1(self):
        vk = uniform(0.5, 2, 8)
        for k in vk:
            (c,) = _coeff_sol("Penetrable", k, [1])

            assert assert_allclose(c[:, 0], ones_like(c[:, 0])) is None
            assert assert_allclose(c[:, 1], ones_like(c[:, 1])) is None
            assert assert_allclose(c[:, 2], zeros_like(c[:, 2]), atol=1e-15) is None
            assert assert_allclose(c[:, 3], zeros_like(c[:, 3]), atol=1e-15) is None
