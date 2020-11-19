from context import claudius as acs
from numpy import array


def _create_Helmholtz_2d(dim, pde, inn_bdy):
    radii = (1,)
    k = 1
    if inn_bdy.startswith("P"):
        εμc = ((1, 1),)
        func = ((),)
        func_der = ((),)
    else:
        εμc = ()
        func = ()
        func_der = ()

    prob = acs.create_probem(dim, pde, inn_bdy, radii, εμc, k, func, func_der)
    return acs.Solution(*prob, array([]))


class TestHelmholtz_2d:
    def test_dirichlet(self):
        assert _create_Helmholtz_2d(2, "Helmholtz", "Dirichlet")

    def test_neuman(self):
        assert _create_Helmholtz_2d(2, "Helmholtz", "Neumann")

    def test_penetrable(self):
        assert _create_Helmholtz_2d(2, "Helmholtz", "Penetrable")


class TestHelmholtz_3d:
    def test_dirichlet(self):
        assert _create_Helmholtz_2d(3, "Helmholtz", "Dirichlet")

    def test_neuman(self):
        assert _create_Helmholtz_2d(3, "Helmholtz", "Neumann")

    def test_penetrable(self):
        assert _create_Helmholtz_2d(3, "Helmholtz", "Penetrable")


"""
class TestMaxwell_3d:
    def test_dirichlet(self):
        assert _create_Helmholtz_2d(3, "Maxwell", "Dirichlet")

    def test_neuman(self):
        assert _create_Helmholtz_2d(3, "Maxwell", "Neumann")

    def test_penetrable(self):
        assert _create_Helmholtz_2d(3, "Maxwell", "Penetrable")
"""
