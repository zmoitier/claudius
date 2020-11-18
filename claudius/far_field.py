from numpy import asarray

from claudius.Helmholtz_2d import f_field as ff_H2d


def far_field(sol, coo_t, coo_p=None):
    Theta = asarray(coo_t)

    if sol.dim == 2:
        if coo_p is not None:
            exit("For dimension 2 the variable coo_p should be None.")

        return ff_H2d(sol, Theta)

    if sol.dim == 3:
        if coo_p is None:
            exit("For dimension 3 the variable φ should not be None.")
        Phi = asarray(coo_p)

        if sol.pde.startswith("H"):
            return None

        if sol.pde.startswith("M"):
            return None
