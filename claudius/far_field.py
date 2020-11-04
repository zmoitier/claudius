from claudius.Helmholtz_2d import f_field as ff_H2d


def far_field(sol, θ, φ=None):
    if (sol.dim == 2) and (sol.pde.startswith("H")):
        if φ is not None:
            exit("For dimension 2 the variable φ should be None.")

        return ff_H2d(sol, θ)

    if (sol.dim == 3) and (sol.pde.startswith("H")):
        if φ is None:
            exit("For dimension 3 the variable φ should not be None.")

        return None

    if (sol.dim == 3) and (sol.pde.startswith("M")):
        if φ is None:
            exit("For dimension 3 the variable φ should not be None.")

        return None
