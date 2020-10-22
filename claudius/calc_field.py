from sys import exit

from numpy import array, isscalar

from claudius import Helmholtz_2d, Helmholtz_3d, Maxwell_3d


def sc_field(sol, r, θ, φ=None):
    R = r if not isscalar(r) else array([r])
    Θ = θ if not isscalar(θ) else array([θ])
    Φ = φ if not isscalar(φ) else array([φ])

    if (sol.dim == 2) and (sol.pde.startswith("H")):
        if φ is not None:
            exit("For dimension 2 the variable φ should be None.")

        return Helmholtz_2d.sc_field(sol, R, Θ)

    if (sol.dim == 3) and (sol.pde.startswith("H")):
        if φ is None:
            exit("For dimension 3 the variable φ should not be None.")

        return Helmholtz_3d.sc_field(sol, R, Θ, Φ)

    if (sol.dim == 3) and (sol.pde.startswith("M")):
        if φ is None:
            exit("For dimension 3 the variable φ should not be None.")

        return Maxwell_3d.sc_field(sol, R, Θ, Φ)


def tt_field(sol, r, θ, φ=None):
    R = r if not isscalar(r) else array([r])
    Θ = θ if not isscalar(θ) else array([θ])
    Φ = φ if not isscalar(φ) else array([φ])

    if (sol.dim == 2) and (sol.pde.startswith("H")):
        if φ is not None:
            exit("For dimension 2 the variable φ should be None.")

        return Helmholtz_2d.tt_field(sol, R, Θ)

    if (sol.dim == 3) and (sol.pde.startswith("H")):
        if φ is None:
            exit("For dimension 3 the variable φ should not be None.")

        return Helmholtz_3d.tt_field(sol, R, Θ, Φ)

    if (sol.dim == 3) and (sol.pde.startswith("M")):
        if φ is None:
            exit("For dimension 3 the variable φ should not be None.")

        return Maxwell_3d.tt_field(sol, R, Θ, Φ)
