from sys import exit

from numpy import array, isscalar

from claudius.Helmholtz_2d import sc_field as scf_H2d
from claudius.Helmholtz_2d import tt_field as ttf_H2d
from claudius.Helmholtz_3d import sc_field as scf_H3d
from claudius.Helmholtz_3d import tt_field as ttf_H3d
from claudius.Maxwell_3d import sc_field as scf_M3d
from claudius.Maxwell_3d import tt_field as ttf_M3d


def sc_field(sol, r, θ, φ=None):
    R = r if not isscalar(r) else array([r])
    Θ = θ if not isscalar(θ) else array([θ])
    Φ = φ if not isscalar(φ) else array([φ])

    if (sol.dim == 2) and (sol.pde.startswith("H")):
        if φ is not None:
            exit("For dimension 2 the variable φ should be None.")

        return scf_H2d(sol, R, Θ)

    if (sol.dim == 3) and (sol.pde.startswith("H")):
        if φ is None:
            exit("For dimension 3 the variable φ should not be None.")

        return scf_H3d(sol, R, Θ, Φ)

    if (sol.dim == 3) and (sol.pde.startswith("M")):
        if φ is None:
            exit("For dimension 3 the variable φ should not be None.")

        return scf_M3d(sol, R, Θ, Φ)


def tt_field(sol, r, θ, φ=None):
    R = r if not isscalar(r) else array([r])
    Θ = θ if not isscalar(θ) else array([θ])
    Φ = φ if not isscalar(φ) else array([φ])

    if (sol.dim == 2) and (sol.pde.startswith("H")):
        if φ is not None:
            exit("For dimension 2 the variable φ should be None.")

        return ttf_H2d(sol, R, Θ)

    if (sol.dim == 3) and (sol.pde.startswith("H")):
        if φ is None:
            exit("For dimension 3 the variable φ should not be None.")

        return ttf_H3d(sol, R, Θ, Φ)

    if (sol.dim == 3) and (sol.pde.startswith("M")):
        if φ is None:
            exit("For dimension 3 the variable φ should not be None.")

        return ttf_M3d(sol, R, Θ, Φ)
