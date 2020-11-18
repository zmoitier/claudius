from sys import exit

from numpy import asarray

from claudius.Helmholtz_2d import sc_field as scf_H2d
from claudius.Helmholtz_2d import tt_field as ttf_H2d
from claudius.Helmholtz_3d import sc_field as scf_H3d
from claudius.Helmholtz_3d import tt_field as ttf_H3d
from claudius.Maxwell_3d import sc_field as scf_M3d
from claudius.Maxwell_3d import tt_field as ttf_M3d


def sc_field(sol, coo_r, coo_t, coo_p=None):
    R, Theta = asarray(coo_r), asarray(coo_t)

    if sol.dim == 2:
        if coo_p is not None:
            exit("For dimension 2 the variable coo_p should be None.")

        return scf_H2d(sol, R, Theta)

    if sol.dim == 3:
        if coo_p is None:
            exit("For dimension 3 the variable φ should not be None.")
        Phi = asarray(coo_p)

        if sol.pde.startswith("H"):
            return scf_H3d(sol, R, Theta, Phi)

        if sol.pde.startswith("M"):
            return scf_M3d(sol, R, Theta, Phi)


def tt_field(sol, coo_r, coo_t, coo_p=None):
    R, Theta = asarray(coo_r), asarray(coo_t)

    if sol.dim == 2:
        if coo_p is not None:
            exit("For dimension 2 the variable coo_p should be None.")

        return ttf_H2d(sol, R, Theta)

    if sol.dim == 3:
        if coo_p is None:
            exit("For dimension 3 the variable φ should not be None.")
        Phi = asarray(coo_p)

        if sol.pde.startswith("H"):
            return ttf_H3d(sol, R, Theta, Phi)

        if sol.pde.startswith("M"):
            return ttf_M3d(sol, R, Theta, Phi)
