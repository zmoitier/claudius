from claudius import solve_prob, to_spheric, trunc_H3d, trunc_None

from .calc_field import f_field, sc_field, tt_field


def scattered_field(prob, coo_xr, coo_yt, coo_zp, type_coord, T=None, M=None):
    r, θ, φ = to_spheric(coo_xr, coo_yt, coo_zp, type_coord)

    M = trunc_None(trunc_H3d, prob.wavenum, T, r, M)
    sol = solve_prob(prob, M)
    return sc_field(sol, r, θ, φ)


def total_field(prob, coo_xr, coo_yt, coo_zp, type_coord, T=None, M=None):
    r, θ, φ = to_spheric(coo_xr, coo_yt, coo_zp, type_coord)

    M = trunc_None(trunc_H3d, prob.wavenum, T, r, M)
    sol = solve_prob(prob, M)
    return tt_field(sol, r, θ, φ)


def far_field(prob, coo_t, coo_p, T=None, M=None):
    M = trunc_None(trunc_H3d, prob.wavenum, T, 2 * prob.radii[-1], M)
    sol = solve_prob(prob, M)
    return f_field(sol, coo_t, coo_p)
