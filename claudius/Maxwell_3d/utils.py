from claudius import solve_prob, to_spheric, trunc_H3d, trunc_None

from .calc_field import f_field, sc_field, tt_field


def scattered_field(prob, c1, c2, c3, coord, T=None, M=None):
    r, θ, φ = to_spheric(c1, c2, c3, coord)

    M = trunc_None(trunc_H3d, prob.k, T, r, M)
    sol = solve_prob(prob, M)
    return sc_field(sol, r, θ, φ)


def total_field(prob, c1, c2, c3, coord, T=None, M=None):
    r, θ, φ = to_spheric(c1, c2, coord)

    M = trunc_None(trunc_H3d, prob.k, T, r, M)
    sol = solve_prob(prob, M)
    return tt_field(sol, r, θ, φ)


def far_field(prob, θ, φ, T=None, M=None):
    M = trunc_None(trunc_H3d, prob.k, T, 2 * prob.radii[-1], M)
    sol = solve_prob(prob, M)
    return f_field(sol, θ, φ)
