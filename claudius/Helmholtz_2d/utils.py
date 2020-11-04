from claudius import solve_prob, to_polar, trunc_H2d, trunc_None

from .calc_field import f_field, sc_field, tt_field


def scattered_field(prob, c1, c2, coord, T=None, M=None):
    r, θ = to_polar(c1, c2, coord)

    M = trunc_None(trunc_H2d, prob.k, T, r, M)
    sol = solve_prob(prob, M)
    return sc_field(sol, r, θ)


def total_field(prob, c1, c2, coord, T=None, M=None):
    r, θ = to_polar(c1, c2, coord)

    M = trunc_None(trunc_H2d, prob.k, T, r, M)
    sol = solve_prob(prob, M)
    return tt_field(sol, r, θ)


def far_field(prob, θ, T=None, M=None):
    M = trunc_None(trunc_H2d, prob.k, T, 2 * prob.radii[-1], M)
    sol = solve_prob(prob, M)
    return f_field(sol, θ)
