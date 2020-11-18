from matplotlib.pyplot import Circle, Rectangle, show, subplots
from numpy import amax, amin, linspace


def _calc_εμ(prob, penetrable, shift):
    vr, vε, vμ = [], [], []

    ρ = prob.radii
    εμ = prob.eps_mu

    if penetrable:
        r = linspace(ρ[0], 0, num=int(128 * ρ[0] / ρ[-1]), endpoint=False)
        vr.append(r)
        vε.append(εμ[0][0](r))
        vμ.append(εμ[0][1](r))

    for n, (ρ1, ρ2) in enumerate(zip(ρ[:-1], ρ[1:]), start=shift):
        r = linspace(ρ1, ρ2, num=int(128 * (ρ2 - ρ1) / ρ[-1]))
        vr.append(r)
        vε.append(εμ[n][0](r))
        vμ.append(εμ[n][1](r))

    ymin = min(*[amin(ε) for ε in vε], *[amin(μ) for μ in vμ], 1)
    ymin = 0.95 * ymin if ymin > 0 else 1.05 * ymin
    ymax = 1.05 * max(*[amax(ε) for ε in vε], *[amax(μ) for μ in vμ], 1)

    return (vr, vε, vμ, (ymin, ymax))


def plot_geometry(prob, T=None):
    penetrable = prob.inn_bdy.startswith("P")

    if T is None:
        T = 1.5 * prob.radii[-1]

    shift = 1 if penetrable else 0

    vr, vε, vμ, (εμmin, εμmax) = _calc_εμ(prob, penetrable, shift)

    fig, ax = subplots(1, 2)

    ax[0].set_xlim(-T, T)
    ax[0].set_ylim(-T, T)
    ax[0].set_aspect("equal")

    ax[1].set_xlim(0, T)
    ax[1].set_ylim(εμmin, εμmax)
    ax[1].grid(True)

    if penetrable:
        ax[0].add_artist(Circle((0, 0), vr[0][0], fill=False, ec="k", lw=2, ls="--"))

        ax[1].plot([vr[0][0], vr[0][0]], [εμmin, εμmax], "k--")
        ax[1].plot(vr[0], vε[0], "C0", lw=2)
        ax[1].plot(vr[0], vμ[0], "C1--", lw=2)
    else:
        ax[0].add_artist(Circle((0, 0), vr[0][0], fc=(0.75, 0.75, 0.75), ec="k", lw=2))

        ax[1].add_artist(
            Rectangle((0, εμmin), vr[0][0], εμmax - εμmin, fc=(0.75, 0.75, 0.75))
        )
        ax[1].plot([vr[0][0], vr[0][0]], [εμmin, εμmax], "k")

    for r_, ε_, μ_ in zip(vr[shift:], vε[shift:], vμ[shift:]):
        ax[0].add_artist(Circle((0, 0), r_[-1], fill=False, ec="k", lw=2, ls="--"))

        ax[1].plot([r_[-1], r_[-1]], [εμmin, εμmax], "k--")
        ax[1].plot(r_, ε_, "C0", lw=2)
        ax[1].plot(r_, μ_, "C1--", lw=2)

    ax[1].plot([vr[-1][-1], T], [1, 1], "C0", lw=2, label="ε")
    ax[1].plot([vr[-1][-1], T], [1, 1], "C1--", lw=2, label="μ")

    ax[1].legend()

    show()


def plot_potential(prob, λ, Vlim=None, T=None):
    penetrable = prob.inn_bdy.startswith("P")

    if T is None:
        T = 1.5 * prob.radii[-1]

    shift = 1 if penetrable else 0

    vr, vε, vμ, (εμmin, εμmax) = _calc_εμ(prob, penetrable, shift)
    vV, vλ = [], []

    for r_, ε_, μ_ in zip(vr, vε, vμ):
        V = 1 / (ε_ * μ_ * r_ * r_)
        if V[0] > 0:
            vV.append(V)
            vλ.append(λ)
        else:
            vV.append(-V)
            vλ.append(-λ)

    if Vlim is None:
        Vmin = min(*[amin(V) for V in vV], *[σ for σ in vλ], 1 / (T * T))
        Vmin = 0.95 * Vmin if Vmin > 0 else 1.05 * Vmin
        Vmax = 1.05 * max(*[amax(V) for V in vV], λ, 1 / (vr[-1][-1] * vr[-1][-1]))
    else:
        Vmin, Vmax = Vlim

    fig, ax = subplots(1, 2)

    ax[0].set_xlim(0, T)
    ax[0].set_ylim(εμmin, εμmax)
    ax[0].grid(True)

    ax[1].set_xlim(0, T)
    ax[1].set_ylim(Vmin, Vmax)
    ax[1].grid(True)

    if penetrable:
        ax[0].plot([vr[0][0], vr[0][0]], [εμmin, εμmax], "k--")
        ax[0].plot(vr[0], vε[0], "C0", lw=2)
        ax[0].plot(vr[0], vμ[0], "C1--", lw=2)

        ax[1].plot([vr[0][0], vr[0][0]], [Vmin, Vmax], "k--")
        ax[1].plot(vr[0], vV[0], "C3", lw=2)
        ax[1].plot([0, vr[0][0]], [vλ[0], vλ[0]], "C2", lw=2)
    else:
        ax[0].add_artist(
            Rectangle((0, εμmin), vr[0][0], εμmax - εμmin, fc=(0.75, 0.75, 0.75))
        )
        ax[0].plot([vr[0][0], vr[0][0]], [εμmin, εμmax], "k")

        ax[1].add_artist(
            Rectangle((0, Vmin), vr[0][0], Vmax - Vmin, fc=(0.75, 0.75, 0.75))
        )
        ax[1].plot([vr[0][0], vr[0][0]], [Vmin, Vmax], "k")

    for r_, ε_, μ_, V_, λ_ in zip(
        vr[shift:], vε[shift:], vμ[shift:], vV[shift:], vλ[shift:]
    ):
        ax[0].plot([r_[-1], r_[-1]], [εμmin, εμmax], "k--")
        ax[0].plot(r_, ε_, "C0", lw=2)
        ax[0].plot(r_, μ_, "C1--", lw=2)

        ax[1].plot([r_[-1], r_[-1]], [Vmin, Vmax], "k--")
        ax[1].plot(r_, V_, "C3", lw=2)
        ax[1].plot([r_[0], r_[-1]], [λ_, λ_], "C2", lw=2)

    ax[0].plot([vr[-1][-1], T], [1, 1], "C0", lw=2, label="ε")
    ax[0].plot([vr[-1][-1], T], [1, 1], "C1--", lw=2, label="μ")

    ax[0].legend()

    r = linspace(vr[-1][-1], T, num=int(128 * (T - vr[-1][-1]) / T))
    ax[1].plot(r, 1 / (r * r), "C3", lw=2, label="V")
    ax[1].plot([vr[-1][-1], T], [λ, λ], "C2", lw=2, label="λ")

    ax[1].legend()

    show()
