from matplotlib.pyplot import Circle, Rectangle, show, subplots
from numpy import amax, amin, linspace, where


def plot_geometry(prob, T=None):
    penetrable = prob.inn_bdy.startswith("P")
    ρ = prob.radii
    εμ = prob.εμ

    if T is None:
        T = 1.5 * ρ[-1]

    if penetrable:
        shift = 1
    else:
        shift = 0

    vr, vε, vμ = [], [], []

    if penetrable:
        r = linspace(ρ[0], 0, num=int(128 * ρ[0] / T), endpoint=False)
        vr.append(r)
        vε.append(εμ[0][0](r))
        vμ.append(εμ[0][1](r))

    for n, (ρ1, ρ2) in enumerate(zip(ρ[:-1], ρ[1:]), start=shift):
        r = linspace(ρ1, ρ2, num=int(128 * (ρ2 - ρ1) / T))
        vr.append(r)
        vε.append(εμ[n][0](r))
        vμ.append(εμ[n][1](r))

    ymin = min(*[amin(ε) for ε in vε], *[amin(μ) for μ in vμ], 1)
    ymin = 0.95 * ymin if ymin > 0 else 1.05 * ymin
    ymax = 1.05 * max(*[amax(ε) for ε in vε], *[amax(μ) for μ in vμ], 1)

    fig, ax = subplots(1, 2)

    ax[0].set_xlim(-T, T)
    ax[0].set_ylim(-T, T)
    ax[0].set_aspect("equal")

    ax[1].set_xlim(0, T)
    ax[1].set_ylim(ymin, ymax)
    ax[1].grid(True)

    if penetrable:
        ax[0].add_artist(Circle((0, 0), ρ[0], fill=False, ec="k", lw=2, ls="--"))

        ax[1].plot([ρ[0], ρ[0]], [ymin, ymax], "k--")
        ax[1].plot(vr[0], vε[0], "C0", lw=2)
        ax[1].plot(vr[0], vμ[0], "C1--", lw=2)
    else:
        ax[0].add_artist(Circle((0, 0), ρ[0], fc=(0.75, 0.75, 0.75), ec="k", lw=2))

        ax[1].add_artist(Rectangle((0, ymin), ρ[0], ymax - ymin, fc=(0.75, 0.75, 0.75)))
        ax[1].plot([ρ[0], ρ[0]], [ymin, ymax], "k")

    for r_, ε_, μ_ in zip(vr[shift:], vε[shift:], vμ[shift:]):
        ax[0].add_artist(Circle((0, 0), r_[-1], fill=False, ec="k", lw=2, ls="--"))

        ax[1].plot([r_[-1], r_[-1]], [ymin, ymax], "k--")
        ax[1].plot(r_, ε_, "C0", lw=2)
        ax[1].plot(r_, μ_, "C1--", lw=2)

    ax[1].plot([ρ[-1], T], [1, 1], "C0", lw=2, label="ε")
    ax[1].plot([ρ[-1], T], [1, 1], "C1--", lw=2, label="μ")

    ax[1].legend()

    show()


def plot_potential(prob, λ, ylim=None, T=None):
    penetrable = prob.inn_bdy.startswith("P")
    ρ = prob.radii
    εμ = prob.εμ

    if T is None:
        T = 1.5 * ρ[-1]

    if penetrable:
        shift = 1
    else:
        shift = 0

    vr, vV, vλ = [], [], []

    if penetrable:
        r = linspace(ρ[0], 0, num=int(128 * ρ[0] / T), endpoint=False)
        V = 1 / (εμ[0][0](r) * εμ[0][1](r) * r * r)

        vr.append(r)
        if V[0] > 0:
            vV.append(V)
            vλ.append(λ)
        else:
            vV.append(-V)
            vλ.append(-λ)

    for n, (ρ1, ρ2) in enumerate(zip(ρ[:-1], ρ[1:]), start=shift):
        r = linspace(ρ1, ρ2, num=int(128 * (ρ2 - ρ1) / T))
        V = 1 / (εμ[n][0](r) * εμ[n][1](r) * r * r)

        vr.append(r)
        if V[0] > 0:
            vV.append(V)
            vλ.append(λ)
        else:
            vV.append(-V)
            vλ.append(-λ)

    if ylim is None:
        ymin = min(*[amin(V) for V in vV], *[σ for σ in vλ], 1 / (T * T))
        ymin = 0.95 * ymin if ymin > 0 else 1.05 * ymin
        ymax = 1.05 * max(*[amax(V) for V in vV], λ, 1 / (ρ[-1] * ρ[-1]))
    else:
        ymin, ymax = ylim

    fig, ax = subplots()

    ax.set_xlim(0, T)
    ax.set_ylim(ymin, ymax)
    ax.grid(True)

    if penetrable:
        ax.plot([ρ[0], ρ[0]], [ymin, ymax], "k--")
        ax.plot(vr[0], vV[0], "C0", lw=2)
        ax.plot([0, ρ[0]], [vλ[0], vλ[0]], "C1", lw=2)
    else:
        ax.add_artist(Rectangle((0, ymin), ρ[0], ymax - ymin, fc=(0.75, 0.75, 0.75)))
        ax.plot([ρ[0], ρ[0]], [ymin, ymax], "k")

    for r_, V_, λ_ in zip(vr[shift:], vV[shift:], vλ[shift:]):
        ax.plot([r_[-1], r_[-1]], [ymin, ymax], "k--")
        ax.plot(r_, V_, "C0", lw=2)
        ax.plot([r_[0], r_[-1]], [λ_, λ_], "C1", lw=2)

    r = linspace(ρ[-1], T, num=int(128 * (T - ρ[-1]) / T))
    ax.plot(r, 1 / (r * r), "C0", lw=2, label="V")
    ax.plot([ρ[-1], T], [λ, λ], "C1", lw=2, label="λ")

    ax.legend()

    show()
