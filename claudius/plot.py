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

    for n, ρn in enumerate(ρ[1:], start=shift):
        ax[0].add_artist(Circle((0, 0), ρn, fill=False, ec="k", lw=2, ls="--"))

        ax[1].plot([ρn, ρn], [ymin, ymax], "k--")
        ax[1].plot(vr[n], vε[n], "C0", lw=2)
        ax[1].plot(vr[n], vμ[n], "C1--", lw=2)

    ax[1].plot([ρ[-1], T], [1, 1], "C0", lw=2, label="ε")
    ax[1].plot([ρ[-1], T], [1, 1], "C1--", lw=2, label="μ")

    ax[1].legend()

    show()
