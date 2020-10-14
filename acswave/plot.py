from matplotlib.pyplot import plot, subplots, show, Circle


def plot_geometry(prob):
    fig, ax = subplots(1, 2)

    δ = prob.radii

    T = 1.5 * δ[-1]
    ax[0].set_xlim(-T, T)
    ax[0].set_ylim(-T, T)
    ax[0].set_aspect("equal")

    if prob.inn_bdy.startswith("P"):
        ax[0].add_artist(Circle((0, 0), δ[0], fill=False, ec="k", lw=2, ls="--"))
    else:
        ax[0].add_artist(Circle((0, 0), δ[0], fc=(0.75, 0.75, 0.75), ec="k", lw=2))

    for r in δ[1:]:
        ax[0].add_artist(Circle((0, 0), r, fill=False, ec="k", lw=2, ls="--"))

    show()
