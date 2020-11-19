from matplotlib.pyplot import Circle, show, subplots, suptitle, tight_layout
from numpy import absolute, amax, angle, imag, pi, real


def plot_real(fig, ax, X, Y, U_real):
    U_max = amax(absolute(U_real))
    p = ax.pcolormesh(
        X, Y, U_real, shading="gouraud", cmap="RdBu_r", vmin=-U_max, vmax=U_max
    )
    fig.colorbar(p, ax=ax)
    ax.set_title("Real part")


def plot_imag(fig, ax, X, Y, U_imag):
    U_max = amax(absolute(U_imag))
    p = ax.pcolormesh(
        X, Y, U_imag, shading="gouraud", cmap="RdBu_r", vmin=-U_max, vmax=U_max
    )
    fig.colorbar(p, ax=ax)
    ax.set_title("Imaginary part")


def plot_abs(fig, ax, X, Y, U_abs):
    U_max = amax(U_abs)
    p = ax.pcolormesh(X, Y, U_abs, shading="gouraud", vmin=0, vmax=U_max)
    fig.colorbar(p, ax=ax)
    ax.set_title("Modulus")


def plot_arg(fig, ax, X, Y, U_arg):
    p = ax.pcolormesh(
        X, Y, U_arg, shading="gouraud", cmap="twilight_shifted_r", vmin=-pi, vmax=pi
    )
    fig.colorbar(p, ax=ax)
    ax.set_title("Argument")


def add_disk(ax, radii, inn_bdy):
    if inn_bdy.startswith("P"):
        ax.add_artist(Circle((0, 0), radii[0], fill=False, ec="k", lw=2, ls="--"))
    else:
        ax.add_artist(Circle((0, 0), radii[0], fc=(0.75, 0.75, 0.75), ec="k", lw=2))

    for ρ in radii[1:]:
        ax.add_artist(Circle((0, 0), ρ, fill=False, ec="k", lw=2, ls="--"))


def plot_field(prob, X, Y, U, type_field):
    fig, ax = subplots(nrows=2, ncols=2, subplot_kw={"aspect": "equal"})

    plot_real(fig, ax[0, 0], X, Y, real(U))
    plot_imag(fig, ax[0, 1], X, Y, imag(U))
    plot_abs(fig, ax[1, 0], X, Y, absolute(U))
    plot_arg(fig, ax[1, 1], X, Y, angle(U))

    for a in (ax[i, j] for i in range(2) for j in range(2)):
        add_disk(a, prob.radii, prob.inn_bdy)

    suptitle(fr"{type_field} with $k = {prob.wavenum}$")

    tight_layout()

    show()
