from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import Circle, show, subplots
from numpy import absolute, amax, amin, exp, pi, real


def list_disk(ax, radii, inn_bdy):
    disks = []
    if inn_bdy.startswith("P"):
        disks.append(
            Circle((0, 0), radii[0], fill=False, ec="k", lw=2, ls="--", animated=True)
        )
    else:
        disks.append(
            Circle((0, 0), radii[0], fc=(0.75, 0.75, 0.75), ec="k", lw=2, animated=True)
        )

    for ρ in radii[1:]:
        disks.append(
            Circle((0, 0), ρ, fill=False, ec="k", lw=2, ls="--", animated=True)
        )

    return disks


def anim_field(prob, X, Y, U, type_field):
    fig, ax = subplots(subplot_kw={"aspect": "equal"})

    ax.set_title(fr"{type_field} with $k = {prob.wavenum}$")

    disks = list_disk(ax, prob.radii, prob.inn_bdy)

    U_max = amax(absolute(U))
    im = ax.imshow(
        real(U),
        interpolation="none",
        cmap="RdBu_r",
        origin="lower",
        aspect="equal",
        extent=(amin(X), amax(X), amin(Y), amax(Y)),
        animated=True,
        vmin=-U_max,
        vmax=U_max,
    )
    fig.colorbar(im, ax=ax)

    nbt = 110
    dt = 2 * pi / (prob.wavenum * nbt)
    expi = exp(-1j * prob.wavenum * dt)

    def init():
        return (im, *(ax.add_artist(disk) for disk in disks))

    def animate(i, U, expi, disks):
        U *= expi
        im.set_array(real(U))
        return (im, *(ax.add_artist(disk) for disk in disks))

    anim = FuncAnimation(
        fig,
        animate,
        fargs=(U, expi, disks),
        init_func=init,
        frames=nbt,
        blit=True,
        interval=20,
        repeat=True,
    )

    show()
