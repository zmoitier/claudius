from matplotlib.collections import LineCollection
from matplotlib.pyplot import show, subplots
from numpy import absolute, amax, angle, array, concatenate, pi


def plot_far_field(Theta, ff):
    am = absolute(ff)
    ph = angle((ff[:-1] + ff[1:]) / 2)

    points = array([Theta, am]).T.reshape(-1, 1, 2)
    segments = concatenate([points[:-2], points[1:-1], points[2:]], axis=1)

    fig, ax = subplots(subplot_kw={"polar": True})

    lc = LineCollection(segments, cmap="twilight_shifted_r", linewidth=2)
    lc.set_array(ph)
    im = ax.add_collection(lc)

    ax.set_ylim(0, 1.05 * amax(am))

    cbar = fig.colorbar(im, ax=ax, ticks=[-pi, -pi / 2, 0, pi / 2, pi])
    cbar.ax.set_ylim(-pi, pi)
    cbar.ax.set_yticklabels(["-π", "-π / 2", "0", "π / 2", "π"])
    cbar.set_label("Argument")
    ax.set_title("Modulus of the Far field")

    show()
