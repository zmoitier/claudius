import matplotlib.pyplot as plt
import numpy as np
from context import accoster as acs

εc = 1.1
μc = 1
k = 1

θ = np.linspace(0, 2 * np.pi, num=255)
ff = [acs.disk_dir.far_field(k, θ), acs.disk_neu.far_field(k, θ)]
name = ["Dirichlet", "Neumann"]

fig, ax = plt.subplots(1, 2, subplot_kw={"polar": True})

for i in range(2):
    am = np.abs(ff[i])
    ph = np.angle(ff[i])

    ax[i].grid(True, zorder=1)
    ax[i].plot(θ, am, "k", alpha=0.5, zorder=2)
    data = ax[i].scatter(
        θ[:-1],
        am[:-1],
        s=5,
        c=ph[:-1],
        cmap="twilight_shifted",
        vmin=-np.pi,
        vmax=np.pi,
        zorder=3,
    )

    cbar = fig.colorbar(data, ax=ax[i], ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cbar.ax.set_yticklabels(["-π", "-π / 2", "0", "π / 2", "π"])
    cbar.set_label("Argument")

    ax[i].set_title(f"{name[i]}: Modulus")

plt.show()
