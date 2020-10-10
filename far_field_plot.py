import matplotlib.pyplot as plt
import numpy as np

from rot_inv_scattering import *

εc = -1.1
μc = 1
k = 4
T = 2

M = M_trunc(k, T)
# sol = disk_dir.solution(k, M)
# sol = disk_neu.solution(k, M)
sol = disk_trans.solution(εc, μc, k, M)

θ = np.linspace(0, 2 * np.pi, num=256, endpoint=False)
ff = sc_far_field(k, sol, θ)

am = np.abs(ff)
ph = np.angle(ff)

fig, ax = plt.subplots(subplot_kw={"polar": True})

data = ax.scatter(
    θ, am, s=5, c=ph, cmap="twilight_shifted", vmin=-np.pi, vmax=np.pi, zorder=2
)

cbar = fig.colorbar(data, ax=ax, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
cbar.ax.set_yticklabels(["-π", "-π / 2", "0", "π / 2", "π"])
cbar.set_label("Argument")

ax.set_title("Modulus")

plt.show()
