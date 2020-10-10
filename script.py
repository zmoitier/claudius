import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from rot_inv_scattering import *

εc = -1.1
μc = 1
k = 4
T = 2

M = M_trunc(k, T)
# sol = disk_dir.solution(k, M)
# sol = disk_neu.solution(k, M)
sol = disk_trans.solution(εc, μc, k, M)

θ = np.linspace(0, 2 * np.pi, num=128)
ff = sc_far_field(k, sol, θ)

am = np.abs(ff)
ph = np.angle(0.5 * (ff[1:] + ff[:-1]))

points = np.array([θ, am]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, ax = plt.subplots(subplot_kw={"polar": True})

lc = LineCollection(
    segments, lw=3, cmap="twilight_shifted", norm=plt.Normalize(-np.pi, np.pi)
)
lc.set_array(ph)
line = ax.add_collection(lc)

cbar = fig.colorbar(line, ax=ax, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
cbar.ax.set_yticklabels(["-π", "-π / 2", "0", "π / 2", "π"])
cbar.set_label("Argument")

ax.set_rlim(0, 1.05 * np.amax(am))
ax.set_title("Modulus")

plt.show()
