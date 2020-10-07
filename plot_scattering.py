from sys import argv

import matplotlib.pyplot as plt
import numpy as np

import rot_inv_scattering.disk as disk

εc = float(argv[1])
μc = float(argv[2])
k = float(argv[3])

N = 64
T = 2
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))
if len(argv) > 4:
    U = disk.scattered_field(εc, μc, k, T, X, Y, "xy")
    which = "Scattered field"
else:
    U = disk.total_field(εc, μc, k, T, X, Y, "xy")
    which = "Total field"

n = 128
θ = np.linspace(0, 2 * np.pi, num=n)
co, si = np.cos(θ), np.sin(θ)

plt.subplot(2, 2, 1)
UrMax = np.amax(np.abs(np.real(U)))
plt.pcolormesh(X, Y, np.real(U), shading="gouraud", cmap="RdBu_r")
plt.plot(co, si, "k--", lw=2)
plt.axis("equal")
plt.clim(-UrMax, UrMax)
plt.colorbar()
plt.title("Real part")

plt.subplot(2, 2, 2)
UiMax = np.amax(np.abs(np.imag(U)))
plt.pcolormesh(X, Y, np.imag(U), shading="gouraud", cmap="RdBu_r")
plt.plot(co, si, "k--", lw=2)
plt.axis("equal")
plt.clim(-UiMax, UiMax)
plt.colorbar()
plt.title("Imaginary part")

plt.subplot(2, 2, 3)
UMax = np.amax(np.abs(U))
plt.pcolormesh(X, Y, np.abs(U), shading="gouraud")
plt.plot(co, si, "k--", lw=2)
plt.axis("equal")
plt.clim(0, UMax)
plt.colorbar()
plt.title("Modulus")

plt.subplot(2, 2, 4)
plt.pcolormesh(X, Y, np.angle(U), shading="gouraud", cmap="twilight_shifted_r")
plt.plot(co, si, "k--", lw=2)
plt.axis("equal")
plt.clim(-np.pi, np.pi)
plt.colorbar()
plt.title("Argument")

plt.suptitle(
    fr"{which}: $\varepsilon_{{\mathsf{{c}}}} \equiv {εc}$, $\mu_{{\mathsf{{c}}}} \equiv {μc}$, and $k = {k}$"
)

plt.tight_layout()

plt.show()
