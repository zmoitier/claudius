import matplotlib.pyplot as plt
import numpy as np

from rot_inv_scattering import *

εc = 1.1
μc = 1
k = 4
T = 2

M = M_trunc(k, T)
# sol = disk_dir.solution(k, M)
sol = disk_neu.solution(k, M)
# sol = disk_trans.solution(εc, μc, k, M)

θ = np.linspace(0, 2 * np.pi, num=128)
ff = sc_far_field(k, sol, θ)
# print(ff)

Irp = np.where(np.real(ff) >= 0)
Irn = np.where(np.real(ff) < 0)

plt.polar(θ, np.abs(ff), "C2.")
# plt.polar(θ[Irn], -np.real(ff[Irn]), "C3.")
plt.grid(True)
plt.show()
