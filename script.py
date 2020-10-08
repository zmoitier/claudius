import matplotlib.pyplot as plt
import numpy as np

from rot_inv_scattering import *

εc = 1.1
μc = 1
k = 2
T = 2

sol = disk_trans.solution(εc, μc, k, T)

r = np.ones(16)
θ = np.linspace(0, 2 * np.pi, num=16)
us = total_field(k, sol, r, θ, "rθ")
print(us)
