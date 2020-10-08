import matplotlib.pyplot as plt
import numpy as np

from rot_inv_scattering import *

k = 2
T = 2

sol = disk_neu.solution(k, T)

r = np.ones(16)
θ = np.linspace(0, 2 * np.pi, num=16)
us = total_field(k, sol, r, θ, "rθ")
print(us)
