import matplotlib.pyplot as plt
import numpy as np

import rot_inv_scattering.disk as disk

εc = -1.1
μc = 1
k = 2

N = 64
T = 2
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))
print(disk.total_field(εc, μc, k, T, X, Y, "xy"))
