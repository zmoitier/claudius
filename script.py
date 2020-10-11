import matplotlib.pyplot as plt
import numpy as np

import accoster as acs

δ = 0.5
εc = -1.1
μc = 1
k = 4
T = 2

M = M_trunc(k, T)
sol = ann_cts.solution(δ, εc, μc, k, M)

x = np.linspace(-T, T, num=32)
y = np.linspace(-T, T, num=32)
X, Y = np.meshgrid(x, y)
u = sc_field(k, sol, x, y, "xy")
print(sol.coeff)
print(sol.func)
print(sol.radii)
