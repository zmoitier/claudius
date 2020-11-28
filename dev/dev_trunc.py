from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, spherical_jn

from claudius import trunc_H2d, trunc_H3d

if argv[1] == "2H":
    fct = jv
    trunc = trunc_H2d

if argv[1] == "3H":
    fct = lambda l, x: np.sqrt(4 * np.pi * (2 * l + 1)) * spherical_jn(l, x)
    trunc = trunc_H3d

nb = 128
mMax = 45
m = np.arange(mMax + 1)
xMax = 25
x = np.linspace(xMax, 0, num=nb, endpoint=False)[::-1]
X, M = np.meshgrid(x, m)

J = np.log10(np.abs(fct(M, X)))
j = np.array(list(map(lambda v: trunc(v, 1), x)))

plt.pcolormesh(X, M, J, shading="auto", vmin=-6, vmax=0)
plt.plot(x, j, "k")

plt.colorbar()

plt.show()
