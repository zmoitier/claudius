from sys import argv

import matplotlib.pyplot as plt
import numpy as np

import claudius_dev as claudius

obs = claudius.create_obstacle("Disk", "Penetrable", [1, 2], [1, 2], [3, 4])
print(obs)

print(obs.sig_rho[0][0](np.zeros(5)))
print(obs.sig_rho[0][1](np.zeros(5)))
print(obs.sig_rho[1][0](np.zeros(5)))
print(obs.sig_rho[1][1](np.zeros(5)))
