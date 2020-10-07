import cProfile
import io
import os
import pstats
import sys

import numpy as np

import rot_inv_scattering.disk as disk


def doprofile(func, filename, *l):
    pr = cProfile.Profile()
    pr.enable()  # début du profiling
    func(*l)  # appel de la fonction
    pr.disable()  # fin du profiling
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    rem = os.path.normpath(os.path.join(os.getcwd(), "..", "..", ".."))
    res = s.getvalue().replace(rem, "")
    res = res.replace(sys.base_prefix, "").replace("\\", "/")
    # ps.dump_stats(filename)
    return res


εc = -1.1
μc = 1
k = 8

N = 256
T = 2
X, Y = np.meshgrid(np.linspace(-T, T, num=N), np.linspace(-T, T, num=N))

if len(sys.argv) > 1:
    print("Scattered field")
    r = doprofile(disk.scattered_field, "profiling.dat", εc, μc, k, T, X, Y, "xy")
else:
    print("Total field")
    r = doprofile(disk.total_field, "profiling.dat", εc, μc, k, T, X, Y, "xy")
print(r)
