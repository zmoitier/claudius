import cProfile
import io
import os
import pstats
import sys

import numpy as np
from scipy.special import hankel1, spherical_jn, spherical_yn

import claudius as acs


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


h1 = lambda l, z: spherical_jn(l, z) + 1j * spherical_yn(l, z)
h2 = lambda l, z: np.sqrt(np.pi / (2 * z)) * hankel1(l + 0.5, z)

n = 10
z = np.linspace(1, 50, num=10)

print(doprofile(h1, "profiling.dat", n, z))
print(doprofile(h2, "profiling.dat", n, z))
