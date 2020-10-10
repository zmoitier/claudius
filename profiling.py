import cProfile
import io
import os
import pstats
import sys

import numpy as np
from rot_inv_scattering import *


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
μc = -1.1
k = 5
T = 2

r = doprofile(disk_trans.solution, "profiling.dat", εc, μc, k, T)
print(r)
