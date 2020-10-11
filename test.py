import matplotlib.pyplot as plt
import numpy as np

import accoster as acs

δ = 0.5
εc = -1.1
μc = 1
k = 4
T = 2

M = acs.M_trunc(k, T)
sol = acs.disk_dir.solution(k, M)
