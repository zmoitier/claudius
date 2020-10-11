from numpy import pi


def normL2_rad(sol):
    n = len(sol.func)
    if n == 0:
        pass
        return 0
    elif n == 1:
        return 0
    elif n == 2:
        δ = sol.radii[0]
        return 0
    else:
        exit("Unsopported number of function in sol.func")


def normL2_disk(k, sol):
    pass
