from sys import exit

from numpy import arctan2, hypot, sqrt


def to_polar(c1, c2, coord):
    if coord == "xy":
        return (hypot(c1, c2), arctan2(c2, c1))
    elif coord == "rθ":
        return (c1, c2)
    else:
        exit("coord = 'xy' for Cartesian or 'rθ' for Polar")


def to_spheric(c1, c2, c3, coord):
    if coord == "xyz":
        r = sqrt(c1 ** 2 + c2 ** 2 + c3 ** 2)
        return (r, arccos(c3 / r), arctan2(c2, c1))
    elif coord == "rθϕ":
        return (c1, c2, c3)
    else:
        exit("coord = 'xyz' for Cartesian or 'rθϕ' for Spheric")
