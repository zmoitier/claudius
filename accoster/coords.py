from sys import exit

from numpy import arctan2, hypot


def to_polar(c1, c2, coord):
    if coord == "xy":
        return (hypot(c1, c2), arctan2(c2, c1))
    elif coord == "rθ":
        return (c1, c2)
    else:
        exit("coord = 'xy' for Cartesian or 'rθ' for Polar")
