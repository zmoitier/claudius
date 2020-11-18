from sys import exit

from numpy import cos, exp


def incident_field(k, c1, c2, c3, coord):
    """Compute the plane wave incident field"""

    if coord == "xyz":
        return exp(1j * k * c3)

    if coord == "rθφ":
        return exp(1j * k * c1 * cos(c3))

    exit("coord = 'xyz' for Cartesian or 'rθφ' for Spheric")
