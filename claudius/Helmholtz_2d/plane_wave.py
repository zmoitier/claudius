from sys import exit

from numpy import exp, ones_like, sin


def incident_field(wavenum, coo_xr, coo_yt, type_coord):
    """Compute the plane wave incident field"""

    if type_coord == "cartesian":
        return exp(1j * wavenum * coo_yt) * ones_like(coo_xr)

    if type_coord == "polar":
        return exp(1j * wavenum * coo_xr * sin(coo_yt))

    exit(
        """Unsupported type_coord use either:
    type_coord = "cartesian" for Cartesian coordinates
    type_coord = "polar" for Polar coordinates"""
    )
