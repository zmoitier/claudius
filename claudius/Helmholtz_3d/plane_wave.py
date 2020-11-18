from sys import exit

from numpy import cos, exp


def incident_field(wavenum, coo_xr, coo_yt, coo_zp, type_coord):
    """Compute the plane wave incident field"""

    if type_coord == "cartesian":
        return exp(1j * wavenum * coo_zp)

    if type_coord == "spherical":
        return exp(1j * wavenum * coo_xr * cos(coo_zp))

    exit(
        """Unsupported type_coord use either:
    type_coord = "cartesian" for Cartesian coordinates
    type_coord = "spherical" for Spherical coordinates"""
    )
