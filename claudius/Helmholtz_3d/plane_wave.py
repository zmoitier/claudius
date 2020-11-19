from sys import exit

from numpy import cos, exp, ones_like


def incident_field(wavenum, coo_xr, coo_yt, coo_zp, type_coord):
    """Compute the plane wave incident field"""

    if type_coord == "cartesian":
        return exp(1j * wavenum * coo_zp) * ones_like(coo_xr) * ones_like(coo_yt)

    if type_coord == "spherical":
        return exp(1j * wavenum * coo_xr * cos(coo_zp)) * ones_like(coo_yt)

    exit(
        """Unsupported type_coord use either:
    type_coord = "cartesian" for Cartesian coordinates
    type_coord = "spherical" for Spherical coordinates"""
    )
