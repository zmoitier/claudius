from sys import exit

from numpy import arccos, arctan2, hypot, sqrt


def to_polar(coo_xr, coo_yt, type_coord):
    if type_coord == "cartesian":
        return (hypot(coo_xr, coo_yt), arctan2(coo_yt, coo_xr))

    if type_coord == "polar":
        return (coo_xr, coo_yt)

    exit(
        """Unsupported type_coord use either:
    type_coord = "cartesian" for Cartesian coordinates
    type_coord = "polar" for Polar coordinates"""
    )


def to_spheric(coo_xr, coo_yt, coo_zp, type_coord):
    if type_coord == "cartesian":
        r = sqrt(coo_xr ** 2 + coo_yt ** 2 + coo_zp ** 2)
        return (r, arctan2(coo_yt, coo_xr), arccos(coo_zp / r))

    if type_coord == "spherical":
        return (coo_xr, coo_yt, coo_zp)

    exit(
        """Unsupported type_coord use either:
    type_coord = "cartesian" for Cartesian coordinates
    type_coord = "spherical" for Spherical coordinates"""
    )
