"""Change coordinates"""
from numpy import arccos, arctan2, hypot, sqrt


def to_polar(coo_x, coo_y):
    """
    r, θ = to_polar(x, y)

    Change Cartesian coordinates to Polar coordinates.

    Parameters
    ----------
    x : array_like
        first Cartesian coordinate
    y : array_like
        seconde Cartesian coordinate

    Returns
    -------
    r : ndarray
        radial coordinate
    θ : ndarray
        angular coordinate
    """

    return (hypot(coo_x, coo_y), arctan2(coo_y, coo_x))


def to_spheric(coo_x, coo_y, coo_z):
    """
    r, θ, φ = to_spheric(x, y, z)

    Change Cartesian coordinates to Spherical coordinates.

    Parameters
    ----------
    x : array_like
        first Cartesian coordinate
    y : array_like
        seconde Cartesian coordinate
    z : array_like
        third Cartesian coordinate

    Returns
    -------
    r : ndarray
        radial distance
    θ : ndarray
        azimuthal angle
    φ : ndarray
        polar angle
    """

    coo_r = sqrt(coo_x * coo_x + coo_y * coo_y + coo_z * coo_z)
    return (coo_r, arctan2(coo_y, coo_x), arccos(coo_z / coo_r))
