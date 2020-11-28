from numpy import cos, pi, sin
from numpy.random import uniform
from numpy.testing import assert_allclose

import claudius


class Test_coord_polar:
    def test_vector(self):
        r, θ = uniform(0, 1, 16), uniform(pi, -pi, 16)
        x, y = r * cos(θ), r * sin(θ)

        r_, θ_ = claudius.to_polar(x, y)

        assert assert_allclose(r_, r) is None
        assert assert_allclose(θ_, θ) is None

    def test_array(self):
        r, θ = uniform(0, 1, (4, 4)), uniform(pi, -pi, (4, 4))
        x, y = r * cos(θ), r * sin(θ)

        r_, θ_ = claudius.to_polar(x, y)

        assert assert_allclose(r_, r) is None
        assert assert_allclose(θ_, θ) is None


class Test_coord_sph:
    def test_vector(self):
        r, θ, φ = uniform(0, 1, 16), uniform(pi, -pi, 16), uniform(0, pi / 2, 16)
        x, y, z = r * cos(θ) * sin(φ), r * sin(θ) * sin(φ), r * cos(φ)

        r_, θ_, φ_ = claudius.to_spheric(x, y, z)

        assert assert_allclose(r_, r) is None
        assert assert_allclose(θ_, θ) is None
        assert assert_allclose(φ_, φ) is None

    def test_array(self):
        r, θ, φ = (
            uniform(0, 1, (2, 2)),
            uniform(pi, -pi, (2, 2)),
            uniform(0, pi / 2, (2, 2)),
        )
        x, y, z = r * cos(θ) * sin(φ), r * sin(θ) * sin(φ), r * cos(φ)

        r_, θ_, φ_ = claudius.to_spheric(x, y, z)

        assert assert_allclose(r_, r) is None
        assert assert_allclose(θ_, θ) is None
        assert assert_allclose(φ_, φ) is None
