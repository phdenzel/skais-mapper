# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.cosmology module."""

import numpy as np
from skais_mapper.rotations import R

def test_init_and_identity_omega():
    """Test initialization and default omega."""
    # Default init (omega=None)
    r = R()
    assert r.omega is None
    arr = np.eye(3)
    result = r(arr)
    np.testing.assert_array_almost_equal(result, arr)
    assert r.omega is not None  # Should be set after first call


def test_call_with_given_omega():
    """"Test calling with a specific omega."""
    omega = np.eye(3)
    r = R(omega=omega)
    arr = np.eye(3)
    result = r(arr)
    np.testing.assert_array_almost_equal(result, arr)


def test_call_with_rotation():
    """Test calling with a rotation matrix."""
    theta = 90
    r = R.x(theta)
    arr = np.array([[0, 1, 0]])
    # Should rotate (0, 1, 0) to (0, 0, 1)
    rotated = r(arr)
    np.testing.assert_array_almost_equal(
        rotated, np.array([[0, 0, 1]]), decimal=6
    )


def test_mul_combines_rotations():
    """Test multiplication of two rotations."""
    r1 = R.x(90)
    r2 = R.y(90)
    r3 = r1 * r2
    # Should be type R and omega = r1.omega @ r2.omega
    assert isinstance(r3, R)
    np.testing.assert_array_almost_equal(r3.omega, r1.omega @ r2.omega)


def test_x_y_z_classmethods_and_static_equivalence():
    """Test class methods and static methods for x, y, z rotations."""
    theta = 45
    rx = R.x(theta)
    expected = R._x(theta)
    np.testing.assert_array_almost_equal(rx.omega, expected)
    ry = R.y(theta)
    expected = R._y(theta)
    np.testing.assert_array_almost_equal(ry.omega, expected)
    rz = R.z(theta)
    expected = R._z(theta)
    np.testing.assert_array_almost_equal(rz.omega, expected)


def test_rotation_static_degrees_vs_radians():
    """Test static methods for x, y, z rotations with degrees and radians."""
    theta_deg = 90
    theta_rad = np.pi / 2
    rx_deg = R._x(theta_deg, degrees=True)
    rx_rad = R._x(theta_rad, degrees=False)
    np.testing.assert_array_almost_equal(rx_deg, rx_rad)
    ry_deg = R._y(theta_deg, degrees=True)
    ry_rad = R._y(theta_rad, degrees=False)
    np.testing.assert_array_almost_equal(ry_deg, ry_rad)
    rz_deg = R._z(theta_deg, degrees=True)
    rz_rad = R._z(theta_rad, degrees=False)
    np.testing.assert_array_almost_equal(rz_deg, rz_rad)


def test_rotation_on_multiple_vectors():
    """Test rotation on multiple vectors."""
    r = R.x(90)
    arr = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    expected = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, -1, 0]
    ])
    np.testing.assert_array_almost_equal(r(arr), expected, decimal=6)
