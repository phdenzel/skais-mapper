# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rotation operators for point cloud arrays."""

from typing import TypeVar
import numpy as np


TRotation = TypeVar("TRotation", bound="R")


class R:
    """Class of rotation operators."""

    def __init__(self, omega: np.ndarray | None = None):
        """Initialize rotator object.

        Args:
            omega: Rotation angle in
        """
        self.omega = omega

    def __call__(self, arr: np.ndarray, **kwargs) -> np.ndarray:
        """Rotate the input according to this rotation operator.

        Args:
            arr: The array to be rotated
            kwargs: Dummy keyword arguments

        Returns:
            arr: The rotated array
        """
        if self.omega is None:
            self.omega = self._x(0)
        arr = arr @ self.omega.transpose()
        return arr

    def __mul__(self, other: TRotation) -> TRotation:
        """Combine rotation operators through multiplication.

        Args:
            other: Another rotation object to be multiplied.

        Returns:
            obj: a new instance of R
        """
        return self.__class__(self.omega @ other.omega)

    @classmethod
    def x(cls, theta: float, degrees: bool = True) -> TRotation:
        """Rotation operator about the current x-axis by angle theta.

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        omega = cls._x(theta, degrees=degrees)
        return cls(omega)

    @classmethod
    def y(cls, theta: float, degrees: bool = True) -> TRotation:
        """Rotation operator about the current y-axis by angle theta.

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        omega = cls._y(theta, degrees=degrees)
        return cls(omega)

    @classmethod
    def z(cls, theta: float, degrees: bool = True) -> TRotation:
        """Rotation operator about the current z-axis by angle theta.

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        omega = cls._z(theta, degrees=degrees)
        return cls(omega)

    @staticmethod
    def _x(theta: float, degrees: bool = True) -> np.ndarray:
        """Rotation matrix about the current x-axis by angle theta.

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        if degrees:
            theta *= np.pi / 180.0
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

    @staticmethod
    def _y(theta: float, degrees: bool = True) -> np.ndarray:
        """Rotation matrix about the current y-axis by angle theta.

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        if degrees:
            theta *= np.pi / 180.0
        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    @staticmethod
    def _z(theta: float, degrees: bool = True) -> np.ndarray:
        """Rotation matrix about the current z-axis by angle theta.

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        if degrees:
            theta *= np.pi / 180.0
        return np.array(
            [
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
                [0, 0, 1],
            ]
        )
