#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.typing as npt
from typing import Callable


def compute_norm(vector: npt.ArrayLike):
    """Compute the norm of a vector.

    Args
    ---
      vector (array):
        input 2-vector

    Returns
    ----
      norm of the input vector
    """
    return np.linalg.norm(vector, 2)


def compute_angle_between(vector1: npt.ArrayLike, vector2: npt.ArrayLike):
    """ Compute the angle (in degrees) between two vectors.

    Args
    ---
      vector1 (array): first 2-vector
      vector2 (array): second 2-vector

    Returns
    ----
      angle  between vector in degrees
    """
    v1 = vector1 / compute_norm(vector1)
    v2 = vector2 / compute_norm(vector2)
    return (180 / np.pi) * np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def _transform_to_matrix(linear_transformation: Callable):
    """Convert a linear transformation defined as a python function into a
    2 x 2 matrix representation."""
    mat = np.zeros((2, 2), dtype=np.float64)
    mat[:, 0] = linear_transformation(np.array([1., 0.]))
    mat[:, 1] = linear_transformation(np.array([0., 1.]))
    return mat


ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])
x_gate = np.array([[0, 1], [1, 0]])
hadamard_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
