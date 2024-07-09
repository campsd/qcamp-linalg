#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.typing as npt


def compute_norm(vector: npt.ArrayLike):
  # Compute the norm of a vector
  return np.linalg.norm(vector, 2)

def compute_angle_between(vector1: npt.ArrayLike, vector2: npt.ArrayLike):
  # Compute the angle (in degrees) between two vectors
  v1 = vector1 / compute_norm(vector1)
  v2 = vector2 / compute_norm(vector2)
  return (180 / np.pi) * np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])
x_gate = np.array([[0, 1], [1, 0]])
hadamard_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
