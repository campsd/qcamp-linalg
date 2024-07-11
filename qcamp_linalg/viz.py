#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import expm, logm
from .util import compute_norm, _transform_to_matrix
from typing import Union, Callable


def cartesian_plane(ax: plt.Axes, nx: int, ny: int):
    '''
    Plot a Cartesian plane with gridlines from -nx, ..., +nx and -ny, ..., +ny
    The origin is marked in red.

    Args
    ---
    ax (plt.Axes):
      matplotlib axes object
    nx (int):
      number of steps in positive (and negative) x-direction
    ny (int):
      number of steps in positive (and negative) y-direction

    Returns
    ---
      matplotlib figure object with gridlines and origin drawn
    '''
    ax.plot(
      0, 0,
      linestyle=None,
      marker='o', markerfacecolor='red', markeredgecolor='red', markersize=10
    )
    arrow_size = 0.25

    # (0, 1) arrow [up]
    ax.arrow(
      0, 0, 0, 1 - arrow_size, head_width=arrow_size, head_length=arrow_size,
      overhang=0.2, facecolor='red', edgecolor='red'
    )
    ax.annotate("(0, 1)", (0, 1), horizontalalignment="left", color="red")

    # (1, 0) arrow [right]
    ax.arrow(
      0, 0, 1 - arrow_size, 0, head_width=arrow_size, head_length=arrow_size,
      overhang=0.2, facecolor='blue', edgecolor='blue'
    )
    ax.annotate("(1, 0)", (1, 0), horizontalalignment="left", color="blue")
    ax.set_xlim(-nx, nx)
    ax.set_ylim(-ny, ny)
    ax.set_xticks(np.arange(-nx, nx+1, 1))
    ax.set_yticks(np.arange(-ny, ny+1, 1))
    ax.grid(alpha=0.25)
    ax.set_aspect('equal', adjustable='box')
    return None


def vector(
    ax: plt.Axes, vector: npt.ArrayLike, draw_arrow=True, alpha=1.0,
    annotate=False
  ):
    '''
    Add (draw) a vector to the Cartesian plane.

    Args
    ---
      ax (plt.Axes):
        matplotlib axes object
      vector (npt.ArrayLike):
        real-valued 2-vector
      draw_arrow (bool):
        draw the arrow or not
      alpha (float):
        opacity of arrow and dot [0,1]
      annotate (Union[False, 1, 2, 3]):
        False:  no annotation
        1:      only coordinates
        2:      only norm
        3:      coordinates and norm
    '''

    # cast vector to NumPy array
    vector = np.array(vector, dtype=np.float64)

    norm_vec = (vector[0]**2 + vector[1]**2)**0.5
    arrow_size = 0.35

    ax.plot(
      vector[0], vector[1],
      marker='o', markerfacecolor='orange', markeredgecolor='orange',
      markersize=5, alpha=alpha
    )
    if draw_arrow:
        ax.arrow(
          0, 0,
          vector[0]*(1 - arrow_size/norm_vec),
          vector[1]*(1 - arrow_size/norm_vec),
          head_width=arrow_size, head_length=arrow_size, overhang=0.2,
          facecolor='black', edgecolor='black', alpha=alpha
        )
    annpos = (
      vector[0] + 0.3 if vector[0] > 0 else vector[0] - 0.7,
      vector[1] + np.sign(vector[1]) * 0.2,
    )
    annalign = 'left' if vector[0] > 0 else 'right'
    if annotate == 1:
        ax.annotate(
          f'({vector[0]:.1f}, {vector[1]:.1f})',
          annpos, horizontalalignment=annalign
        )
    elif annotate == 2:
        ax.annotate(
          f'{compute_norm(vector):.1f}', annpos, horizontalalignment=annalign
        )
    elif annotate == 3:
        ax.annotate(
          f'({vector[0]:.1f}, {vector[1]:.1f})\n{compute_norm(vector):.1f}',
          annpos, horizontalalignment=annalign
        )
    return None


def cartesian_cloud(nx: int, ny: int):
    '''Generate a cartesian colorized point cloud.

    Args
    ----
      nx (int):
        number of steps in positive (and negative) x-direction
      ny (int):
        number of steps in positive (and negative) y-direction

    Returns
    ----
      structured NumPy array:
        'xy': coordinates
        'color': rgb colors
    '''
    n_points = (2*nx+1) * (2*ny+1)
    cloud = np.zeros(
      n_points, dtype=[('xy', float, (2,)), ('color', float, (3,))]
    )

    x_vals = np.linspace(-nx, nx, 2*nx+1)
    y_vals = np.linspace(-ny, ny, 2*ny+1)
    xy_grid = np.row_stack([[x, y] for x in x_vals for y in y_vals])
    cloud['xy'] = xy_grid

    # This function assigns a unique color based on position
    def colorizer(x, y):
        """
        Map x-y coordinates to a rgb color
        """
        r = min(1, 1-y/ny)
        g = min(1, 1+y/ny)
        b = 1/nx + x/nx**2
        return (r, g, b)

    colors = list(map(colorizer, xy_grid[:, 0], xy_grid[:, 1]))
    cloud['color'] = colors
    return cloud


def radial_cloud(ntheta: int, nr: int, rmax: float):
    """Generate a radial colorized point cloud.

    Args
    ----
      ntheta (int):
        number of points between [0, 2*pi] (angles)
      nr (int):
        number of points along rays
      rmax (float):
        maximum radius

    Returns
    ----
      structured NumPy array:
        'xy': coordinates
        'color': rgb colors
    """
    n_points = ntheta * nr
    cloud = np.zeros(
      n_points, dtype=[('xy', float, (2,)), ('color', float, (3,))]
    )

    r_vals = np.linspace(0, rmax, nr)
    theta_vals = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    rtheta_grid = np.row_stack(
      [[r, theta] for r in r_vals for theta in theta_vals]
    )
    xy_vals = np.row_stack(
      [
         [r*np.cos(theta), r*np.sin(theta)] for r in r_vals
         for theta in theta_vals
      ]
    )
    cloud['xy'] = xy_vals

    def colorizer(r, theta):
        """Map (r, theta) coordinates to color"""
        red = min(1, 1-theta/(2*np.pi))
        green = min(1, theta/(2*np.pi))
        blue = r/nr
        return (red, green, blue)

    colors = list(map(colorizer, rtheta_grid[:, 0], rtheta_grid[:, 1]))
    cloud['color'] = colors
    return cloud


def circle_cloud(ntheta: int):
    """Generate a colorized point cloud on the unit circle.

    Args
    ----
      ntheta (int):
        number of angles


    Returns
    ----
      structured NumPy array:
        'xy': coordinates
        'color': rgb colors
    """
    n_points = ntheta
    cloud = np.zeros(
      n_points, dtype=[('xy', float, (2,)), ('color', float, (3,))]
    )
    theta_vals = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    xy_vals = np.row_stack(
      [[np.cos(theta), np.sin(theta)] for theta in theta_vals]
    )
    cloud['xy'] = xy_vals

    def colorizer(theta):
        """Map (theta) coordinates to color"""
        red = min(1, 1-theta/(2*np.pi))
        green = min(1, theta/(2*np.pi))
        blue = theta/(4*np.pi)
        return (red, green, blue)

    colors = list(map(colorizer, theta_vals))
    cloud['color'] = colors
    return cloud


def animate(
    linear_transformation: Union[npt.ArrayLike, Callable],
    vectors: Union[npt.ArrayLike, None] = None,
    pointcloud: Union[npt.ArrayLike, None] = None,
    unitary: bool = False
):
    """Animate how a linear transformation changes a (set of) vector(s) and/or
    a cloud of points in the plane.

    Args
    ----
      linear_transformation (Callable or matrix):
        implementation of the 2D linear transformation, either as a python
        function or a 2x2 matrix.
      vectors (array):
        array of 2-vectors to be animated, shape: nvecs x 2
      pointcloud (structured array):
        cloud of points to be animated: 'xy' list of coordinates, 'color' list
        of rgb colors.
      unitary (boolean):
        true or false - determines the interpolation method from just linear
        interpolation if false, to an interpolation that preserves the norm.
    Returns
    -----
      animation object.
    """

    # cast vectors to NumPy array
    vectors = np.array(vectors, dtype=np.float64)

    # animation settings
    frames = 30
    interval = 100
    nx = 8
    ny = 8

    # generate plot
    fig, ax = plt.subplots()

    # early return
    if (vectors is None) and (pointcloud is None):
        cartesian_plane(ax, nx, ny)
        plt.show()
        return None

    # convert to matrix representation
    if callable(linear_transformation):
        linear_transformation = _transform_to_matrix(linear_transformation)

    log_mat = logm(linear_transformation)

    def _interpolate(mat, t, unitary=False):
        if unitary:
            return expm(t*log_mat)
        return (1 - t) * np.eye(2) + t * mat  # linear interpolation

    def init():
        return []

    def _animate(frame):
        ax.clear()
        cartesian_plane(ax, nx, ny)
        progress = frame / (frames - 1)
        current_mat = _interpolate(linear_transformation, progress, unitary)
        if vectors is not None:
            if frame == 0:
                for v in vectors:
                    vector(ax, v, draw_arrow=True, annotate=1)
            elif frame == frames - 1:
                vectors_current = (current_mat @ vectors.T).T
                for v, vv in zip(vectors, vectors_current):
                    vector(ax, v, draw_arrow=True, annotate=1, alpha=0.2)
                    vector(ax, vv, draw_arrow=True, annotate=1)
            else:
                vectors_current = (current_mat @ vectors.T).T
                for v, vv in zip(vectors, vectors_current):
                    vector(ax, v, draw_arrow=True, annotate=1, alpha=0.2)
                    vector(ax, vv, draw_arrow=True, annotate=False)
        if pointcloud is not None:
            pointcloud_current = (current_mat @ pointcloud['xy'].T).T
            ax.scatter(
              pointcloud_current[:, 0], pointcloud_current[:, 1],
              s=36, c=pointcloud['color'], edgecolor="none"
            )
        return ax

    ani = animation.FuncAnimation(
      fig, _animate, frames=frames, init_func=init, blit=False,
      interval=interval, repeat=False
    )
    return ani


def bloch_circle(
        state: Union[npt.ArrayLike, None] = None,
        gate: Union[npt.ArrayLike, None] = None
):
    """
    Visualize a vector/state both in cartesian plane and Bloch circle. The
    state has to be normalized and the global phase (sign) is ignored.
    Optionally, a gate can be provided and applied to the state.


    Args
    ----
      state (array):
        normalized qubit state (optional)
      gate (array)
        2 x 2 quantum gate (optional)
    """
    if state is not None:
        # cast to NumPy
        state = np.array(state, dtype=np.float64)

        if gate is not None:
            state = gate@state
        # validate state
        assert np.abs(compute_norm(state) - 1.0) < 1e-10
        phase = -1.0 if state[1] < 0 else 1.0
        state = phase * state
        state_color = (state[1], 0, np.abs(state[0]))

    # generate figure
    fig, axes = plt.subplots(1, 2)
    # left side: cartesian plane
    cartesian_plane(axes[0], 2, 2)
    # semi-circle:
    nthetas = 100
    thetas = np.linspace(0, np.pi, nthetas)
    colors = [(np.sin(t), 0, np.abs(np.cos(t))) for t in thetas]
    axes[0].scatter(
      np.cos(thetas), np.sin(thetas), s=18, c=colors, edgecolors="none"
    )
    # plot the state
    if state is not None:
        if phase < 0:
            axes[0].arrow(
              0, 0, phase*state[0]*0.85,
              phase*state[1]*0.85, head_width=0.15, head_length=0.15,
              overhang=0.2, facecolor='black', edgecolor='black', alpha=0.2
            )
        axes[0].arrow(
          0, 0, state[0]*0.85,
          np.abs(state[1])*0.85, head_width=0.15, head_length=0.15,
          overhang=0.2, facecolor=state_color, edgecolor=state_color
        )

    axes[0].annotate(
        "|+>", (0.75, 0.75), horizontalalignment="left", color=(0.5, 0, 0.5)
    )
    axes[0].annotate(
        "|->", (-0.75, 0.75), horizontalalignment="right", color=(0.5, 0, 0.5)
    )
    axes[0].set_xlim(-1.5, 1.5)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_title("Cartesian plane", {'fontsize': 16})

    # right side: bloch-circle
    arrow_size = 0.25
    # |0> arrow [up]
    axes[1].arrow(
        0, 0, 0, 1 - arrow_size,
        head_width=arrow_size, head_length=arrow_size, overhang=0.2,
        facecolor='blue', edgecolor='blue'
    )
    axes[1].annotate("|0>", (0, 1.1), horizontalalignment="left", color="blue")
    # |1> arrow [down]
    axes[1].arrow(
        0, 0, 0, -1 + arrow_size,
        head_width=arrow_size, head_length=arrow_size, overhang=0.2,
        facecolor='red', edgecolor='red'
    )
    axes[1].annotate(
        "|1>", (0, -1.25), horizontalalignment="left", color="red"
    )
    axes[1].scatter(
        np.cos(2*thetas+np.pi/2), np.sin(2*thetas+np.pi/2), s=18, c=colors,
        edgecolors="none"
    )
    # plot state
    if state is not None:
        state_angle = np.arctan2(state[1], state[0])
        axes[1].arrow(
          0, 0, np.cos(2*state_angle+np.pi/2)*0.85,
          np.sin(2*state_angle+np.pi/2)*0.85, head_width=0.15,
          head_length=0.15, overhang=0.2, facecolor=state_color,
          edgecolor=state_color
        )
    axes[1].annotate(
        "|+>", (-1.1, 0), horizontalalignment="right", color=(0.5, 0, 0.5)
    )
    axes[1].annotate(
        "|->", (1.1, 0), horizontalalignment="left", color=(0.5, 0, 0.5)
    )
    axes[1].set_title("Bloch circle", {'fontsize': 16})
    axes[1].set_xticks(np.arange(-2, 3, 1))
    axes[1].set_yticks(np.arange(-2, 3, 1))
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].grid(alpha=0.25)
    axes[1].set_aspect('equal', adjustable='box')
    plt.show()
