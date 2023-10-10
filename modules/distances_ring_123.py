"""
Device-side distance finding functions for ring geometry in 1D, 2D or 3D.

This is a geometry that is locally Euclidean, but with looping conditions.
In 1D, it's a ring.
In 2D it's a torus, or bagel, or the video game Asteroids.
You can also call it R/Z, for R the real numbers and Z the integers.

There are 2 straight-line paths between 2 points in 1D, and an infinite
number of straight-line paths between points in 2D and above.
However, since the space acts like a Euclidean space filled with infinite
tiled copies of itself, you only need to check the distance to the points
in the same or adjacent copies.  The copying is parallel to the axes,
so you can just project onto 1D in each coordinate and decide which of the
two directions is shorter.  Once you have the three numbers, apply
Pythagoras' Theorem to find the straight-line distance.

For the looping, we assume that the size in each direction is simply
dw * (neuron_count_w + 1)
for w = x, y, z
and ignore the even_offset_vw values.
"""

from numba import cuda
from math import pi, e

from modules.general.ParamPlus import lookup

dx = lookup["dx"]
dy = lookup["dy"]
dz = lookup["dz"]
dimension = lookup["dimension"]
neuron_count_x = lookup["neuron_count_x"]
neuron_count_y = lookup["neuron_count_y"]
neuron_count_z = lookup["neuron_count_z"]

#even_offset_yx = lookup["even_offset_yx"]
#even_offset_zx = lookup["even_offset_zx"]
#even_offset_zy = lookup["even_offset_zy"]

#-------------------------------------------------------------------------------

@cuda.jit(device = True)
def get_smallest_path(a, b, total_length):
    """
    Calculates the shorter 1D ring distance between two points.
    """
    d1 = abs(a - b)
    d2 = total_length - d1
    distance = d1 * (d1 < d2) + d2 * (d1 >= d2)
    return distance

@cuda.jit(device = True)
def get_distance_from(coordinates, n, x, y, z):
    """
    Calculates the Euclidean ring distance between a neuron and a given point.
    """
    #x_distance = (neuron_count_x + 1) * dx
    x_distance = neuron_count_x * dx
    distance = get_smallest_path(coordinates[n, 0], x, x_distance)
    if dimension > 1:
        distance = distance**2
        y_distance = neuron_count_y * dy
        distance += get_smallest_path(coordinates[n, 1], y, y_distance)**2
        if dimension > 2:
            z_distance = neuron_count_z * dz
            distance += get_smallest_path(coordinates[n, 2], z, z_distance)**2
        distance = distance**0.5
    return distance

@cuda.jit
def get_distance_between(coordinates, n_1, n_2):
    """Calculates the Euclidean ring distance between two neurons."""
    x = coordinates[n_1, 0]
    if dimension > 1:
        y = coordinates[n_1, 1]
    else:
        y = 0
    if dimension > 2:
        z = coordinates[n_1, 2]
    else:
        z = 0
    return get_distance_from(coordinates, n_2, x, y, z)

@cuda.jit
def get_distance_from_zero(coordinates, n):
    """
    Calculates the Euclidean distance between a neuron and the origin.
    """
    return get_distance_from(coordinates, n, 0, 0, 0)
