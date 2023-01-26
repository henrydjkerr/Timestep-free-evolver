"""
Device-side distance finding functions for R-spaces, 1/2/3D.
"""

from numba import cuda
from math import pi, e

from modules.ParamPlus import lookup

dx = lookup["dx"]
dy = lookup["dy"]
dz = lookup["dz"]
dimension = lookup["dimension"]
neuron_count_x = lookup["neuron_count_x"]
neuron_count_y = lookup["neuron_count_y"]
neuron_count_z = lookup["neuron_count_z"]
even_offset_yx = lookup["even_offset_yx"]
even_offset_zx = lookup["even_offset_zx"]
even_offset_zy = lookup["even_offset_zy"]

#-------------------------------------------------------------------------------

@cuda.jit(device = True)
def get_distance_from(coordinates, n, x, y, z):
    """Calculates the Euclidean distance between a neuron and a given point."""
    if dimension == 1:
        distance = abs(coordinates[n, 0] - x)
    elif dimension > 1:
        distance2 =   (coordinates[n, 0] - x)**2 \
                    + (coordinates[n, 1] - y)**2
        if dimension == 3:
            distance2 += (coordinates[n, 2] - z)**2
        distance = distance2**0.5
    return distance

@cuda.jit
def get_distance_between(coordinates, n_1, n_2):
    """Calculates the Euclidean distance between two neurons."""
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
    return get_distance_from(coordinates, n, 0, 0, 0)
