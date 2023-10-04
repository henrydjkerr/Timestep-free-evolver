"""
Device-side distance finding functions for ring geometry in 1D.
That is, a line where you loop from one end to the other, and distance
is found by travelling along the line in whichever direction is shortest.
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
even_offset_yx = lookup["even_offset_yx"]
even_offset_zx = lookup["even_offset_zx"]
even_offset_zy = lookup["even_offset_zy"]

#-------------------------------------------------------------------------------

@cuda.jit(device = True)
def get_distance_from(coordinates, n, x, y, z):
    """
    Calculates the 1D ring distance between a neuron and a given point.
    """
    d_1 = abs(coordinates[n, 0] - x)
    d_2 = ((neuron_count_x + 1) * dx) - d_1
    distance = d_1 * (d_1 < d_2) + d_2 * (d_1 >= d_2)
    return distance

@cuda.jit
def get_distance_between(coordinates, n_1, n_2):
    """Calculates the Euclidean distance between two neurons."""
    x = coordinates[n_1, 0]
    return get_distance_from(coordinates, n_2, x, 0, 0)

@cuda.jit
def get_distance_from_zero(coordinates, n):
    """
    Calculates the Euclidean distance between a neuron and the origin.
    """
    x = 0.5 * ((neuron_count_x - 1) * dx)
    return get_distance_from(coordinates, n, x, 0, 0)
