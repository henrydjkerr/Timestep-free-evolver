"""
Just a gaussian curve function for the device-side.
"""

from numba import cuda
from math import pi, e


@cuda.jit(device = True)
def curve(distance, sigma):
    """Generates a Gaussian curve"""
    value = (1 / (sigma * (2 * pi) ** 0.5)) * e**(-0.5 * (distance / sigma)**2)
    return value
