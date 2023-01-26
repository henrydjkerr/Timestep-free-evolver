"""
Sets the input current array to a middle-centred Gaussian curve without
any uniform baseline input.
"""

from numba import cuda
from math import pi, e

from modules import device_gaussian
from modules import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
sigma = lookup["input_sigma"]
strength = lookup["input_strength"]

@cuda.jit()
def input_init(input_strength_d, coordinates_d):
    """Initialises input_strength array with a bellcurve"""
    n = cuda.grid(1)
    if n < neurons_number:
        position = Control.d.get_distance_from_zero(coordinates_d, n)
        input_strength_d[n] =  strength * device_gaussian.curve(position, sigma)
