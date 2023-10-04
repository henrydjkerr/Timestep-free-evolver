"""
Sets the input current array be equal everywhere.
"""

from numba import cuda
from math import pi, e

from modules import device_gaussian
from modules.general import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
strength = lookup["input_base_before"]

@cuda.jit()
def array_init(input_strength_d, coordinates_d):
    """Initialises input_strength array with a constant value."""
    n = cuda.grid(1)
    if n < neurons_number:
        input_strength_d[n] = strength
