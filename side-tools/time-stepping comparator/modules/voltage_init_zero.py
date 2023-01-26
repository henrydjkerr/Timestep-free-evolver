"""
Initialises the voltage array on device with all zeros.
"""

from numba import cuda

from modules import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]

@cuda.jit()
def voltage_init(voltage_d, coordinates_d):
    """Set voltage array to all zeros."""
    n = cuda.grid(1)
    if n < neurons_number:
        voltage_d[n] = 0
