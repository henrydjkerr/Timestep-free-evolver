"""
Leaves the voltage array untouched.  Intended for use with imported data.
"""

from numba import cuda

from modules.general import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]

@cuda.jit()
def array_init(voltage_d, coordinates_d):
    """Do nothing."""
    pass
