"""
Initialises the voltage array on device with random independent values
between v_r and v_th.
Reliant on setting up the host-side voltage array with random values in
the first place, so it's not doing the heavy lifting.
"""


from numba import cuda

from modules.general import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
v_r = lookup["v_r"]
v_th = lookup["v_th"]

@cuda.jit()
def array_init(voltage_d, coordinates_d):
    """Set voltage array to randomly distribute between v_r and v_th."""
    n = cuda.grid(1)
    if n < neurons_number:
        voltage_d[n] = v_r + (v_th - v_r) * voltage_d[n]
