"""
Sets the voltage to oscillate spatially around v_r with maximum
amplitude (v_th - v_r), using the mean average of three sine waves
with coprime periods.
"""

from numba import cuda
from math import sin

from modules import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
v_r = lookup["v_r"]
v_th = lookup["v_th"]
dx = lookup["dx"]

@cuda.jit()
def voltage_init(voltage_d):
    """Set voltage array to oscillate around v_r using sine waves."""
    n = cuda.grid(1)
    neurons_number = len(voltage_d)
    if n < neurons_number:
        pos = dx * n
        change = (1/3) * (sin(7*pos) + sin(11*pos) + sin(17*pos))
        voltage_d[n] = v_r + (v_th - v_r) * change
