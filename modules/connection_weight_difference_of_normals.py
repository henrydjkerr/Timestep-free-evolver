"""
Device-side functions for calculating connection strength.
This version expresses the 'Mexican hat' shape as a difference between
two Gaussian or Normal curves with mean zero.
"""

from numba import cuda

from modules import device_gaussian
from modules.general import Control
lookup = Control.lookup

sigma_e = lookup["normal_sigma_e"]
sigma_i = lookup["normal_sigma_i"]
strength_e = lookup["normal_strength_e"]
strength_i = lookup["normal_strength_i"]

dimension = lookup["dimension"]
dx = lookup["dx"]
dy = lookup["dy"]
dz = lookup["dz"]

density_factor = 1
dims = [dx, dy, dz]
for k in range(dimension):
    density_factor *= dims[k]

#-------------------------------------------------------------------------------

@cuda.jit(device = True)
def connection_weight(diff):
    """
    Generates a hat-shaped function the difference between two Gaussian curves.
    """
    signal = strength_e * device_gaussian.curve(diff, sigma_e) \
             - strength_i * device_gaussian.curve(diff, sigma_i)
    signal *= density_factor
    return signal
