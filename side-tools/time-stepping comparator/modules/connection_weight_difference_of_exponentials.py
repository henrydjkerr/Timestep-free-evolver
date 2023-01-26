"""
Device-side functions for calculating connection strength.
This version expresses the connection strength as the difference between
two exponenial functions with negative exponents.
"""

from numba import cuda
from math import pi, e

from modules.ParamPlus import lookup

strength_e = lookup["exp_strength_e"]
strength_i = lookup["exp_strength_i"]
exp_e = lookup["exp_exponent_e"]
exp_i = lookup["exp_exponent_i"]

dimension = lookup["dimension"]
dx = lookup["dx"]
dy = lookup["dy"]
dz = lookup["dz"]
beta = lookup["synapse_decay"]

density_factor = beta
dims = [dx, dy, dz]
for k in range(dimension):
    density_factor *= dims[k]
#This is consistent within a single dimension, but not across dimensions
#E.g. consider using the radial symmetry to collapse down a single neuron's
#view of a 2D population into 1D; instead of uniform density, the density
#increases proportionally with distance.

#-------------------------------------------------------------------------------

#@cuda.jit(device = True)
def connection_weight(diff):
    """
    Generates a hat-shaped function from the difference between
    two exponential curves with negative exponent.
    """
    signal = density_factor * (strength_e * e**(-exp_e * abs(diff)) 
                               - strength_i * e**(-exp_i * abs(diff)))
    return signal
