"""
Device functions for calculating the voltage and its derivative at given times.
"""

from numba import cuda
from math import pi, e

from modules.ParamPlus import lookup

synapse_decay = lookup["synapse_decay"]

#------------------------------------------------------------------------------

@cuda.jit(device = True)
def get_vt(t, v_0, s_0, I):
    """Calculates v(t) from initial conditions"""
    value =  I + (s_0 / (1 - synapse_decay)) * e**(-synapse_decay * t) \
            + (v_0 - I - (s_0 / (1 - synapse_decay))) * e**(-t)
    return value

@cuda.jit(device = True)
def get_dvdt(t, v_actual, s_0, I):
    """Calculates the derivative of v(t) given v(t), s(0) and I"""
    value = I - v_actual + s_0 * e**(-synapse_decay * t)
    return value

