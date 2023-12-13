"""
Sheap subthreshold 'checker' that doesn't really check.
Just figures out a time when the firing event should have either happened,
or will never happen.

Haven't put in error checks for the divide-by-zero error if q = 0.
Or if (p - synapse_decay)**2 = q**2.
"""

from numba import cuda
from math import log

from modules.general import Control
lookup = Control.lookup

synapse_decay = lookup["synapse_decay"]
v_th = lookup["v_th"]
v_r = lookup["v_r"]
neurons_number = lookup["neurons_number"]
C = lookup["C"]
D = lookup["D"]

blocks = lookup["blocks"]
threads = lookup["threads"]

def fire_check(arrays):
    voltage_d = arrays["voltage"]
    wigglage_d = arrays["wigglage"]
    synapse_d = arrays["synapse"]
    input_strength_d = arrays["input_strength"]
    fire_flag_d = arrays["fire_flag"]
    lower_bound_d = arrays["lower_bound"]
    upper_bound_d = arrays["upper_bound"]
    firing_time_d = arrays["firing_time"]
    fire_check_device[threads, blocks](voltage_d, wigglage_d, synapse_d,
                                       input_strength_d,
                                       fire_flag_d, lower_bound_d,
                                       upper_bound_d, firing_time_d)

@cuda.jit()
def fire_check_device(voltage_d, wigglage_d, synapse_d, input_strength_d,
                      fire_flag_d, lower_bound_d, upper_bound_d, firing_time_d):
    """
    Checks whether the neuron can fire and records firing bounds.
    We use the firing_time_d to hold the start point for a non-interval-type
    root-finding scheme.

    Expects arrays: voltage, synapse, input_strength, fire_flag,
    lower_bound, upper_bound, firing_time
    """
    
    n = cuda.grid(1)
    if n < neurons_number:
        fire_flag_d[n] = 1
        v_0 = voltage_d[n]
        u_0 = wigglage_d[n]
        s_0 = synapse_d[n]
        I = input_strength_d[n]
        
        case = 0
        firing_time_d[n] = 0

        #Calculate some basic values
        p = 0.5*(D + 1)
        q2 = 0.25 * ( (D - 1)**2 - 4C) #"q squared"
        q = abs(q2)**0.5

        #Calculate some less-basic values
        s0_part = s_0 * (2*p - synapse_decay - 1) / ((p - synapse_decay)**2 - q2)
        I_part = I * (2*p - 1) / (p**2 - q2)

        c_coeff = v_0 - s_part - I_part

        s0_part_s = s_0 * ((p**2 + q**2) + synapse_decay*(1 - p) - p)
        s0_part_s /= ((p - synapse_decay)**2 - q2)
        I_part_s = I * (p**2 + q2 - p) / (p**2 - q2)

        s_coeff = (s0_part_s + I_part_s + (1 - p)*v_0 + u_0) / q

        target = abs(v_th - I_part)

        #Do some branching to find our limits
        if q2 >= 0:
            limit_1 = log(abs((c_coeff - s_coeff)/(2 * target))) / (p - q)
        else:
            limit_1 = log((abs(c_coeff) + abs(s_coeff))/target) / p
        limit_2 = log(abs(s0_part / target)) / synapse_decay

        limit = max(limit_1, limit_2)

        #Set the area we believe any solution will be in
        lower_bound[n] = 0
        upper_bound[n] = limit
        
        

