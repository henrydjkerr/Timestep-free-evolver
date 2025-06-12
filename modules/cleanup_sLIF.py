"""
postclean - run after each main loop.  Reset things that need to be reset,
    move forward through time things that need to be moved forward, and
    propagate signals.
"""

from numba import cuda
from math import e
import numpy

from modules.general import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
leniency_threshold = lookup["leniency_threshold"]
v_r = lookup["v_r"]

threads = lookup["threads"]
blocks = lookup["blocks"]


def postclean(arrays, fastest_time):
    synapse_decay = lookup["synapse_decay"]

    voltage_d = arrays["voltage"]
    synapse_d = arrays["synapse"]
    input_strength_d = arrays["input_strength"]
    fire_flag_d = arrays["fire_flag"]
    lower_bound_d = arrays["lower_bound"]
    upper_bound_d = arrays["upper_bound"]
    firing_time_d = arrays["firing_time"]
    coordinates_d = arrays["coordinates"]
    firing_neurons_d = arrays["firing_neurons"]
    postclean_device[blocks, threads](voltage_d, synapse_d, input_strength_d,
                                      fire_flag_d, lower_bound_d, upper_bound_d,
                                      firing_time_d, coordinates_d,
                                      firing_neurons_d,
                                      synapse_decay, fastest_time)

@cuda.jit()
def postclean_device(voltage_d, synapse_d, input_strength_d,
                     fire_flag_d, lower_bound_d, upper_bound_d,
                     firing_time_d, coordinates_d, firing_neurons_d,
                     synapse_decay, fastest_time_d):
    """Updates voltages, synaptic values, processes firing signals, resets flags."""
    n = cuda.grid(1)
    if n < neurons_number:
        #Calculate updated voltage
        if fire_flag_d[n]:   #If firing
            voltage_d[n] = v_r
        else:                   #If not firing
            voltage_d[n] = Control.v.get_vt(fastest_time_d, voltage_d[n],
                                            synapse_d[n], input_strength_d[n],
                                            synapse_decay)
        #Update synapse value wrt time evolution
        synapse_d[n] *= e**(-synapse_decay * fastest_time_d)
        #Update synapse value wrt other neurons firing
        for m in firing_neurons_d:
            if m > neurons_number:
                break
            elif n != m:
                distance = Control.d.get_distance_between(coordinates_d, n, m)
                synapse_d[n] += synapse_decay * Control.c.connection_weight(distance)
        lower_bound_d[n] = 0
        upper_bound_d[n] = 0
        firing_time_d[n] = 0
