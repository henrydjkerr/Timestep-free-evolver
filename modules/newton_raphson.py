"""
Newton-Raphson root-finding function for finding t such that v(t) = v_th.
"""

from numba import cuda

from modules.general import Control
lookup = Control.lookup

v_th = lookup["v_th"]
error_bound = lookup["error_bound"]
neurons_number = lookup["neurons_number"]

blocks = lookup["blocks"]
threads = lookup["threads"]

def find_firing_time(arrays):
    synapse_decay = lookup["synapse_decay"]
    voltage_d = arrays["voltage"]
    synapse_d = arrays["synapse"]
    input_strength_d = arrays["input_strength"]
    fire_flag_d = arrays["fire_flag"]
    lower_bound_d = arrays["lower_bound"]
    upper_bound_d = arrays["upper_bound"]
    firing_time_d = arrays["firing_time"]  
    find_firing_time_device[blocks, threads](voltage_d, synapse_d,
                                             input_strength_d, fire_flag_d,
                                             lower_bound_d, upper_bound_d,
                                             synapse_decay, firing_time_d)

@cuda.jit()
def find_firing_time_device(voltage_d, synapse_d, input_strength_d,
                            fire_flag_d, lower_bound_d, upper_bound_d,
                            synapse_decay, firing_time_d):
    """
    Computes firing time estimates using the Newton-Raphson scheme.
    Terminates once the time estimates appear to have converged within
    some bound.
    Fails silently if this takes more than 100 iterations.
    Honestly I'm not sure what to do if the scheme somehow fails.
    It shouldn't if I've set up the mathematical conditions on the
    initial condition correctly.  If.
    """

    n = cuda.grid(1)
    if n < neurons_number:
        if fire_flag_d[n]:
            v_0 = voltage_d[n]
            if v_0 > v_th:
                #Edge case, don't want to root-solve in this case
                firing_time_d[n] = 0
                return
            s_0 = synapse_d[n]
            I = input_strength_d[n]
            t_old = firing_time_d[n]
            for count in range(100):
                v_test = Control.v.get_vt(t_old, v_0, s_0, I, synapse_decay)
                v_deriv = Control.v.get_dvdt(t_old, v_test, s_0, I,
                                             synapse_decay)
                t_new = t_old + (v_th - v_test) / v_deriv
                if abs(t_new - t_old) <= error_bound:
                    firing_time_d[n] = t_new
                    return
                else:
                    t_old = t_new

##    n = cuda.grid(1)
##    if n < neurons_number:
##        if fire_flag_d[n]:
##            t_old = firing_time_d[n]
##            for count in range(100):
##                v_test = Control.v.get_value(t_old, n, arrays)
##                v_grad = Control.v.get_gradient(t_old, v_test, n, arrays)
##                t_new = t_old + (v_th - v_test) / v_grad
##                if abs(t_new - t_old) <= error_bound:
##                    firing_time_d[n] = t_new
##                    return
##                else:
##                    t_old = t_new

#This deletes an edge-case detection... is that worth it?
#(Specifically, when v_0 > v_th)
