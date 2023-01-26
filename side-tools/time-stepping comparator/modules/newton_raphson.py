"""
Newton-Raphson root-finding function for finding t such that v(t) = v_th.
"""

from numba import cuda

from modules import Control
lookup = Control.lookup

v_th = lookup["v_th"]
error_bound = lookup["error_bound"]
neurons_number = lookup["neurons_number"]


@cuda.jit()
def find_firing_time(voltage_d, synapse_d, input_strength_d,
                     fire_flag_d, lower_bound_d, upper_bound_d, firing_time_d):
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
                v_test = Control.v.get_vt(t_old, v_0, s_0, I)
                v_deriv = Control.v.get_dvdt(t_old, v_test, s_0, I)
                t_new = t_old + (v_th - v_test) / v_deriv
                if abs(t_new - t_old) <= error_bound:
                    firing_time_d[n] = t_new
                    return
                else:
                    t_old = t_new
