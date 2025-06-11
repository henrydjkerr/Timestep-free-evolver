"""
Newton-Raphson-style root-finding function for finding t such that v(t) = v_th.
"""

from math import pi, e
from numba import cuda

from modules.general import Control
lookup = Control.lookup

v_th = lookup["v_th"]
error_bound = lookup["error_bound"]
neurons_number = lookup["neurons_number"]
synapse_decay = lookup["synapse_decay"]

blocks = lookup["blocks"]
threads = lookup["threads"]

def find_firing_time(arrays):
    R = lookup["R"]
    D = lookup["D"]
    beta = lookup["synapse_decay"]
    
    voltage_d = arrays["voltage"]
    synapse_d = arrays["synapse"]
    wigglage_d = arrays["wigglage"]
    input_strength_d = arrays["input_strength"]
    fire_flag_d = arrays["fire_flag"]
    lower_bound_d = arrays["lower_bound"]
    upper_bound_d = arrays["upper_bound"]
    firing_time_d = arrays["firing_time"]
    find_firing_time_device[blocks, threads](voltage_d, synapse_d, wigglage_d,
                                             input_strength_d, fire_flag_d,
                                             lower_bound_d, upper_bound_d,
                                             beta, R, D, firing_time_d)

@cuda.jit()
def find_firing_time_device(voltage_d, synapse_d, wigglage_d, input_strength_d,
                            fire_flag_d, lower_bound_d, upper_bound_d,
                            beta, R, D, firing_time_d):
    """
    Seeks the firing time v(t) = v_th, using the 'maximum acceleration method',
    my modification of the Newton-Raphson algorithm.
    Steps through an interval using NR-style steps, restricted to make sure
    they never step over a root.
    If reaches a value over upper_bound_d[n], reports no firing time found.
    """
    n = cuda.grid(1)
    if n < neurons_number:
        p = 0.5*(D+1)
        q2 = 0.25 * ((D - 1)**2 - 4*R)
        if q2 >= 0:
            abs_q = q2**0.5
        else:
            abs_q = (-q2)**0.5
        
        if fire_flag_d[n]:
            #Load variables
            v_0 = voltage_d[n]
            if v_0 >= v_th:
                #Edge case, don't want to root-solve in this case
                firing_time_d[n] = 0
                return
            s_0 = synapse_d[n]
            u_0 = wigglage_d[n]
            I = input_strength_d[n]
            start_time = lower_bound_d[n]
            end_time = upper_bound_d[n]
            #Calculate derived constants
            A = Control.v.coeff_trig(v_0, s_0, u_0, I, beta, R, D)
            B = Control.v.coeff_synapse(s_0, beta, R, D)
            #Start iterations
            t_old = start_time
            for count in range(100):
                #Calculate upper bounds on derivatives
                #You need fewer iterations if you do it inside the loop
                Mvelo = abs(A * (p**2 + abs_q**2)**0.5) * e**(-p * start_time) \
                        + max(-beta * B * e**(-beta * start_time),
                              -beta * B * e**(-beta * end_time))
                Maccel = max(abs(A * (p**4 + abs_q**4)**0.5) * e**(-p * start_time) \
                             + max(beta**2 * B * e**(-beta * start_time),
                                   beta**2 * B * e**(-beta * end_time)),
                             0)
                if Mvelo <= 0:
                    #Gradient is negative from here on, so you'll never cross
                    fire_flag_d[n] = 0
                    return
                #Perform modified Newton-Raphson method
                v_test = Control.v.get_vt(t_old, v_0, s_0, u_0, I, beta, R, D)
                v_deriv = Control.v.get_dvdt(t_old, v_test, v_0, s_0, u_0, I,
                                             beta, R, D)
                m = min(Mvelo,
                        0.5*(v_deriv \
                             + (v_deriv**2 + 4*Maccel*(v_th - v_test))**0.5))
                if m <= 0:
                    fire_flag_d[n] = 0
                    break
                t_new = t_old + (v_th - v_test) / m
                if abs(t_new - t_old) <= error_bound:
                    firing_time_d[n] = t_new
                    break
                elif t_new > end_time:
                    fire_flag_d[n] = 0
                    break
                else:
                    t_old = t_new
            #Currently silently failing if it takes too many iterations
    if count == 99:
        fire_flag_d[n] = 0
    return
