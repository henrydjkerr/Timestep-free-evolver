"""
Newton-Raphson root-finding function for finding t such that v(t) = v_th.
"""

from math import pi, e, ceil
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
                                             firing_time_d)

@cuda.jit()
def find_firing_time_device(voltage_d, synapse_d, wigglage_d, input_strength_d,
                            fire_flag_d, lower_bound_d, upper_bound_d,
                            firing_time_d):
    """
    First, estimate the time the envelope of oscillation first surpasses
    v_th using the Newton-Raphson scheme.
    Second, step through the ensuing interval (of length one period) to find
    the actual firing time.  If no firing time found, report failed firing.
    
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
            u_0 = wigglage_d[n]
            I = input_strength_d[n]
            t_old = firing_time_d[n]
            #First, use Newton-Raphson to find start of possible firing window
            for count in range(100):
                v_test = Control.v.get_vt_upper(t_old, v_0, s_0, u_0, I)
                v_deriv = Control.v.get_dvdt_upper(t_old, v_0, s_0, u_0, I)
                t_new = t_old + (v_th - v_test) / v_deriv
                if abs(t_new - t_old) <= error_bound:
                    firing_time_d[n] = t_new
                    break
                else:
                    t_old = t_new
            #Ideally we'd stop here and take another look at whether we can
            #drop any neurons.  Would need a change to the main loop.
                    
            #Second, seek through window
            #...which needs a window
            t_start = firing_time_d[n]
            phase_offset = Control.v.trig_phase(v_0, s_0, u_0, I)
            #This is the next time that the oscillating term hits the top
            # of its envelope
            t_end = 2 * pi * ceil((Control.v.abs_q * t_start
                                   + phase_offset)/(2 * pi)) \
                     - phase_offset                   
            #We also need an upper bound on the gradient:
            p = Control.v.p
            dv_max = abs(Control.v.coeff_trig(v_0, s_0, u_0, I)) \
                     * (p**2 - Control.v.q2) * e**(-p * t_start)
            B = Control.v.coeff_synapse(s_0)
            if B > 0:
                exp_t = t_end
            else:
                exp_t = t_start
            dv_max -= B * synapse_decay * e**(-synapse_decay * exp_t)
            #Now we start stepping, using dv_max
            t_old = t_start
            v_old = Control.v.get_vt(t_old, v_0, s_0, u_0, I)
            t_new = t_old
            v_new = v_old
            grad_old = -1.0
            grad_new = -1.0
            NR_flag = False
            for count in range(100):
                if t_new > t_end:
                    #If you pass t_end, there's no root in the interval
                    fire_flag_d[n] = 0
                    return
                elif abs(v_new - v_th) < 0.01: #Placeholder; don't hardcode
                    #If you've reached close enough, finish
                    firing_time_d[n] = t_new
                    return
                elif v_new > v_th:
                    #If you've somehow overshot, finish
                    #Not sure best way to error-handle this honestly
                    firing_time_d[n] = t_new
                    return
                elif (grad_old > grad_new) and (grad_new > 0):
                    #You're now in territory where Newton-Raphson works better
                    NR_flag = True
                    break
                #Update step
                t_old = t_new
                v_old = v_new
                t_new += (v_th - v_new) / dv_max
                v_new = Control.v.get_vt(t_new, v_0, s_0, u_0, I)
                #Keep a record of the gradient
                grad_old = grad_new
                grad_new = (v_new - v_old) / (t_new - t_old)
                #(Don't get divide-by-zero since t_new = t_old <-> v_old = v_th)

            #If that didn't return, you're back to Newton-Raphson
            #Starting where the previous step left off
            if NR_flag == True:
                for count in range(100):
                    t_old = t_new
                    v_old = v_new
                    v_new = Control.v.get_vt_upper(t_old, v_0, s_0, u_0, I)
                    v_deriv = Control.v.get_dvdt_upper(t_old, v_0, s_0, u_0, I)
                    t_new = t_old + (v_th - v_new) / v_deriv
                    if abs(t_new - t_old) <= error_bound:
                        firing_time_d[n] = t_new
                        return
                    elif v_new < v_old:
                        #If your value of v decreases, there's no root here
                        fire_flag_d[n] = False
                        return
                    elif t_new > t_end:
                        #If you pass t_end, there's no root in the interval
                        fire_flag_d[n] = False
                        return
            else:
                #Not sure what to do here yet
                pass
            #Shouldn't really get here except in weird cases
            firing_time_d[n] = t_new
            return
