"""
Routine for triaging subthreshold synaptic (ssLIF) neurons in the oscillatory
case.  Weeds out neurons that definitely won't fire, and produces intervals
containing any firing time of neurons that will or might fire.

Firing is determined by v(t) = v_th.
The triage is performed by considering the upper bound of the oscillation
envelope, and treating it like the voltage function for the sLIF neurons.

More to come
"""

from math import pi, e
from numba import cuda

from modules.general import Control
lookup = Control.lookup

#------------------------------------------------------------------------------

##@cuda.jit(device = True)
##def coeff_trig(v_0, s_0, u_0, I):
##    #Coefficient of the trigonometric terms in the true voltage equation
##    part_c = v_0 - coeff_synapse(s_0) - coeff_const(I)
##    part_s = s_0 * (p**2 + q2 - p*(synapse_decay + 1) + synapse_decay)
##    part_s += I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0
##    part_s *= -(1 / abs_q)
##    if (part_c >= 0):
##        sign = 1
##    else:
##        sign = -1
##    return sign * (part_c**2 + part_s**2)**0.5
##
##@cuda.jit(device = True)
##def coeff_synapse(s_0):
##    #Coefficient of the synapse decay term in the true voltage equation
##    return s_0 * (2*p - synapse_decay - 1) / ((p - synapse_decay)**2 - q2)
##
##@cuda.jit(device = True)
##def coeff_const(I):
##    #Long term limit in the true voltage equation
##    return I * (2*p - 1) / (p*2 - q2)
##
##@cuda.jit(device = True)
##def false_v(t, T, B, K):
##    return T*e**(-p * t) + B*e**(-synapse_decay * t) + K
##
##@cuda.jit(device = True)
##def false_dvdt(t, T, B):
##    return -(p * T * e**(-p * t) + synapse_decay * B * e**(-synapse_decay * t))

#------------------------------------------------------------------------------

synapse_decay = lookup["synapse_decay"]
decay_tolerance = lookup["decay_tolerance"]
v_th = lookup["v_th"]
v_r = lookup["v_r"]
neurons_number = lookup["neurons_number"]

C = lookup["C"]
D = lookup["D"]

blocks = lookup["blocks"]
threads = lookup["threads"]


p = 0.5*(D + 1)
abs_q = abs(0.5*((D - 1)**2 - 4*C)**0.5)
q2 = -abs_q**2
period = 2*pi / abs_q

    

def fire_check(arrays):
    """
    Wrapper function so we don't need to specify which arrays are needed
    in the main program file.
    """
    voltage_d = arrays["voltage"]
    synapse_d = arrays["synapse"]
    wigglage_d = arrays["wigglage"]
    input_strength_d = arrays["input_strength"]
    fire_flag_d = arrays["fire_flag"]
    lower_bound_d = arrays["lower_bound"]
    upper_bound_d = arrays["upper_bound"]
    firing_time_d = arrays["firing_time"]
    fire_check_device[blocks, threads](voltage_d, synapse_d, wigglage_d,
                                       input_strength_d,
                                       fire_flag_d, lower_bound_d,
                                       upper_bound_d, firing_time_d)

@cuda.jit()
def fire_check_device(voltage_d, synapse_d, wigglage_d, input_strength_d,
                      fire_flag_d, lower_bound_d, upper_bound_d, firing_time_d):
    """
    Checks whether the neuron can fire and records firing bounds.
    We use the firing_time_d to hold the start point for a non-interval-type
    root-finding scheme.

    Expects arrays: voltage, synapse, wigglage, input_strength, fire_flag,
    lower_bound, upper_bound, firing_time
    """
    
    n = cuda.grid(1)
    if n < neurons_number:
        fire_flag_d[n] = 0
        v_0 = voltage_d[n]
        s_0 = synapse_d[n]
        u_0 = wigglage_d[n]
        I = input_strength_d[n]

        T = Control.v.coeff_trig(v_0, s_0, u_0, I)
        B = Control.v.coeff_synapse(s_0)
        K = Control.v.coeff_const(I)
        
        case = 0
        firing_time_d[n] = 0
        extreme_exists = False
        
        if v_0 > v_th:
            #Trivial case A: neuron is already "firing"
            case = 1
            firing_time_d[n] = 0
            lower_bound_d[n] = 0
            upper_bound_d[n] = 0
        else:
            #No trivial non-firing case this time
            #First check if an extreme point exists
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if (p == synapse_decay) or (B == 0) or (T*B >= 0):
                extreme_exists = False
            else:
                extreme_exists = True
                e_temp = -(p * T) / (synapse_decay * B)
                extreme_time = cuda.libdevice.log(e_temp) / (p - synapse_decay)
                extreme_v = Control.v.get_vt_upper(extreme_time,
                                                   v_0, s_0, u_0, I)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            #With that knowledge, determine different cases
            if extreme_exists:
                if (extreme_v >= v_th) and (extreme_time > 0):
                    #'Firing' case for maximum
                    case = 2
                    #firing_time_d[n] = 0
                    #lower_bound_d[n] = 0
                    #upper_bound_d[n] = extreme_time
                elif (extreme_v < v_th) and (v_th < K):
                    #Firing case for minimum
                    #Still need to sort into sub-cases
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    inflect_time = cuda.libdevice.log(e_temp * p/synapse_decay) \
                                   / (p - synapse_decay)
                    inflect_v = Control.v.get_vt_upper(inflect_time,
                                                       v_0, s_0, u_0, I)
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    if inflect_time <= 0:
                        case = 3
                        #firing_time_d[n] = 0
                        #lower_bound_d[n] = 0
                        #upper_bound_d[n] = ???
                    elif inflect_v >= v_th:
                        if extreme_time > 0:
                            case = 4
                            #firing_time_d[n] = inflect_time
                            #lower_bound_d[n] = extreme_time
                            #upper_bound_d[n] = inflect_time
                        else:
                            case = 5
                            #firing_time_d[n] = inflect_time
                            #lower_bound_d[n] = 0
                            #upper_bound_d[n] = inflect_time
                    else:
                        case = 6
                        #firing_time_d[n] = inflect_time
                        #lower_bound_d[n] = inflect_time
                        #upper_bound_d[n] = ???
                #Otherwise, not firing
            elif (K > v_th) and (T + B != 0):
                #Firing case for no extreme point
                #First condition: long-term limit above the firing threshold
                #Second condition: function isn't entirely flat
                case = 7
                temp = -cuda.libdevice.log((v_th - K)/(T + B))
                firing_time_d[n] = temp
                lower_bound_d[n] = temp
                upper_bound_d[n] = temp + period

        if case > 0:
            #Doing a single round of Newton-Raphson to improve bounds
            #NR init = 0 = lower bound: 2, 3
            #NR init = 0 = upper bound: none
            #NR init = t_i = lower bound: 6
            #NR init = t_i = upper bound: 4, 5
            fire_flag_d[n] = 1
            if case in (4, 5, 6):
                alt_inflect = inflect_time + (v_th - inflect_v) \
                           / Control.v.get_dvdt_upper(inflect_time,
                                                      v_0, s_0, u_0, I)
            if case in (2, 3):
                alt_zero = (v_th - v_0) / Control.v.get_dvdt_upper(0, v_0, s_0,
                                                                   u_0, I)
            #Now to fill in the values of arrays per case
            #Case 1 has already been handled as trivial (and values not shared)
            #As has case 7
            #Newton-Raphson initial conditions:
            if case in (2, 3):
                firing_time_d[n] = alt_zero
            elif case in (4, 5, 6):
                firing_time_d[n] = alt_inflect         
            #Lower bounds:
            if case in (2, 3, 5, 7):
                lower_bound_d[n] = alt_zero
            elif case == 4:
                lower_bound_d[n] = extreme_time
            elif case == 6:
                lower_bound_d[n] = alt_inflect
            #Upper bounds:
            if case == 2:
                upper_bound_d[n] = extreme_time
            elif case in (4, 5):
                upper_bound_d[n] = alt_inflect
            elif case in (3, 6):
                #Blindly jump forward exponentially until you get an upper bound
                #Might improve your lower bound for ~free
                m = 0
                while True:
                    test_t = 2**m
                    if test_t > lower_bound_d[n]:
                        temp_v = Control.v.get_dvdt_upper(test_t,
                                                          v_0, s_0, u_0, I)
                        if temp_v > v_th:
                            upper_bound_d[n] = test_t
                            if (m != 0) and (test_t / 2 > lower_bound_d[n]):
                                lower_bound_d[n] = test_t / 2
                            break
                    m += 1
            #Make allowance for the oscillation period
            upper_bound_d[n] += period


