"""
Routine for triaging subthreshold synaptic (ssLIF) neurons in the oscillatory
case.  Weeds out neurons that definitely won't fire, and produces intervals
containing any firing time of neurons that will or might fire.

Firing is determined by v(t) = v_th.
The triage is performed by considering the upper bound of the oscillation
envelope, and treating it like the voltage function for the sLIF neurons.

More to come
"""

from math import pi, e, ceil
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
#period = 2*pi / abs_q

    

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
    fire_check_device[blocks, threads](voltage_d, synapse_d, wigglage_d,
                                       input_strength_d,
                                       fire_flag_d, lower_bound_d,
                                       upper_bound_d)

@cuda.jit()
def fire_check_device(voltage_d, synapse_d, wigglage_d, input_strength_d,
                      fire_flag_d, lower_bound_d, upper_bound_d):
    """
    Checks whether the neuron can fire and records firing bounds.
    We use the firing_time_d to hold the start point for a non-interval-type
    root-finding scheme.

    Expects arrays: voltage, synapse, wigglage, input_strength, fire_flag,
    lower_bound, upper_bound
    """
    
    n = cuda.grid(1)
    if n < neurons_number:        
        v_0 = voltage_d[n]
        s_0 = synapse_d[n]
        u_0 = wigglage_d[n]
        I = input_strength_d[n]

        case = 0
        extreme_exists = False
        fire_flag_d[n] = 0
        lower_bound_d[n] = 0
        upper_bound_d[n] = 0

##        #Hacking past this
##        if v_0 > 0.6:
##            fire_flag_d[n] = 1
##            lower_bound_d[n] = 0
##            upper_bound_d[n] = 1
##    return
##
##def blah():       
        if v_0 > v_th:
            #Trivial case: neuron is already firing
            case = 1
            lower_bound_d[n] = 0
            upper_bound_d[n] = 0
        else:
            #No trivial non-firing case this time
            #First check if an extreme point exists
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            A = abs(Control.v.coeff_trig(v_0, s_0, u_0, I))
            B = Control.v.coeff_synapse(s_0)
            K = Control.v.coeff_const(I)
            if (p == synapse_decay) or (B == 0) or (A*B >= 0):
                extreme_exists = False
            else:
                extreme_exists = True
                extreme_time = cuda.libdevice.log(-(p * A) /
                                                  (B * synapse_decay)) \
                                    / (p - synapse_decay)
                extreme_v = Control.v.get_vt_upper(extreme_time,
                                                   v_0, s_0, u_0, I)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            #With that knowledge, determine different cases
            if A+B+K > v_th:
                if extreme_exists and extreme_time > 0 and extreme_v < v_th and K > v_th:
                    #Special case for where there are two crossing intervals
                    case = 3
                    #lower_bound_d[n] = 0
                    #upper_bound depends on the inflection point
                else:
                    #For any other case you just take the initial interval
                    case = 2
                    lower_bound_d[n] = 0
                    upper_bound_d[n] = 0
            elif extreme_exists:
                #Cases where A+B+K < v_th and you have an extreme point
                if extreme_time > 0 and extreme_v > v_th:
                    #There is a maximum that takes you over the threshold
                    case = 2
                    lower_bound_d[n] = 0
                    upper_bound_d[n] = extreme_time
                elif K > v_th:
                    #Since we know A+B+K < v_th, this isn't a maximum
                    #So this covers all other minima
                    case = 4
                    #Both bounds depend on inflection point
            elif K > v_th:
                #No extreme, don't start above v_th, but long term limit goes over
                case = 5
                lower_bound_d[n] = 0
                #upper_bound has to be brute-forced
            #In any other case: no firing

        #Now start processing cases
        #Case 0: no firing detected
        #Case 1: neuron is already firing (trivial)
        #Case 2: lower and upper bound easily determined, but still need period
            #   update on upper bound
        #Case 3: lower bound = 0, upper bound depends on inflection point
        #Case 4: both bounds depend on inflection point
        #Case 5: lower bound = 0, need to step through for upper bound
        if case > 0:
            fire_flag_d[n] = 1
        if case in (3, 4):
            #Need to sort out stuff about the inflection point for minima
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            inflect_time = cuda.libdevice.log(-(p**2 * A)
                                              / (synapse_decay**2 * B)) \
                           / (p - synapse_decay)
            if inflect_time > 0:
                inflect_v = Control.v.get_vt_upper(inflect_time,
                                                   v_0, s_0, u_0, I)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if inflect_time <= 0:
                lower_bound_d[n] = 0
                case = 5
            elif inflect_v < v_th:
                lower_bound_d[n] = inflect_time * (case - 3)
                case = 5
            elif extreme_time < 0:
                lower_bound_d[n] = 0
                upper_bound_d[n] = inflect_time
            else:
                lower_bound_d[n] = extreme_time * (case - 3)
                upper_bound_d[n] = inflect_time
        if case == 5:
            #Blindly jump forward exponentially until you get an upper bound
            m = 0
            while True:
                test_t = 2**m
                if test_t > lower_bound_d[n]:
                    temp_v = Control.v.get_vt_upper(test_t, v_0, s_0, u_0, I)
                    if temp_v > v_th:
                        upper_bound_d[n] = test_t
                        break
                m += 1
        if case > 1:
            #Update the upper bound to account for the oscillation period
            theta = Control.v.trig_phase(v_0, s_0, u_0, I)
            upper_bound_d[n] = (2 * pi * ceil((abs_q * upper_bound_d[n]
                                            + theta) / (2*pi) ) - theta)/abs_q
    #return
            


