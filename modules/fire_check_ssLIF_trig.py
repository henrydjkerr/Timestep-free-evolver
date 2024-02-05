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

        A = abs(Control.v.coeff_trig(v_0, s_0, u_0, I))
        B = Control.v.coeff_synapse(s_0)
        K = Control.v.coeff_const(I)
        theta = Control.v.trig_phase(v_0, s_0, u_0, I)
        
        case = 0
        firing_time_d[n] = 0
        extreme_exists = False
        
        if v_0 > v_th:
            #Trivial case: neuron is already firing
            case = 1
            lower_bound_d[n] = 0
            upper_bound_d[n] = 0
        else:
            #No trivial non-firing case this time
            #First check if an extreme point exists
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if (p == synapse_decay) or (B == 0) or (A*B >= 0):
                extreme_exists = False
            else:
                extreme_exists = True
                #e_temp = -(p * T) / (synapse_decay * B)
                extreme_time = cuda.libdevice.log(
                    -(p * T) / (B * synapse_decay)) / (p - synapse_decay)
                extreme_v = Control.v.get_vt_upper(extreme_time,
                                                   v_0, s_0, u_0, I)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            #With that knowledge, determine different cases
            if extreme_exists:
                if extreme_v < v_th and v_th < K:
                    #Minimum below threshold, long-term limit goes over
                    #Calculate inflection for subcases
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    inflect_time = cuda.libdevice.log(e_temp * p/synapse_decay) \
                                   / (p - synapse_decay)
                    inflect_v = Control.v.get_vt_upper(inflect_time,
                                                       v_0, s_0, u_0, I)
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    if inflect_time <= 0:
                        #Everything interesting happens before t=0
                        case = 2
                        lower_bound_d[n] = 0
                        #upper_bound_d[n] = ???
                    else:
                        if inflect_v < v_th:
                            #Inflection time is a lower bound
                            case = 3
                            lower_bound_d[n] = inflection_time
                            #upper_bound_d[n] = ???
                        elif extreme_time > 0:
                            #Extreme time is a lower bound
                            #Inflection time is an upper bound
                            case = 4
                            lower_bound_d[n] = extreme_time
                            upper_bound_d[n] = inflection_time
                        else:
                            #Extreme time is before 0
                            #Inflection time is still an upper bound
                            case = 5
                            lower_bound_d[n] = 0
                            upper_bound_d[n] = inflection_time
                        if A+B+K > v_th:
                            #Envelope starts above threshold, then drops below
                            #So there are two windows to search
                            lower_bound_d[n] = 0
                            #Just set the lower bound to 0 and it's fine
                elif A+B+K > v_th:
                    #Envelope starts above threshold
                    #No funny business with going below then back above
                    case = 6
                    lower_bound_d[n] = 0
                    upper_bound_d[n] = 0
                elif extreme_v > v_th and extreme_time > 0:
                    #Envelope starts below threshold
                    #Extreme point is a maximum
                    #Extreme point has not yet occurred
                    case = 7
                    lower_bound_d[n] = 0
                    upper_bound_d[n] = extreme_time
                #else: no firing
            elif A+B+K > v_th:
                #No extreme, start above threshold
                case = 6
                lower_bound_d[n] = 0
                upper_bound_d[n] = 0
            elif K > v_th:
                #No extreme, start below threshold, long term limit goes over
                case = 2
                lower_bound_d[n] = 0
                #upper_bound_d[n] = ???
            #In any other case: no firing

            #Now start processing cases
            if case > 0:
                fire_flag_d[n] = 1
            if case in (2, 3):
                #Blindly jump forward exponentially until you get an upper bound
                m = 0
                while True:
                    test_t = 2**m
                    if test_t > lower_bound_d[n]:
                        temp_v = Control.v.get_dvdt_upper(test_t,
                                                          v_0, s_0, u_0, I)
                        if temp_v > v_th:
                            upper_bound_d[n] = test_t
                            break
                    m += 1
            if case > 1:
                #Update the upper bound to account for the oscillation period
                upper_bound_d[n] = (2 * pi * ceil((abs_q * upper_bound_d[n]
                                                   + theta) / (2*pi) )
                                    - theta)/|q|
                
            


