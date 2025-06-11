"""
For a neuron represented by the synaptic-LIF model, determines whether
or not its voltage value will ever reach v_th.
If it does, it additionally calculates a lower bound and an upper bound.
Additionally, the firing_time array is loaded with a convergent
starting point for the Newton-Raphson numerical root-finding method.

Due to the singularity in the equation when synapse_decay = 1, this version
is specifically for cases where synapse_decay != 1.

There are several different possible cases that require different
mathematical treatments.

In terms of calculating whether the neuron will fire and the associated
bounds and NR starting point, there are 7 distinct firing cases.
"""

from numba import cuda

from modules.general import Control
lookup = Control.lookup

decay_tolerance = lookup["decay_tolerance"]
v_th = lookup["v_th"]
v_r = lookup["v_r"]
neurons_number = lookup["neurons_number"]

blocks = lookup["blocks"]
threads = lookup["threads"]

def fire_check(arrays):
    """
    Wrapper function so we don't need to specify which arrays are needed
    in the main program file.
    """
    synapse_decay = lookup["synapse_decay"]
    
    voltage_d = arrays["voltage"]
    synapse_d = arrays["synapse"]
    input_strength_d = arrays["input_strength"]
    fire_flag_d = arrays["fire_flag"]
    lower_bound_d = arrays["lower_bound"]
    upper_bound_d = arrays["upper_bound"]
    firing_time_d = arrays["firing_time"]
    fire_check_device[blocks, threads](voltage_d, synapse_d, input_strength_d,
                                       fire_flag_d, lower_bound_d,
                                       upper_bound_d,
                                       synapse_decay, firing_time_d)

@cuda.jit()
def fire_check_device(voltage_d, synapse_d, input_strength_d, fire_flag_d,
                      lower_bound_d, upper_bound_d,
                      synapse_decay, firing_time_d):
    """
    Checks whether the neuron can fire and records firing bounds.
    We use the firing_time_d to hold the start point for a non-interval-type
    root-finding scheme.

    Expects arrays: voltage, synapse, input_strength, fire_flag,
    lower_bound, upper_bound, firing_time
    """
    
    n = cuda.grid(1)
    if n < neurons_number:
        fire_flag_d[n] = 0
        v_0 = voltage_d[n]
        s_0 = synapse_d[n]
        I = input_strength_d[n]
        
        case = 0
        firing_time_d[n] = 0
        extreme_exists = False
        
        if v_0 > v_th:
            #Trivial case A: neuron is already firing
            case = 1
            firing_time_d[n] = 0
            lower_bound_d[n] = 0
            upper_bound_d[n] = 0
        elif (I <= v_th) and (s_0 <= 0):
            #Trivial case B: neuron will not fire
            pass
        else:
            #Implicitly assuming synapse decay rate != 0
            #Now to look for the extreme point
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            #This is the main bit where synapse_decay = 1 matters
            if s_0 == 0:
                #This bit is the same for both equations
                extreme_exists = False
            else:
                if abs(synapse_decay - 1) > decay_tolerance:
                    #Case synapse_decay != 1
                    e_temp = (1 - synapse_decay)/synapse_decay \
                             * (I - v_0 + s_0/(1 - synapse_decay))
                    e_gradient = e_temp / s_0
                    e_inflect = e_temp / (s_0 * synapse_decay)
                    #e_inflect relates to the zero of the second derivative
                    if e_gradient <= 0:
                        #Can't have a real value x that makes e**x negative
                        extreme_exists = False
                    else:
                        extreme_exists = True
                        #Find the extreme time
                        extreme_time = 1/(1 - synapse_decay) \
                                       * cuda.libdevice.log(e_gradient)
                        if extreme_time >= 0:
                            #Don't want to do e**(+ve) to avoid OverflowError
                            extreme_v = Control.v.get_vt(extreme_time,
                                                         v_0, s_0, I,
                                                         synapse_decay)
                else:
                    #Case synapse_decay ~= 1
                    #Already ruled out lack of extreme point by s_0 != 0
                    extreme_exists = True
                    extreme_time = 1 - (v_0 - I)/s_0
                    if extreme_time >= 0:
                        #Don't want to do e**(+ve) to avoid OverflowError
                        extreme_v = Control.v.get_vt(extreme_time,
                                                     v_0, s_0, I, synapse_decay)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            #With that knowledge, determine different cases
            if extreme_exists:
                if extreme_time > 0 and extreme_v >= v_th:
                    #Firing case for maximum
                    case = 2
                    #firing_time_d[n] = 0
                    #lower_bound_d[n] = 0
                    #upper_bound_d[n] = extreme_time
                elif I > v_th:
                    #Firing case for minimum
                    #Still need to sort into sub-cases
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    #Another bit where we split based on synapse_decay = 1
                    if abs(synapse_decay - 1) > decay_tolerance:
                        inflect_time = 1/(1 - synapse_decay) \
                                       * cuda.libdevice.log(e_inflect)
                    else:
                        inflect_time = extreme_time + 1
                    if inflect_time >= 0:
                        inflect_v = Control.v.get_vt(inflect_time,
                                                     v_0, s_0, I, synapse_decay)
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
            elif I > v_th:
                #Firing case for no extreme point
                case = 7
                #firing_time_d[n] = 0
                #lower_bound_d[n] = 0
                #upper_bound_d[n] = ???

        if case > 0:
            #Doing a single round of Newton-Raphson to improve bounds
            #NR init = 0 = lower bound: 2, 3, 7
            #NR init = 0 = upper bound: none
            #NR init = t_i = lower bound: 6
            #NR init = t_i = upper bound: 4, 5
            fire_flag_d[n] = 1
            if case in (4, 5, 6):
                alt_inflect = inflect_time + (v_th - inflect_v) \
                           / Control.v.get_dvdt(inflect_time, inflect_v, s_0, I,
                                                synapse_decay)
            if case in (2, 3, 7):
                alt_zero = (v_th - v_0) / Control.v.get_dvdt(0, v_0, s_0, I,
                                                             synapse_decay)
            #Now to fill in the values of arrays per case
            #Case 1 has already been handled as trivial (and values not shared)
            #Newton-Raphson initial conditions:
            if case in (2, 3, 7):
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
            elif case in (3, 6, 7):
                #Blindly jump forward exponentially until you get an upper bound
                #Might improve your lower bound for ~free
                m = 0
                while True:
                    test_t = 2**m
                    if test_t > lower_bound_d[n]:                    
                        temp_v = Control.v.get_vt(test_t, v_0, s_0, I,
                                                  synapse_decay)
                        if temp_v > v_th:
                            upper_bound_d[n] = test_t
                            if (m != 0) and (test_t / 2 > lower_bound_d[n]):
                                lower_bound_d[n] = test_t / 2
                            break
                    m += 1


