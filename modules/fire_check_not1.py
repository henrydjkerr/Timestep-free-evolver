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
bounds and NR starting point, there are 12 distinct cases.
2 of these are trivial, and un-numbered.
The non-trivial cases require computations related to extreme points.
Cases 9 and 10 are non-firing cases.
The cases are described further in 'fire_check_not1 cases.odt'
"""

from numba import cuda

from modules.general import Control
lookup = Control.lookup

synapse_decay = lookup["synapse_decay"]
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
    voltage_d = arrays["voltage"]
    synapse_d = arrays["synapse"]
    input_strength_d = arrays["input_strength"]
    fire_flag_d = arrays["fire_flag"]
    lower_bound_d = arrays["lower_bound"]
    upper_bound_d = arrays["upper_bound"]
    firing_time_d = arrays["firing_time"]
    fire_check_device[blocks, threads](voltage_d, synapse_d, input_strength_d,
                                       fire_flag_d, lower_bound_d,
                                       upper_bound_d, firing_time_d)

@cuda.jit()
def fire_check_device(voltage_d, synapse_d, input_strength_d, fire_flag_d,
                      lower_bound_d, upper_bound_d, firing_time_d):
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
        s_0 = synapse_d[n]
        I = input_strength_d[n]
        
        case = 0
        firing_time_d[n] = 0
        extreme_exists = False
        e_t_positive = False
        
        if v_0 > v_th:
            #Trivial case A: neuron is already firing
            lower_bound_d[n] = 0
            upper_bound_d[n] = 0
            firing_time_d[n] = 0
        elif (I <= v_th) and (s_0 <= 0):
            #Trivial case B: neuron will not fire
            fire_flag_d[n] = 0
        else:
            #Implicitly assuming synapse decay rate != 0
            #Now to look for the extreme point
            if s_0 == 0:
                extreme_exists = False
            else:
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
                    extreme_v = Control.v.get_vt(extreme_time, v_0, s_0, I)

            if extreme_exists:
                #Still need to check if we cross v_th
                if (extreme_time >= 0) and (extreme_v == v_th):
                    #Exceptional case
                    case = 1
                elif (extreme_time >= 0) and (extreme_v > v_th):
                    #Extreme is a maximum, and by trivial cases has t > 0
                    case = 2
                elif (extreme_v < v_th):
                    #Know that extreme_v < v_th, so it's a minimum
                    inflect_time = 1/(1 - synapse_decay) \
                                   * cuda.libdevice.log(e_inflect)
                    inflect_v = Control.v.get_vt(inflect_time, v_0, s_0, I)
                    if (I <= v_th):
                        #Can't fire with a minimum + low long-term limit
                        case = 9
                        fire_flag_d[n] = 0
                    elif (extreme_time >= 0) and (inflect_v > v_th):
                        #t=0 -> minimum -> cross v_th -> inflection
                        case = 3
                    elif (extreme_time >= 0) and (inflect_v <= v_th):
                        #t=0 -> minimum -> inflection -> cross v_th
                        case = 4
                    elif (extreme_time < 0) and (inflect_v > v_th):
                        #minimum -> t=0 -> cross v_th -> inflection
                        case = 5
                    elif (extreme_time < 0) and (inflect_time >= 0):
                        #minimum -> t=0 -> inflection -> cross v_th
                        case = 6
                    else:
                        #minimum -> inflection -> t=0 -> cross v_th
                        case = 7
                else:
                    #Maximum was before t=0, so it's all downhill
                    case = 10
                    fire_flag_d[n] = 0
                    
            else:
                if (I > v_th):
                    #No extreme, long-term limit takes you over
                    case = 8
                else:
                    #Long term limit doesn't take you over
                    case = 10
                    fire_flag_d[n] = 0

            if case in (3, 4, 5, 6):
                #Precomputing this because it's useful
                a_bound = inflect_time + (v_th - inflect_v) \
                           / Control.v.get_dvdt(inflect_time, inflect_v, s_0, I)
                firing_time_d[n] = a_bound
            #Setting lower bounds first
            if case == 1:
                #Accidentally already found the firing time.  Unlikely.
                lower_bound_d[n] = extreme_time
                upper_bound_d[n] = extreme_time
                firing_time_d[n] = extreme_time
            elif case in (2, 7, 8):
                #Use the first Newton-Raphson iteration from t=0 as lower bound
                #Could consider just using t=0 and see if it's faster
                lower_bound_d[n] = (v_th-v_0)/Control.v.get_dvdt(0, v_0, s_0, I)
                firing_time_d[n] = lower_bound_d[n]
            elif case == 3:
                lower_bound_d[n] = extreme_time
            elif case in (4, 6):
                lower_bound_d[n] = a_bound
            elif case == 5:
                lower_bound_d[n] = 0
            #Now upper bounds
            if case == 2:
                upper_bound_d[n] = extreme_time
            elif case in (3, 5):
                upper_bound_d[n] = a_bound
            elif case in (4, 6, 7, 8):
                #Blindly moving forward with exponentially large steps
                #Copy code here
                m = 0
                while True:
                    test_t = 2**m
                    if test_t > lower_bound_d[n]:                    
                        temp_v = Control.v.get_vt(test_t, v_0, s_0, I)
                        if temp_v > v_th:
                            upper_bound_d[n] = test_t
                            if (m != 0) and (test_t / 2 > lower_bound_d[n]):
                                lower_bound_d[n] = test_t / 2
                            break
                    m += 1
            #All Newton-Raphson initial conditions were done along the way

