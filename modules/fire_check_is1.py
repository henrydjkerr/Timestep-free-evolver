"""
For a neuron represented by the synaptic-LIF model, determines whether
or not its voltage value will ever reach v_th.
If it does, it additionally calculates a lower bound and an upper bound.
The lower bound is specifically chosen so that it can be used as a
starting point for the Newton-Raphson method that converges to the
correct root.

This code is specifically to handle the singularity in the equations when
synapse_decay = 1.  
Due to the singularity in the equation when synapse_decay = 1, this version
is specifically for cases where synapse_decay != 1.

There are several different possible cases that require different
mathematical treatments.

In terms of calculating whether the neuron will fire and the associated
bounds and NR starting point, there are 10 distinct cases.
2 of these are trivial, and un-numbered.
The non-trivial cases require computations related to extreme points.
Cases 3 is a non-firing case.
The cases are described further in 'fire_check_is1 cases.odt'
"""

from numba import cuda

from modules import Control
lookup = Control.lookup

synapse_decay = lookup["synapse_decay"]
v_th = lookup["v_th"]
v_r = lookup["v_r"]
neurons_number = lookup["neurons_number"]


@cuda.jit()
def fire_check(voltage_d, synapse_d, input_strength_d,
               fire_flag_d, lower_bound_d, upper_bound_d, firing_time_d):
    """
    Checks whether the neuron can fire and records firing bounds.
    We use the firing_time_d to hold the start point for a non-interval-type
    root-finding scheme.
    """
    n = cuda.grid(1)
    if n < neurons_number:
        fire_flag_d[n] = 1
        v_0 = voltage_d[n]
        s_0 = synapse_d[n]
        I = input_strength_d[n]
        
        case = 0
        firing_time_d[n] = 0
        
        if v_0 > v_th:
            #Trivial case A: neuron is already firing
            lower_bound_d[n] = 0
            upper_bound_d[n] = 0
            firing_time_d[n] = 0
        elif (I <= v_th) and (s_0 <= 0):
            #Trivial case B: neuron will not fire
            fire_flag_d[n] = 0
        elif (I > v_th) and (s_0 == 0):
            #No extreme exists
            case = 8
        else:
            #Implicitly assuming synapse decay rate != 0
            extreme_time = 1 + (I - v_0) / s_0
            extreme_v = Control.v.get_vt(extreme_time, v_0, s_0, I)
            if s_0 < 0:
                inflect_time = extreme_time + 1
                inflect_v = Control.v.get_vt(inflect_time, v_0, s_0, I)
                
            #Start sorting through cases
            if (s_0 > 0) and (extreme_time >= 0) and (extreme_v == v_th):
                #Exceptional case where maximum is on the firing threshold
                case = 1
            elif (s_0 > 0) and (extreme_time >= 0) and (extreme_v > v_th):
                #Extreme is a maximum, takes you over threshold
                case = 2
            elif (s_0 > 0) and (extreme_v < v_th):
                #Extreme is a maximum, non-firing case
                case = 3
                fire_flag_d[n] = 0
            #Covered all cases with (s_0 >= 0),
            #so everything else has a minimum
            #I > v_th implicit from trivial cases being filtered already
            elif (I > v_th) and (inflect_time >= 0) and (inflect_v < v_th):
                #t=0 -> inflection -> cross v_th
                #Don't care when the minimum is
                case = 4
            elif (I > v_th) and (inflect_time < 0):
                #minimum -> inflection -> t=0 -> cross v_th
                case = 5
            elif (I > v_th) and (extreme_time > 0) and (inflect_v >= v_th):
                #t=0 -> minimum -> cross v_th -> inflection
                case = 6
            elif (I > v_th) and (extreme_time <= 0) and (inflect_v >= v_th):
                #minimum -> t=0 -> cross v_th -> inflection
                case = 7
            else:
                #I don't think I missed anything, but
                fire_flag_d[n] = 0

        if case in (4, 6, 7):
            a_bound = inflect_time + (v_th - inflect_v) \
                       / Control.v.get_dvdt(inflect_time, inflect_v, s_0, I)
            firing_time_d[n] = a_bound
        #Setting lower bounds first
        if case == 1:
            #Accidentally already found the firing time.  Unlikely.
            lower_bound_d[n] = extreme_time
            upper_bound_d[n] = extreme_time
            firing_time_d[n] = extreme_time
        elif case in (2, 5, 8):
            #Use the first Newton-Raphson iteration from t=0 as lower bound
            #Could consider just using t=0 and see if it's faster
            lower_bound_d[n] = (v_th-v_0)/Control.v.get_dvdt(0, v_0, s_0, I)
            firing_time_d[n] = lower_bound_d[n]
        elif case == 4:
            lower_bound_d[n] = a_bound
        elif case == 6:
            lower_bound_d[n] = extreme_time
        elif case == 7:
            lower_bound_d[n] = 0
        #Now upper bounds
        if case == 2:
            upper_bound_d[n] = extreme_time
        elif case in (6, 7):
            upper_bound_d[n] = a_bound
        elif case in (4, 5, 8):
            #Blindly moving forward with exponentially large steps
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

