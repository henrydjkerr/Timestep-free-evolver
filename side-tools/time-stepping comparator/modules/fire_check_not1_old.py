"""
For a neuron represented by the synaptic-LIF model, determines whether
or not its voltage value will ever reach v_th.
If it does, it additionally calculates a lower bound and an upper bound.
The lower bound is specifically chosen so that it can be used as a
starting point for the Newton-Raphson method that converges to the
correct root.

Due to the singularity in the equation when synapse_decay = 1, this version
is specifically for cases where synapse_decay != 1.
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
    """Checks whether the neuron can fire and records firing bounds."""
    n = cuda.grid(1)
    if n < neurons_number:
        fire_flag_d[n] = 1
        v_0 = voltage_d[n]
        s_0 = synapse_d[n]
        I = input_strength_d[n]
        condition = 0
        
        lower_bound_type = 0
        upper_bound_type = 0
        
        if v_0 > v_th:
            #"Error"-catching case
            lower_bound_d[n] = 0
            upper_bound_d[n] = 0
            lower_bound_type = -1
            upper_bound_type = -1
        elif (I < v_th) and (s_0 <= 0):
            #Easy no-firing case
            fire_flag_d[n] = 0
        else:
            #Assuming synapse decay rate != 1
            #Also assuming synapse decay rate != 0
            if s_0 == 0:
                extreme_exists = False
                #condition = 1
            else:
                #Only know it's not either trivial case
                e_temp = (1 - synapse_decay)/synapse_decay \
                         * (I - v_0 + s_0/(1 - synapse_decay))
                e_gradient = e_temp / s_0
                e_inflect = e_temp / (s_0 * synapse_decay)
                if e_gradient <= 0:
                    extreme_exists = False
                else:
                    extreme_time = 1/(1 - synapse_decay) \
                                   * cuda.libdevice.log(e_gradient)
                    if extreme_time >= 0:
                        extreme_exists = True
                    else:
                        extreme_exists = False
            if extreme_exists:
                #Still need to check solution exists!
                extreme_v = Control.v.get_vt(extreme_time, v_0, s_0, I)
                if extreme_v == v_th:
                    lower_bound_d[n] = extreme_time
                    upper_bound_d[n] = extreme_time
                    fire_flag_d[n] = 1
                elif extreme_v > v_th:
                    lower_bound_type = 1 #Use gradient from t=0
                    upper_bound_d[n] = extreme_time
                    upper_bound_type = -1 #Already set
                    fire_flag_d[n] = 1
                elif (extreme_v < v_0) and (I > v_th):
                    lower_bound_type = 2 #Use gradient from inflection point
                    upper_bound_type = 1
                    fire_flag_d[n] = 1
                else:
                    fire_flag_d[n] = 0
            elif (I > v_th):
                #No extreme but long-term limit takes you over
                if (e_inflect > 0) and ((e_inflect-1) * (1-synapse_decay) > 0):
                    lower_bound_type = 2
                else:
                    lower_bound_type = 1
                upper_bound_type = 1
                fire_flag_d[n] = 1
            else:
                #No extreme, and long-term limit is below threshold
                fire_flag_d[n] = 0
                
            if lower_bound_type == 1:
                lower_bound_d[n] = (v_th-v_0)\
                                   / Control.v.get_dvdt(0, v_0, s_0, I)
            elif lower_bound_type == 2:
                inflect_time = 1/(1 - synapse_decay) \
                               * cuda.libdevice.log(e_inflect)
                inflect_v = Control.v.get_vt(inflect_time, v_0, s_0, I)
                lower_bound_d[n] = inflect_time + (v_th - inflect_v) \
                                   / Control.v.get_dvdt(inflect_time,
                                                        inflect_v, s_0, I)
                #But doesn't that only work if inflect_v < v_th?
                #Technically yes as a "lower bound", but it's fine as NR initial
            if upper_bound_type == 1:
                m = 0
                while True:
                    test_t = 2**m
                    if test_t > lower_bound_d[n]:                    
                        temp_v = Control.v.get_vt(test_t, v_0, s_0, I)
                        if temp_v > v_th:
                            upper_bound_d[n] = test_t
                            break
                        #Could try to also improve the lower bound if I'm lucky
                        #But I don't record the value there actively
                    m += 1
                    #No failure condition?
            #if n == 50: print(lower_bound[n], upper_bound[n])
            #Something feels ugly here but eh.
