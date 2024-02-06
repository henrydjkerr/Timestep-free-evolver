from math import pi, e, cos, sin, cosh, sinh, atan, atanh, log, ceil
import random



#------------------------------------------------------------------------------

def c(x):
    if q2 < 0:
        return cos(x)
    else:
        return cosh(x)

def s(x):
    if q2 < 0:
        return sin(x)
    else:
        return sinh(x)

#Functions for deriving coefficients
def part_c(v_0, s_0, u_0, I):
    return v_0 - coeff_synapse(s_0) - coeff_const(I)

##def part_s(v_0, s_0, u_0, I, p, abs_q, q2):
##    value = s_0 * (p**2 + q2 - p*(beta + 1) + beta)
##    value =  -(value+ (I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0) / abs_q)
##    return value
##    #return -((s_0 * (p**2 + q2 - p*(beta + 1) + beta)) \
##    #       + (I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0) / abs_q)

def part_s(v_0, s_0, u_0, I):
    value = s_0 * (p**2 + q2 - p*(beta + 1) + beta) / ((p - beta)**2 - q2)
    value += I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0
    value /= -abs_q
    return value

def coeff_trig(v_0, s_0, u_0, I):
    #Coefficient of the "trigonometric" terms in the true voltage equation
    #Though it's the same if they're not trigonometric, actually
    c_part = part_c(v_0, s_0, u_0, I)
    s_part = part_s(v_0, s_0, u_0, I)
    if (c_part >= 0):
        sign = 1
    else:
        sign = -1
    return sign * (c_part**2 + s_part**2)**0.5

def trig_phase(v_0, s_0, u_0, I):
    #Calculates the phase offset of the trigonometric/hyperbolic terms
    c_part = part_c(v_0, s_0, u_0, I)
    s_part = part_s(v_0, s_0, u_0, I)
    if q2 < 0:
        #Trigonometric case
        return atan(-s_part/c_part)
    else:
        #Hyperbolic case
        return atanh(s_part/c_part)
    #TODO: what if c_part = 0?

def coeff_synapse(s_0):
    #Coefficient of the synapse decay term in the true voltage equation
    return s_0 * (2*p - beta - 1) / ((p - beta)**2 - q2)

def coeff_const(I):
    #Long term limit in the true voltage equation
    return I * (2*p - 1) / (p**2 - q2)

#------------------------------------------------------------------------------
#Functions for the actual value of v and u
def get_vt(t, v_0, s_0, u_0, I):
    """Calculates v(t) from initial conditions"""
    T = coeff_trig(v_0, s_0, u_0, I)
    theta = trig_phase(v_0, s_0, u_0, I)
    B = coeff_synapse(s_0)
    K = coeff_const(I)
    #return T*e**(-p * t)*c(abs_q*t + theta, p, abs_q, q2) + B*e**(-beta * t) + K
    return T*(e**(-p * t))*cos(abs_q*t + theta) + B*e**(-beta * t) + K

def get_dvdt(t, v_actual, s_0, u_0, I):
    """Calculates the derivative of v(t) given its current value"""
    u_actual = get_ut(t, v_0, s_0, u_0, I)
    return I - v_actual - u_actual + s_0*e**(-beta * t)

def get_ut(t, v_0, s_0, u_0, I):
    """Calculates u(t) from initial conditions"""
    s_cluster = C * s_0 / ((p - beta)**2 - q2)
    I_cluster = C * I / (p**2 - q2)
    
    part_c = u_0 - s_cluster - I_cluster
    part_c *= e**(-p * t) * cos(abs_q * t)

    part_s = (s_cluster * (p - beta)) + (I_cluster * p) + ((p - 1)* u_0) - C*v_0
    part_s *= e**(-p * t) * sin(abs_q * t) / abs_q

    part_beta = e**(-beta * t) * s_cluster
    return part_c - part_s + part_beta + I_cluster

#For the upper bound of v
def get_vt_upper(t, v_0, s_0, u_0, I):
    """Calculates the upper bound on v(t) given by setting cos = 1"""
    A = coeff_trig(v_0, s_0, u_0, I)
    B = coeff_synapse(s_0)
    K = coeff_const(I)
    return A*e**(-p * t) + B*e**(-beta * t) + K

def get_dvdt_upper(t, v_0, s_0, u_0, I):
    """Calculates the derivative of the upper bound on v(t)"""
    A = coeff_trig(v_0, s_0, u_0, I)
    B = coeff_synapse(s_0)
    return -(p * A*e**(-p * t) + beta * B*e**(-beta * t))

#------------------------------------------------------------------------------
#Didn't upload the latest version of fire_check_ssLIF_trig to github...

def fire_check(v_0, s_0, u_0, I):
    """
    Checks whether the neuron can fire and records firing bounds.
    We use the firing_time_d to hold the start point for a non-interval-type
    root-finding scheme.
    """

    case = 0
    lower_bound = 0
    upper_bound = 0
    extreme_exists = False
    fire_flag = False
    
    if v_0 > v_th:
        #Trivial case: neuron is already firing
        case = 1
        lower_bound = 0
        upper_bound = 0
    else:
        #No trivial non-firing case this time
        #First check if an extreme point exists
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        A = abs(coeff_trig(v_0, s_0, u_0, I))
        B = coeff_synapse(s_0)
        K = coeff_const(I)
        if (p == beta) or (B == 0) or (A*B >= 0):
            extreme_exists = False
        else:
            extreme_exists = True
            extreme_time = log(-(p * A) / (B * beta)) / (p - beta)
            if extreme_time >= 0:
                #Conditional so we don't get OverflowErrors
                extreme_v = get_vt_upper(extreme_time, v_0, s_0, u_0, I)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #With that knowledge, determine different cases
        if A+B+K > v_th:
            if extreme_exists and extreme_time > 0 and extreme_v < v_th and K > v_th:
                #Special case for where there are two crossing intervals
                case = 3
                lower_bound = 0
                #upper_bound depends on the inflection point
            else:
                #For any other case you just take the initial interval
                case = 2
                lower_bound = 0
                upper_bound = 0
        elif extreme_exists:
            #Cases where A+B+K < v_th and you have an extreme point
            if extreme_time > 0 and extreme_v > v_th:
                #There is a maximum that takes you over the threshold
                lower_bound = 0
                upper_bound = extreme_time
            elif K > v_th:
                #Since we know A+B+K < v_th, this isn't a maximum
                #So this covers all other minima
                case = 4
                #Both bounds depend on inflection point
        elif K > v_th:
            #No extreme, don't start above v_th, but long term limit goes over
            case = 5
            lower_bound = 0
            #upper_bound has to be brute-forced
        #In any other case: no firing
    #Now start processing cases
    if case > 0:
        fire_flag = True
    if case in (3, 4):
        #Need to sort out stuff about the inflection point for minima
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        inflect_time = log(-(p**2 * A) / (beta**2 * B)) / (p - beta)
        if inflect_time > 0:
            inflect_v = get_vt_upper(inflect_time, v_0, s_0, u_0, I)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if inflect_time <= 0:
            lower_bound = 0
            case = 5
        elif inflect_v < v_th:
            lower_bound = inflect_time * (case - 3)
            case = 5
        elif extreme_time < 0:
            lower_bound = 0
            upper_bound = inflect_time
        else:
            lower_bound = extreme_time * (case - 3)
            upper_bound = inflect_time
    if case == 5:
        #Blindly jump forward exponentially until you get an upper bound
        m = 0
        while True:
            test_t = 2**m
            if test_t > lower_bound:
                temp_v = get_vt_upper(test_t, v_0, s_0, u_0, I)
                if temp_v > v_th:
                    upper_bound = test_t
                    break
            m += 1
    if case > 1:
        #Update the upper bound to account for the oscillation period
        theta = trig_phase(v_0, s_0, u_0, I)
        upper_bound = (2 * pi * ceil((abs_q * upper_bound + theta)
                                     / (2*pi)) - theta)/abs_q
    return(fire_flag, lower_bound, upper_bound)



#------------------------------------------------------------------------------

def find_firing_time(v_0, s_0, u_0, I, start_time, end_time):
    """
    Seeks the firing time v(t) = v_th, using the 'maximum acceleration method',
    my modification of the Newton-Raphson algorithm.
    Steps through an interval using NR-style steps, restricted to make sure
    they never step over a root.
    If reaches a value over upper_bound_d[n], reports no firing time found.
    """
    #Load variables
    if v_0 > v_th:
        #Edge case, don't want to root-solve in this case
        return (True, 0, 0)
    #Calculate derived constants
    A = coeff_trig(v_0, s_0, u_0, I)
    B = coeff_synapse(s_0)
    #Calculate upper bounds on derivatives
    Mvelo = abs(A * (p**2 + abs_q**2)**0.5) * e**(-p * start_time) \
            + max(-synapse_decay * B * e**(-synapse_decay * start_time),
                  -synapse_decay * B * e**(-synapse_decay * end_time))
    Maccel = max(abs(A * (p**4 + abs_q**4)**0.5) * e**(-p * start_time) \
                 + max(synapse_decay**2 * B * e**(-synapse_decay * start_time),
                       synapse_decay**2 * B * e**(-synapse_decay * end_time)),
                 0)
    if Mvelo <= 0:
        return (False, 0, 0)
    #Start iterations
    t_old = start_time
    counter = 0
    for count in range(100):
        counter += 1
        v_test = get_vt(t_old, v_0, s_0, u_0, I)
        v_deriv = get_dvdt(t_old, v_test, s_0, u_0, I)
        try:
            m = min(Mvelo,
                    0.5*(v_deriv \
                         + (v_deriv**2 + 4*Maccel*(v_th - v_test))**0.5))
            #print(t_old, v_test, m)
            #print(t_old + (1 - v_test)/m)
        except TypeError:
            #print(Mvelo, v_deriv, Maccel, v_test)
            print("Position error")
            print("Time:", t_old)
            print("Voltage:", v_test)
            print("M:", Mvelo)
            print("M':", Maccel)
            print("v_deriv:", v_deriv)
            print(counter)
            raise
        if m <= 0:
            return (False, t_old, counter)
        t_new = t_old + (v_th - v_test) / m
        if abs(t_new - t_old) <= error_bound:
            return (True, t_new, counter)
        elif t_new > end_time:
            return (False, t_new, counter)
        else:
            t_old = t_new
    #Currently silently failing if it takes too many iterations
    return (False, t_new, counter)


#------------------------------------------------------------------------------

v_th = 1
v_r = 0

##beta = 1.995
##synapse_decay = beta
##C = 2.850
##D = 2.077
##v_0 = -0.18
##s_0 = 2.050
##u_0 = 0.766
##I = 0.718

v_0 = -0.45
s_0 = 2.201
u_0 = -0.83
I = 0.990
beta = 1.560
C = 2.183
D = 1.529

beta = 2.554
C = 2.729
D = 3.864
v_0 = 0.601
s_0 = 1.258
u_0 = -1.37
I = 0.737

v_0 = -0.7366913762715817
s_0 = 1.43877533028168
u_0 = 0.49549990146727874
I = 1.0666990972531414
beta = 3.454704939258241
C = 1.3194081275570249
D = 1.8612391512003994

v_0 = -0.12358751876809682
s_0 = -2.6534389873966973
u_0 = -1.640959313776784
I = 1.1929969370369464
C = 0.30041492031562556
D = 1.9667230932557858
beta = 1.21306163732562

synapse_decay = beta
p = 0.5*(D+1)
q2 = 0.25 * ((D - 1)**2 - 4*C)
abs_q = abs(q2**0.5)

error_bound = 0.000001

lower_bound = 0
upper_bound = 10

test_mode = False
if test_mode:
    loop_count = 1
else:
    loop_count = 10000

true_count = 0
true_iters = 0
false_count = 0
false_iters = 0

for x in range(loop_count):
    if not test_mode:
        beta = random.uniform(1.1, 4.1)
        synapse_decay = beta
        C = random.uniform(0.1, 5)
        D = random.uniform(0.1, 5)
    p = 0.5*(D+1)
    q2 = 0.25 * ((D - 1)**2 - 4*C)
    abs_q = abs(q2**0.5)
    if q2 < 0:
        if not test_mode:
            v_0 = random.uniform(-1, 1)
            s_0 = random.uniform(-3, 3)
            u_0 = random.uniform(-3, 3)
            I = random.uniform(0.7, 1.2)
        #print("beta = {}, C = {}, D = {}".format(str(beta)[:5],
        #                                         str(C)[:5],
        #                                         str(D)[:5]))
        #print("v_0 = {}, s_0 = {}, u_0 = {}, I = {}".format(str(v_0)[:5],
        #                                                    str(s_0)[:5],
        #                                                    str(u_0)[:5],
        #                                                    str(I)[:5]))

        prelim = fire_check(v_0, s_0, u_0, I)
        if prelim[0] == True:
            lower_bound = prelim[1]
            upper_bound = prelim[2]
            result = find_firing_time(v_0, s_0, u_0, I, lower_bound, upper_bound)
        #print("Fired: {}, at time: {}, number of iterations: {}".format(
        #    result[0], str(result[1])[:5], result[2]), "\n")
            if result[0] == True:
                true_count += 1
                true_iters += result[2]
            else:
                false_count +=1
                false_iters += result[2]
        else:
            #false_count += 1
            pass
    else:
        pass

if true_count > 0:
    print("Average iterations to root:", true_iters/true_count)
if false_count > 0:
    print("Average iterations to no root:", false_iters/false_count)
