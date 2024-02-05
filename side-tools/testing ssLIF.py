from math import pi, e, cos, sin, cosh, sinh, atan, atanh
import random



#------------------------------------------------------------------------------

def c(x, p, abs_q, q2):
    if q2 < 0:
        return cos(x)
    else:
        return cosh(x)

def s(x, p, abs_q, q2):
    if q2 < 0:
        return sin(x)
    else:
        return sinh(x)

#Functions for deriving coefficients
def part_c(v_0, s_0, u_0, I, p, abs_q, q2):
    return v_0 - coeff_synapse(s_0, p, abs_q, q2) - coeff_const(I, p, abs_q, q2)

##def part_s(v_0, s_0, u_0, I, p, abs_q, q2):
##    value = s_0 * (p**2 + q2 - p*(beta + 1) + beta)
##    value =  -(value+ (I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0) / abs_q)
##    return value
##    #return -((s_0 * (p**2 + q2 - p*(beta + 1) + beta)) \
##    #       + (I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0) / abs_q)

def part_s(v_0, s_0, u_0, I, p, abs_q, p2):
    value = s_0 * (p**2 + q2 - p*(beta + 1) + beta) / ((p - beta)**2 - q2)
    value += I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0
    value /= -abs_q
    return value

def coeff_trig(v_0, s_0, u_0, I, p, abs_q, q2):
    #Coefficient of the "trigonometric" terms in the true voltage equation
    #Though it's the same if they're not trigonometric, actually
    c_part = part_c(v_0, s_0, u_0, I, p, abs_q, q2)
    s_part = part_s(v_0, s_0, u_0, I, p, abs_q, q2)
    if (c_part >= 0):
        sign = 1
    else:
        sign = -1
    return sign * (c_part**2 + s_part**2)**0.5

def trig_phase(v_0, s_0, u_0, I, p, abs_q, q2):
    #Calculates the phase offset of the trigonometric/hyperbolic terms
    c_part = part_c(v_0, s_0, u_0, I, p, abs_q, q2)
    s_part = part_s(v_0, s_0, u_0, I, p, abs_q, q2)
    if q2 < 0:
        #Trigonometric case
        return atan(-s_part/c_part)
    else:
        #Hyperbolic case
        return atanh(s_part/c_part)
    #TODO: what if c_part = 0?

def coeff_synapse(s_0, p, abs_q, q2):
    #Coefficient of the synapse decay term in the true voltage equation
    return s_0 * (2*p - beta - 1) / ((p - beta)**2 - q2)

def coeff_const(I, p, abs_q, q2):
    #Long term limit in the true voltage equation
    return I * (2*p - 1) / (p**2 - q2)

#------------------------------------------------------------------------------
#Functions for the actual value of v and u
def get_vt(t, v_0, s_0, u_0, I, p, abs_q, q2):
    """Calculates v(t) from initial conditions"""
    T = coeff_trig(v_0, s_0, u_0, I, p, abs_q, q2)
    theta = trig_phase(v_0, s_0, u_0, I, p, abs_q, q2)
    B = coeff_synapse(s_0, p, abs_q, q2)
    K = coeff_const(I, p, abs_q, q2)
    #return T*e**(-p * t)*c(abs_q*t + theta, p, abs_q, q2) + B*e**(-beta * t) + K
    return T*e**(-p * t)*cos(abs_q*t + theta) + B*e**(-beta * t) + K

def get_dvdt(t, v_actual, v_0, u_0, I, p, abs_q, q2):
    """Calculates the derivative of v(t) given its current value"""
    u_actual = get_ut(t, v_0, s_0, u_0, I, p, abs_q, q2)
    return I - v_actual - u_actual + s_0*e**(-beta * t) + I

def get_ut(t, v_0, s_0, u_0, I, p, abs_q, q2):
    """Calculates u(t) from initial conditions"""
    s_cluster = C * s_0 / ((p - beta)**2 - q2)
    I_cluster = C * I / (p**2 - q2)
    
    part_c = u_0 - s_cluster - I_cluster
    part_c *= e**(-p * t) * cos(abs_q * t)

    part_s = (s_cluster * (p - beta)) + (I_cluster * p) + ((p - 1)* u_0) - C*v_0
    part_s *= e**(-p * t) * sin(abs_q * t) / abs_q

    part_beta = e**(-beta * t) * s_cluster
    return part_c - part_s + part_beta + I_cluster

###For the upper bound of v
##def get_vt_upper(t, v_0, s_0, u_0, I):
##    """Calculates the upper bound on v(t) given by setting cos = 1"""
##    T = coeff_trig(v_0, s_0, u_0, I)
##    B = coeff_synapse(s_0)
##    K = coeff_const(I)
##    return T*e**(-p * t) + B*e**(-beta * t) + K
##
##def get_dvdt_upper(t, v_0, s_0, u_0, I):
##    """Calculates the derivative of the upper bound on v(t)"""
##    T = coeff_trig(v_0, s_0, u_0, I)
##    B = coeff_synapse(s_0)
##    return -(p * T*e**(-p * t) + beta * B*e**(-beta * t))

#------------------------------------------------------------------------------
#Didn't upload the latest version of fire_check_ssLIF_trig to github...

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
    A = coeff_trig(v_0, s_0, u_0, I, p, abs_q, q2)
    B = coeff_synapse(s_0, p, abs_q, q2)
    #Calculate upper bounds on derivatives
    Mvelo = abs(A * (p**2 + abs_q**2)**0.5) * e**(-p * start_time) \
            + max(-synapse_decay * B * e**(-synapse_decay * start_time),
                  -synapse_decay * B * e**(-synapse_decay * end_time))
    Maccel = abs(A * (p**4 + abs_q**4)**0.5) * e**(-p * start_time) \
             + max(synapse_decay**2 * B * e**(-synapse_decay * start_time),
                   synapse_decay**2 * B * e**(-synapse_decay * end_time))
    if Mvelo <= 0:
        return (False, 0, counter)
    #Start iterations
    t_old = start_time
    counter = 0
    for count in range(100):
        counter += 1
        v_test = get_vt(t_old, v_0, s_0, u_0, I, p, abs_q, q2)
        v_deriv = get_dvdt(t_old, v_test, s_0, u_0, I, p, abs_q, q2)
        try:
            m = min(Mvelo,
                    0.5*(v_deriv \
                         + (v_deriv**2 + 4*Maccel*(v_th - v_test))**0.5))
        except TypeError:
            #print(Mvelo, v_deriv, Maccel, v_test)
            print("Position error")
            print("Time:", t_old)
            print("Voltage:", v_test)
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

beta = 1.995
synapse_decay = beta
C = 2.850
D = 2.077

p = 0.5*(D+1)
q2 = 0.25 * ((D - 1)**2 - 4*C)
abs_q = abs(q2**0.5)

error_bound = 0.000001

v_0 = -0.18
s_0 = 2.050
u_0 = 0.766
I = 0.718

lower_bound = 0
upper_bound = 10


for x in range(1):#00):
    #beta = random.uniform(1.1, 4.1)
    #synapse_decay = beta
    #C = random.uniform(0.1, 5)
    #D = random.uniform(0.1, 5)
    
    #p = 0.5*(D+1)
    #q2 = 0.25 * ((D - 1)**2 - 4*C)
    #abs_q = abs(q2**0.5)
    if q2 < 0:
        #v_0 = random.uniform(-1, 1)
        #s_0 = random.uniform(-3, 3)
        #u_0 = random.uniform(-3, 3)
        #I = random.uniform(0.7, 1.2)

        print("beta = {}, C = {}, D = {}".format(str(beta)[:5],
                                                 str(C)[:5],
                                                 str(D)[:5]))
        print("v_0 = {}, s_0 = {}, u_0 = {}, I = {}".format(str(v_0)[:5],
                                                            str(s_0)[:5],
                                                            str(u_0)[:5],
                                                            str(I)[:5]))
        result = find_firing_time(v_0, s_0, u_0, I, lower_bound, upper_bound)
        print("Fired: {}, at time: {}, number of iterations: {}".format(
            result[0], str(result[1])[:5], result[2]), "\n")
    else:
        pass

#Something wrong with calcs somewhere
