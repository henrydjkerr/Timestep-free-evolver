import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import e, erf, pi, cos, sin, log
from scipy import optimize

beta = 6
A = 2
a = 1
B = 2
b = 2

D = 1

I_raw = 0.9

v_th = 1
v_r = 0

C_min = 0.001
C_max = 3.001
C_steps = 30
x_count = 6
y_count = 12

def derived_from_C(new_C):
    C = new_C
    I = I_raw * (C + D) / D
    #Derived values
    p = 0.5*(D+1)
    q = 0.5*( (D-1)**2 -4*C )**0.5
    if type(q) != type((-1)**0.5):
        print("q is", q)
        print("q should be imaginary for this to work")
        raise TypeError
    abs_q = abs(q)
    q2 = -abs_q**2
    return I, p, abs_q, q2



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#General integration function

def one(x):
    return 1

def calc_integral(abs_q, t, mu, sigma, param, param2 = None,
                  func_name = None, lower = None, upper = None):
    #Parse the optional inputs
    if func_name == "sine":
        func = sin
    elif func_name == "cosine":
        func = cos
    else:
        func = one
    if param2 == None:
        param2 = param
    if upper == None:
        upper = t
        
    #Window of integration
    if lower == None:
        #lower = upper - 3*sigma
        lower = upper - 5
    else:
        #lower = max(upper - 3*sigma, lower)
        lower = max(upper - 5, lower)
    divisions = 100
    dT = (upper - lower) / divisions
    total = 0
    for x in range(divisions):
        T = lower + dT * (x + 0.5)
        total += func(abs_q * (t - T))  \
                 * e**(-0.5 * ((T - mu)/sigma)**2 - param*t + param2*T)
    total *= dT / (sigma * (2*pi)**0.5)
    return total

#-----------------------------------------------------------------------------
#Calculations for s

def s(t):
    return beta * (part_s(A, a, t) - part_s(B, b, t))

def part_s(inputs, Z, z, t):
    c = inputs[0]
    firing_times = [0] + inputs[1:-1]
    C = inputs[-1]
    I, p, abs_q, q2 = derived_from_C(C)
    
    total = 0
    sigma = z / c
    for mu in firing_times:
        total += calc_integral(abs_q, t, mu, sigma, beta)
    total *= Z
    return total

#-----------------------------------------------------------------------------
#Calculations for v

def v(inputs, t):
    c = inputs[0]
    firing_times = [0] + inputs[1:-1]
    C = inputs[-1]
    I, p, abs_q, q2 = derived_from_C(C)
    
    coeff_cos = (2*p - beta - 1) / ((p - beta)**2 - q2)
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / (((p - beta)**2 - q2)*abs_q)
    
    part_I = I * (2*p - 1) / (p**2 - q2)
    part_init = 0
    if t_old != None:
        part_I *= 1 - e**(-p * (t - t_old)) * cos(abs_q * (t - t_old))
        part_I -= I * (p**2 + q2 - p) / ((p**2 - q2) * abs_q)           \
                  * e**(-p * (t - t_old)) * sin(abs_q * (t - t_old))
        part_init = e**(-p * (t - t_old))       \
                    * (v_r * cos(abs_q * (t - t_old))
                       - ((v_r*(1 - p) + u_old) / abs_q)
                          * sin(abs_q * (t - t_old)))
    part_s = s(t) * coeff_cos

    coeff_cos = (2*p - beta - 1) / ((p - beta)**2 - q2)
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / (((p - beta)**2 - q2)*abs_q)
    part_t_old = 0
    if t_old != None:
        part_t_old = s(t_old) * e**(-p * (t - t_old))       \
                     * (  coeff_cos * cos(abs_q * (t - t_old))
                        + coeff_sin * sin(abs_q * (t - t_old)))

    return part_I + part_init + part_s - part_t_old \
           + part_v(inputs, A, a, t) - part_v(inputs, B, b, t)

def part_v(inputs, Z, z, t):
    c = inputs[0]
    firing_times = [0] + inputs[1:-1]
    C = inputs[-1]
    I, p, abs_q, q2 = derived_from_C(C)
    
    coeff_cos = (2*p - beta - 1) / ((p - beta)**2 - q2)
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / (((p - beta)**2 - q2)*abs_q)

    sigma = z / c
    part_cos = 0
    part_sin = 0
    for mu in firing_times:
        part_cos += calc_integral(abs_q, t, mu, sigma, p,
                                  func_name = "cosine", lower = t_old)
        part_sin += calc_integral(abs_q, t, mu, sigma, p,
                                  func_name = "sine", lower = t_old)
    part_cos *= coeff_cos
    part_sin *= coeff_sin
    return -beta * Z * (part_cos + part_sin)

    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

