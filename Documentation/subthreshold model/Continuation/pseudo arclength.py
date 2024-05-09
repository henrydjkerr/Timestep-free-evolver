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

v_th = 1
v_r = 0

C_min = 0.001
C_max = 3.001
C_steps = 30
x_count = 6
y_count = 12

def update_C(new_C):
    global C
    global I
    global p
    global q2
    global abs_q

    #C = C_min + (C_max - C_min) * (k/C_steps)
    C = new_C
    I = 0.90 * (C + D) / D
    #Derived values
    p = 0.5*(D+1)
    q = 0.5*( (D-1)**2 -4*C )**0.5
    if type(q) != type((-1)**0.5):
        print("q is", q)
        print("q should be imaginary for this to work")
        raise TypeError
    abs_q = abs(q)
    q2 = -abs_q**2
    return

update_C(1)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#General integration function

def one(x):
    return 1

def calc_integral(t, mu, sigma, param, param2 = None,
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

def part_s(Z, z, t):
    total = 0
    sigma = z / c
    for mu in firing_times:
        total += calc_integral(t, mu, sigma, beta)
    total *= Z
    return total

#-----------------------------------------------------------------------------
#Calculations for v

def v(t, t_old = None, u_old = None):
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
           + part_v(A, a, t, t_old) - part_v(B, b, t, t_old)

def part_v(Z, z, t, t_old):
    coeff_cos = (2*p - beta - 1) / ((p - beta)**2 - q2)
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / (((p - beta)**2 - q2)*abs_q)

    sigma = z / c
    part_cos = 0
    part_sin = 0
    for mu in firing_times:
        part_cos += calc_integral(t, mu, sigma, p,
                                  func_name = "cosine", lower = t_old)
        part_sin += calc_integral(t, mu, sigma, p,
                                  func_name = "sine", lower = t_old)
    part_cos *= coeff_cos
    part_sin *= coeff_sin
    return -beta * Z * (part_cos + part_sin)

#-----------------------------------------------------------------------------
#Calculations for u

def u(t, t_old = None, u_old = None):
    coeff_cos = 1 / ((p - beta)**2 - q2)
    coeff_sin = (p - beta) / (((p - beta)**2 - q2) * abs_q)
    
    part_I = C * I / (p**2 - q2)
    part_init = 0
    if t_old != None:
        part_I *= 1 - e**(-p * (t - t_old)) * cos(abs_q * (t - t_old))
        part_I -= C * I * (p / ((p**2 - q2) * abs_q))   \
                  * e**(-p * (t - t_old)) * sin(abs_q * (t - t_old))
        part_init = e**(-p * (t - t_old))       \
                    * (u_old * cos(abs_q * (t - t_old))
                       + ((C * v_r + u_old * (1 - p)) / abs_q)
                          * sin(abs_q * (t - t_old)))
    part_s = s(t) * C * coeff_cos

    part_t_old = 0
    if t_old != None:
        part_t_old = s(t_old) * e**(-p * (t - t_old))       \
                     * C * (  coeff_cos * cos(abs_q * (t - t_old))
                            + coeff_sin * sin(abs_q * (t - t_old)))
##
##    print("u_old", u_old)
##    print("part_I", part_I)
##    print("part_init", part_init)
##    print("part_s", part_s)
##    print("part_t_old", part_t_old)
##    print("part_A", part_u(A, a, t, t_old))
##    print("part_B", part_u(B, b, t, t_old))
##        
    return part_I + part_init + part_s - part_t_old \
           + part_u(A, a, t, t_old) - part_u(B, b, t, t_old)

def part_u(Z, z, t, t_old):
    coeff_cos = 1 / ((p - beta)**2 - q2)
    coeff_sin = (p - beta) / (((p - beta)**2 - q2) * abs_q)

    sigma = z / c
    part_cos = 0
    part_sin = 0
    for mu in firing_times:
        part_cos += calc_integral(t, mu, sigma, p,
                                   func_name = "cosine", lower = t_old)
        part_sin += calc_integral(t, mu, sigma, p,
                                   func_name = "sine", lower = t_old)
    part_cos *= coeff_cos
    part_sin *= coeff_sin
    return -C * beta * Z * (part_cos + part_sin)
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

spikes = 2

#def root_function(c_try, t_1_try):
def root_function(inputs):
    global c
    global firing_times
    global C
    global base_estimate
    global tangent
    
    c = inputs[0]
    firing_times = [0] + inputs[1:-1]
    C = inputs[-1]
    update_C(C)
    
    outputs = np.zeros(spikes + 1)
    #Roots for firing times
    for n in range(spikes):
        t = firing_times[n]
        if n == 0:
            outputs[n] = v(t) - v_th
            u_old = u(t)
        else:
            outputs[n] = v(t, t_old, u_old) - v_th
            u_old = u(t, t_old, u_old)
        t_old = t
    #Roots for being correct distance away
    outputs[-1] = np.dot((firing_times - base_estimate), tangent)
    return outputs


def jacobian_like(vector, tangent, h):
    assert h > 0
    
    global c
    global firing_times
    global C
    c = vector[0] 
    firing_times_raw = [0] + vector[1:-1]
    C = vector[-1]
    update_C(C)
    
    matrix = np.ndarray((spikes+1, spikes+1))
    #Finite difference (centred) to approximate derivative
    #This is real hacky but a consequence of how it's currently set up
    
    #First: varying c
    col = 0
    c += h
    for row in range(spikes):
        t = firing_times_raw[row]
        if row == 0:
            matrix[row, col] = v(t)
            u_old = u(t)
        else:
            matrix[row, 0] = v(t, t_old, u_old)
            u_old = u(t, t_old, u_old)
        t_old = t
    c -= 2*h
    for row in range(spikes):
        t = firing_times_raw[row]
        if row == 0:
            matrix[row, col] -= v(t)
            u_old = u(t)
        else:
            matrix[row, 0] -= v(t, t_old, u_old)
            u_old = u(t, t_old, u_old)
        matrix[row, 0] /= 2*h
        t_old = t
    c = vector[0]
    #Second: varying each firing time
    for col in range(1, spikes):
        firing_times = firing_times_raw[:]
        firing_times[col] += h
        for row in range(spikes):
            t = firing_times[row]
            if row == 0:
                matrix[row, col] = v(t)
                u_old = u(t)
            else:
                matrix[row, col] = v(t, t_old, u_old)
                u_old = u(t, t_old, u_old) 
            t_old = t
        firing_times[col] -= h
        for row in range(spikes):
            t = firing_times[row]
            if row == 0:
                matrix[row, col] -= v(t)
                u_old = u(t)
            else:
                matrix[row, col] -= v(t, t_old, u_old)
                u_old = u(t, t_old, u_old)
            matrix[row, col] /= 2*h
            t_old = t
    #Third: varying C (or R, as I'll probably end up calling it later)
    col = spikes
    C += h
    update_C(C)
    for row in range(spikes):
        t = firing_times_raw[row]
        if row == 0:
            matrix[row, col] = v(t)
            u_old = u(t)
        else:
            matrix[row, col] = v(t, t_old, u_old)
            u_old = u(t, t_old, u_old) 
        t_old = t
    C -= 2*h
    update_C(C)
    for row in range(spikes):
        t = firing_times_raw[row]
        if row == 0:
            matrix[row, col] -= v(t)
            u_old = u(t)
        else:
            matrix[row, col] -= v(t, t_old, u_old)
            u_old = u(t, t_old, u_old) 
        matrix[row, col] /= 2*h
        t_old = t


    #Fourth: just fill in the tangent
    row = spikes
    for col in range(spikes+1):
        matrix[spikes, col] = tangent[col]

    return matrix

#------------------------------------------------------------------------------













