"""This one is adapted for graze tracking.
This means tracking a third time, though it doesn't register as firing.
You also want to make sure dv/dt = I - v - u + s = 0 at that time.

For now at least, assumes the graze happens after all firing events.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import e, erf, pi, cos, sin, log
from scipy import optimize
from scipy import linalg
from scipy import differentiate
import time

beta = 6
A = 2
a = 1
B = 2
b = 2

R = 3
D = 1
I_raw = 0.9

v_th = 1
v_r = 0

def update_R(new_R, new_D):
    global R
    global D
    global I
    global p
    global q2
    global abs_q

    R = new_R
    D = new_D
    I = I_raw * (R + D) / D
    #Derived values
    p = 0.5*(D+1)
    interior = (D-1)**2 -4*R
    if interior <= 0:
        abs_q = 0.5 * (-interior)**0.5
        q2 = -abs_q**2
    else:
        print("q is 0.5 *", interior, "** 0.5")
        print("q should be imaginary for this to work")
        raise TypeError
    q2 = -abs_q**2
    return

update_R(R, D)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#General integration function

def one(x):
    return 1

divisions = 800
stdevs = 4
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
    lower_temp = min(upper, mu - sigma**2 * param2) - stdevs * sigma
    if lower == None:    
        lower = upper - stdevs * sigma
    lower = max(lower_temp, lower)
    
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
##    if np.isnan(beta * (part_s(A, a, t) - part_s(B, b, t))):
##        print("NaN tripped")
##        print(t, A, a, B, b)
##        print(R, D, beta)
##        print(firing_times)
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
    
    part_I = R * I / (p**2 - q2)
    part_init = 0
    if t_old != None:
        part_I *= 1 - e**(-p * (t - t_old)) * cos(abs_q * (t - t_old))
        part_I -= R * I * (p / ((p**2 - q2) * abs_q))   \
                  * e**(-p * (t - t_old)) * sin(abs_q * (t - t_old))
        part_init = e**(-p * (t - t_old))       \
                    * (u_old * cos(abs_q * (t - t_old))
                       + ((R * v_r + u_old * (1 - p)) / abs_q)
                          * sin(abs_q * (t - t_old)))
    part_s = s(t) * R * coeff_cos

    part_t_old = 0
    if t_old != None:
        part_t_old = s(t_old) * e**(-p * (t - t_old))       \
                     * R * (  coeff_cos * cos(abs_q * (t - t_old))
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
    return -R * beta * Z * (part_cos + part_sin)
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#def root_function(c_try, t_1_try):
def root_function(inputs):
    global c
    global firing_times
    global R
    #global D
    #global beta
    #global base_estimate
    #global tangent
    
    c = inputs[0]
    graze_t = inputs[-2]
    R = inputs[-1]
    #beta = inputs[-1]
    update_R(R, D)
    
    firing_times = np.zeros(spikes)
    graze_n = 0
    for n in range(1, spikes):
##        print(n, firing_times[n], inputs[n])
        firing_times[n] = inputs[n]
        if firing_times[n] < graze_t:
            graze_n = n
            #graze_n becomes number of the spike just before the graze
    
    
    outputs = np.zeros(len(inputs))
    #Roots for firing times
    for n in range(spikes):
        t = firing_times[n]
        #print(n, t)
        if n == 0:
            outputs[n] = v(t) - v_th
            u_old = u(t)
        else:
            outputs[n] = v(t, t_old, u_old) - v_th
            u_old = u(t, t_old, u_old)
        t_old = t
        #Dealing with graze bifurcation
        if n == graze_n:
            #print("graze", n, t_old, graze_t)
            temp_v = v(graze_t, t_old, u_old)
            outputs[-2] = temp_v - v_th
            outputs[-1] = I - temp_v - u(graze_t, t_old, u_old) + s(graze_t)
    #print(outputs)
    return outputs

def root_for_jacobian(inputs):
    #It's how the scipy docs say to do it
    return np.apply_along_axis(root_function, axis=0, arr=inputs)

##def diy_jacobian(inputs):
##    #Because sometimes the official one seems to do weird stuff
##    dim = len(inputs)
##    matrix = np.ndarray((dim, dim))
##    for col in 
    


#------------------------------------------------------------------------------

graze_R = []
start_index = 70

max_spikes = 16
for spikes in range(1, max_spikes + 1):
    print(spikes)
    #Load in initial estimate for graze position
    infile = open(str(spikes) + "spike.txt")
    lines = infile.readlines()
    line = lines[start_index]
    values = line.strip("\n").split(",")
    guess = [float(value) for value in values]
    R = guess.pop()
    update_R(R, D)
    infile.close()
    #Estimate of graze position
    if spikes == 1:
        guess.append(0.8)
    elif spikes == 2:
        guess.append(2 * guess[-1])
    else:
        guess.append(guess[-1] + (guess[-1] - guess[-2]))
    guess.append(R) #Yes, we pop R then put it back on, it keeps the order nice
    print("guess =", guess)
    #Perform NR
    soln = optimize.root(root_function, guess, tol=0.00000005)
    if soln.success:
        print("solution =", soln.x)
        this_R = soln.x[-1]
        graze_R.append(soln.x[:])
        start_index = int(this_R / 0.05) + 1 #Natural param. continuation in R
    else:
        print("Something went wrong")
        graze_R.append(0)

#Write results to file
outfile = open("grazes.txt", mode="w")
for graze in graze_R:
    line = ""
    for value in graze:
        line += str(value) + ","
    outfile.write(line[:-1] + "\n")
outfile.close()




