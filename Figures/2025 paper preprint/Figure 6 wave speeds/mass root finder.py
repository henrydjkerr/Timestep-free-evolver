"""Tool for scattershot finding a number of wave solutions to sketch out
solution branches"""

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

R = 3
D = 1

v_th = 1
v_r = 0

#2.896641324335779,1.648568279903316,2.8028942111260324

#R_range = np.linspace(0.6, 12, 24)
#c_range = np.linspace(0.1, 4.1, 9)
#t_range = np.linspace(0.1, 4.1, 9)

def update_R(new_R):
    global R
    global I
    global p
    global q2
    global abs_q

    R = new_R
    I = 0.90 * (R + D) / D
    #Derived values
    p = 0.5 * (D+1)
    q = 0.5 * np.emath.sqrt( (D-1)**2 -4*R )
    if type(q) != type(np.emath.sqrt(-1)):
        print("q is", q)
        print("q should be imaginary for this to work")
        raise TypeError
    abs_q = abs(q)
    q2 = -abs_q**2
    return

update_R(R)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#General integration function

def one(x):
    return 1

#divisions = 150
#stdevs = 3
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

#spikes = 2

def root_function(inputs):
    global c
    global firing_times
    c = inputs[0]
    firing_times = [0]
    for n in range(1, spikes):
        firing_times.append(inputs[n])
    error = [v(0) - v_th]
    u_old = u(0)
    for n in range(1, spikes):
        error.append(v(firing_times[n], firing_times[n-1], u_old) - v_th)
        u_old = u(firing_times[n], firing_times[n-1], u_old)
    return error


#------------------------------------------------------------------------------

#Smallest difference between values of R allowed
precision = 0.05

#Prepare arrays
spikes_flag = True
spikes = 0
R_array = []
ic_array = []
#Load in initial guesses from PAC
filename = "1spike_stable_postgraze"

infile = open(filename + ".txt")
for line in infile:
    part = line.strip("\n").split(",")
    if spikes_flag:
        spikes = len(part) - 1
        spikes_flag = False
    R = round(float(part[-1]) / precision) * precision
    #Bulk out the difference in case of any gaps in R due to rounding
    while len(R_array) > 0 and R - R_array[-1] > precision * 1.5:
        R_array.append(R_array[-1] + precision)
        ic_array.append(ic_array[-1])
    #Populate ic_array with new values
    if len(R_array) == 0 or R_array[-1] < R:
        R_array.append(R)
        ic_array.append([])
        ic_array[-1].append(float(part[0]))
        for n in range(1, spikes):
            ic_array[-1].append(float(part[n]))
infile.close()

#Prepare refinement of gridded estimates
outfile = open("regular_" + filename + ".txt", "w")
results_x = []
results_y = []

#Perform refinement of gridded estimates
n = 1
N = len(R_array)
for n in range(N):
    R = R_array[n]
    update_R(R)
    print(n, "/", N)
    guess = ic_array[n]
    soln = optimize.root(root_function, guess, tol=0.000001)
    if soln.success and soln.x[0] >= 0:
        results_x.append(R)
        results_y.append(soln.x[0])
        line = ""
        for item in soln.x:
            line += "{},".format(item)
        outfile.write(line + "{}\n".format(R))
outfile.close()

#Display for sanity checking
plt.figure()
plt.scatter(results_x, results_y, s=0.4)
plt.xlabel("$R$")
plt.ylabel("Wave speed $c$")
plt.title("Wave solutions by speed for given $R$")
plt.show()









