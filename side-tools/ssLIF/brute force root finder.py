"""
Samples a lattice of points in the (c, t_1, ..., t_n, R) space, rootfinding a
wave solution from each one.
This is to kickstart the process of branch finding.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import e, erf, pi, cos, sin, log
from scipy import optimize
import time

beta = 6
A = 2
a = 1
B = 2
b = 2

D = 1

v_th = 1
v_r = 0
v_rest = 0.9


def update_R(new_R):
    global R
    global I
    global p
    global q2
    global abs_q

    R = new_R
    I = v_rest * (R + D) / D
    #Derived values
    p = 0.5 * (D+1)
    q = 0.5 * np.emath.sqrt((D-1)**2 -4*R)
    if type(q) != type(np.emath.sqrt(-1.0)):
        print("q is", q, "R is", R, "D is", D)
        print("q should be imaginary for this to work")
        raise TypeError
    abs_q = abs(q)
    q2 = -abs_q**2
    return

update_R(1)

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


def root_function(inputs):
    global c
    global firing_times
    c = inputs[0]
    firing_times = [0]
    for n in range(1, spikes):
        firing_times.append(inputs[n])
    #error_1 = v(0) - v_th
    #u_old = u(0)
    #error_2 = v(firing_times[1], 0, u_old) - v_th
    error = [v(0) - v_th]
    u_old = u(0)
    for n in range(1, spikes):
        error.append(v(firing_times[n], firing_times[n-1], u_old) - v_th)
        u_old = u(firing_times[n], firing_times[n-1], u_old)
    return error


#------------------------------------------------------------------------------

spikes = 3
R_count = 10
c_count = 10
t_count = 5

#outfile = open("brute_force_join.txt", "w")
outfile = open("dummy.txt", "w")

stopwatch = time.time()

t_array_base = np.linspace(0.1, 5, t_count)
t_array = np.ndarray((t_count**(spikes-1), spikes))
for n in range(len(t_array)):
    t_array[n][0] = 0.0
    for m in range(1, spikes):
        index = n
        for k in range(m-1):
            index = index // t_count
        t_array[n][m] = t_array[n][m-1] + t_array_base[index % t_count]

print(time.time() - stopwatch)

results_x = []
results_y = []
results_whole = []
results_last = []
for init_R in np.linspace(0.5, 5, R_count):
    print("R:", init_R)
    print(time.time() - stopwatch)
    update_R(init_R)
    for init_c in np.linspace(0.1, 5, c_count):
        for t_entries in t_array:
            guess = [init_c]
            for n in range(1, spikes):
                guess.append(t_entries[n])
            #guess = [init_c, init_t]
            soln = optimize.root(root_function, guess, tol=0.00005)
            if soln.success and soln.x[0] > 0 and soln.x[1] > 0:
                results_x.append(R)
                results_y.append(soln.x[0])
                results_last.append(soln.x[-2])
                results_whole.append(soln.x[:])
                line = ""
                for entry in soln.x:
                    line += str(entry) + ","
                line += str(R) + "\n"
                outfile.write(line)
outfile.close()
print(time.time() - stopwatch)

latest = max(results_last)
var_last = np.array(results_last) / latest

plt.figure()
plt.scatter(results_x, results_y, s=1, c=var_last)
plt.xlabel("$R$")
plt.ylabel("Wave speed $c$")
plt.title("Wave solutions for a {}-spike wave".format(spikes))
plt.show()




















