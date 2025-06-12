import numpy as np
import matplotlib.pyplot as plt
from math import e, erf, pi, cos, sin

dx = 0.3
neurons_number = 100
neurons_y = 100

c, t, C = 2.69362964354985,2.158687482521653,2.0

beta = 6
A = 2
a = 1
B = 2
b = 2

D = 1

I = 0.9 * (D + C) / D
print("I =", I)

firing_times = [0, t]
u_at_firing_times = firing_times[:]

v_r = 0

#Derived values
p = 0.5*(D+1)
q = 0.5*( (D-1)**2 -4*C )**0.5
if type(q) != type((-1)**0.5):
    print("q is", q)
    print("q should be imaginary for this to work")
    raise TypeError
abs_q = abs(q)
q2 = -abs_q**2

#-----------------------------------------------------------------------------
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
    
#-----------------------------------------------------------------------------

v_file = open("v-import-ssLIF-2D.txt", "w")
u_file = open("u-import-ssLIF-2D.txt", "w")
s_file = open("s-import-ssLIF-2D.txt", "w")
line_end = "\n"

#Finding values of u at firing times
for k, t in enumerate(firing_times):
    if k == 0:
        u_at_firing_times[k] = u(t)
    else:
        u_at_firing_times[k] = u(t, firing_times[k-1], u_at_firing_times[k-1])

v_line = []
u_line = []
s_line = []

for n in range(neurons_number):
    t = -(n - neurons_number//2) * dx / c
    #print(n, t)
    t_last = None
    u_last = None
    for k, firing_time in enumerate(firing_times):
        if firing_time < t:
            t_last = firing_time
            u_last = u_at_firing_times[k]
    voltage = v(t, t_last, u_last)
    wigglage = u(t, t_last, u_last)
    synapse = s(t)
    v_line.append(str(voltage))
    u_line.append(str(wigglage))
    s_line.append(str(synapse))

for y in range(neurons_y):
    for n in range(neurons_number):
        if (n == neurons_number - 1) and (y == neurons_y - 1):
            line_end = ""
        v_file.write(v_line[n] + line_end)
        u_file.write(u_line[n] + line_end)
        s_file.write(s_line[n] + line_end)
v_file.close()
u_file.close()
s_file.close()


