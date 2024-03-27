import numpy as np
import matplotlib.pyplot as plt
from math import e, erf, pi, cos, sin

##beta = 4
##A = 2
##a = 1
##B = 2
##b = 3
##
##I = 3.4
##
##C = 6
##D = 2
##
##c = 0.9
##firing_times = [0, 0.53, 0.9, 1.2, 1.45, 1.7, 2.0, 2.3, 2.7, 3.1, 3.6]
##u_at_firing_times = firing_times[:]

##beta = 6
##A = 2
##a = 1
##B = 2
##b = 2
##
##I = 1.74
##
##C = 2
##D = 2
##
##c = 0.3
##firing_times = [0, 1.12, 2.05, 2.9, 3.75, 4.65, 5.7]


##beta = 6
##A = 2
##a = 1
##B = 2
##b = 2.8
##
##C = 2.0001
##D = 1
##
##I = 0.90 * (C + D) / D
##
##c = 3.55
##firing_times = [0]


##beta = 4
##A = 2
##a = 1
##B = 2
##b = 2
##
##C = 2.0001
##D = 1
##
##I = 0.90 * (C + D) / D
##
##c = 0.15
##firing_times = [0.0, 1.0, 1.8, 2.5, 3.2, 3.9,
##                4.5, 5.1, 5.7, 6.3, 6.9,
##                7.6, 8.3, 9.0, 9.7, 10.3, 10.9,
##                11.5, 12.1, 12.8, 13.5, 14.3, 15.2,
##                16.1, 17.0, 18.0, 19, 20, 21.1, 22.3]

##beta = 6
##A = 2
##a = 1
##B = 2
##b = 2
##
##C = 4.001
##D = 1
##
##I = 0.90 * (C + D) / D
##
##c = 1.0
##firing_times = [0.0, 0.6, 0.95, 1.3, 1.65, 1.95, 2.3, 2.7, 3.1, 3.5, 4.0]

beta = 6
A = 2
a = 1
B = 2
b = 2

C = 2.001
D = 1

#(Trying to) get 2 spikes immediate
c = 1.78
t = 0.62

#2 spikes with a moderate gap
c = 2.6
t = 1.8

###2 spikes with a larger gap
##c = 2.70
##t = 2.2

###"Separated" solution, C = 2 case, presumed stable
##c = 2.71
##t = 5.0

###2 close together, but wants a third firing
##c = 0.3
##t = 1.1
##
##
##C = 0.001
###Normal and direct 2-spike
##c = 0.84
##t = 1.3
##
###Somehow this also works
###Unstable case?
##c = 0.2
##t = 3.4
##
###Wave where the s-troughs are overlapping
###Unstable solution, presumably
##c = 0.36
##t = 13
##
###Wave where the two packets are ~independent
###Can take whatever firing time you like above ~7
##c = 1.7
##t = 7
##
##C = 2.001

##
###Not quite separated, and has post-firing anyway so it doesn't work
##c = 0.64
##t = 6



firing_times = [0, t]
u_at_firing_times = firing_times[:]

v_r = 0

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

#-----------------------------------------------------------------------------
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
    
#-----------------------------------------------------------------------------

steps = 601
t_values = np.linspace(-3, 9, steps)
voltage = np.zeros(steps)
wigglage = np.zeros(steps)
synapse = np.zeros(steps)

#Finding values of u at firing times
for k, t in enumerate(firing_times):
    if k == 0:
        u_at_firing_times[k] = u(t)
    else:
        u_at_firing_times[k] = u(t, firing_times[k-1], u_at_firing_times[k-1])

print(v(0))
print(v(firing_times[1], 0, u_at_firing_times[0]))

for n in range(steps):
    t = t_values[n]
    t_last = None
    u_last = None
    for k, firing_time in enumerate(firing_times):
        if firing_time < t:
            t_last = firing_time
            u_last = u_at_firing_times[k]
    voltage[n] = v(t, t_last, u_last)
    #voltage[n] = v(t)
    wigglage[n] = u(t, t_last, u_last)
    #wigglage[n] = u(t)
    synapse[n] = s(t)


plt.figure()
plt.plot(t_values, synapse, c="#ff7f41", label="$s$")
plt.plot(t_values, wigglage, c="#9569be", label="$u$")
plt.plot(t_values, voltage, c="#007d69", label="$v$")

plt.gca().invert_xaxis()

plt.title("""\n
Travelling wave profile for multi-spike wave, $c$ = {}
""".format(c))
plt.xlabel("Co-moving coordinate $t - x/c$")
plt.ylabel("Value")

plt.axhline(1, linestyle="dashed", c="#999999", label="$v_{th}$")
plt.axhline(0, linestyle="dotted", c="#999999", label="$v_r$")
plt.axvline(0, c="#999999")
plt.legend(loc = "lower left")
plt.show()

