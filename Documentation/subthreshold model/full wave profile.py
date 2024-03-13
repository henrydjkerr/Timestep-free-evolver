import numpy as np
import matplotlib.pyplot as plt
from math import e, erf, pi, cos, sin

beta = 4
A = 2
a = 1
B = 2
b = 3

I = 3.4

C = 6
D = 2

c = 3.24

##is_2D = False
##
###If in 2D:
##if is_2D:
##    A = A * a * ((2*pi)**0.5)
##    B = B * b * ((2*pi)**0.5)

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
#Redoing

def v(t):
    return (I * (2*p - 1) / (p**2 - q2)) \
           + beta * (part_v(A, a, t) - part_v(B, b, t))

def part_v(Z, z, t):
    coeff_cos = (2*p - beta - 1) / ( (p - beta)**2 - q2 )
    coeff_sin = (p**2 + q2 - p*(beta + 1) + beta) / ( (p - beta)**2 - q2 )
    coeff_sin /= abs_q

    part_p = -coeff_cos * calc_integral(z, t, p, "cosine")
    part_p += coeff_sin * calc_integral(z, t, p, "sine")

    part_beta = coeff_cos * calc_integral(z, t, beta, None)
    return Z * (part_p + part_beta)


def u(t):
    return C*I / (p**2 - q2) \
           + C * beta * (part_u(A, a, t) - part_u(B, b, t))

def part_u(Z, z, t):
    coeff_cos = 1 / ( (p - beta)**2 - q2 )
    coeff_sin = (p - beta) / ( (p - beta)**2 - q2 )
    coeff_sin /= abs_q

    part_p = -coeff_cos * calc_integral(z, t, p, "cosine")
    part_p += coeff_sin * calc_integral(z, t, p, "sine")

    part_beta = coeff_cos * calc_integral(z, t, beta, None)
    return Z * (part_p + part_beta)


def s(t):
    return beta * (part_s(A, a, t) - part_s(B, b, t))

def part_s(Z, z, t):
    return Z * calc_integral(z, t, beta, None)

#And then the new integral solver (numerical midpoint method)

def one(x):
    return 1

def calc_integral(z, t, param, func_id,
                  lower = None, upper = None, param2 = None):
    if upper == None:
        upper = t
    if param2 == None:
        param2 = param
        
    if func_id == "sine":
        func = sin
    elif func_id == "cosine":
        func = cos
    else:
        func = one

    #sigma = z/c
    #mu = sigma**2 * param

    divisions = 500
    max_length = 5 #3*sigma
    if lower != None and upper - lower < max_length:
        length = upper - lower
    else:
        length = max_length
    dT = length/divisions
    total = 0
    for x in range(divisions):
        T = upper - (x+0.5)*dT  #Midpoint method
        value = func(abs_q * (T - t))     \
                * (c / (z * (2*pi)**0.5)) \
                * e**( -(c**2 / (2*z**2)) * T**2 + param*T - param2*t)
        total += dT * value
    return total

#-----------------------------------------------------------------------------

v_r = 0
def v_after(t):
    values_I =   I * ((2*p - 1) / (p**2 - q2))        \
               * (1 - e**(-p * t) * cos(abs_q * t))   \
               - I * (p**2 + q2 - p)                  \
               / ((p**2 - q2) * abs_q) * e**(-p*t) * sin(abs_q * t)

    values_init = e**(-p * t)                   \
                  * ( v_r * cos(abs_q * t)      
                     - ((v_r * (1 - p) + u(0)) / abs_q) * sin(abs_q * t))

    return values_I + values_init + beta * (part_v_after(A, a, t) -
                                            part_v_after(B, b, t))

def part_v_after(Z, z, t):
    coeff_diff = (2*p - beta - 1) / ((p - beta)**2 - q2)
    coeff_sum = (p**2 + q2 - p*(beta + 1) + beta) \
                / (((p - beta)**2 - q2) * abs_q)

    cos_inside = coeff_diff * calc_integral(z, t, p, "cosine", lower = 0)
    sin_inside = coeff_sum  * calc_integral(z, t, p, "sine", lower = 0)
    part_beta  = coeff_diff * calc_integral(z, t, beta, None)
    trig_outside = (coeff_diff * cos(abs_q * t)
                    + coeff_sum * sin(abs_q * t)) \
                    * calc_integral(z, t, beta, None, upper = 0, param2 = p)
    return Z * (-cos_inside + sin_inside + part_beta - trig_outside)
                    
    



#-----------------------------------------------------------------------------

steps = 800
t_values = np.linspace(-4, 4, steps)
voltage = np.zeros(steps)
wigglage = np.zeros(steps)
synapse = np.zeros(steps)

for n in range(steps):
    t = t_values[n]
    if t <= 0:
        voltage[n] = v(t)
        wigglage[n] = u(t)
        synapse[n] = s(t)
    else:
        #voltage[n] = v_after(t)
        voltage[n] = v(t)
        wigglage[n] = u(t)
        synapse[n] = s(t)

plt.figure()
plt.plot(t_values, voltage, c="#0033dd", label="$v$")
plt.plot(t_values, synapse, c="#33dd00", label="$s$")
plt.plot(t_values, wigglage, c="#dd0033", label="$u$")

plt.title("""\n
Variables for leftwards-moving ssLIF travelling wave, $c$ = {}
""".format(c))
plt.xlabel("Time")
plt.ylabel("Value")

plt.axhline(1, linestyle="dashed", c="#999999", label="$v_{th}$")
plt.axhline(0, linestyle="dotted", c="#999999", label="$v_r$")
plt.axvline(0, c="#999999")
plt.legend(loc = "upper left")
plt.show()

