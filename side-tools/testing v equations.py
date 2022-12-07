import numpy
import matplotlib.pyplot as plt

from math import pi, e, erf

I = 0.9
beta = 2
A = 1
a = 1
B = 0
b = 1
Delta = 1
sqrt2 = 2**0.5
c = 3.48


def part_v(Z, z, t):
    coeff = Z*z * e**((z*beta / (c * sqrt2))**2)
    part_one = e**((1 - beta)*t) * 0.5 \
               * (1 + erf((c/(z*sqrt2)) * (t - (z**2 * beta / c**2))))
    part_two = e**(0.5*(1 - beta**2) * (z/c)**2) * 0.5 \
               * (1 + erf((c/(z*sqrt2)) * (t - (z**2 / c**2))))
    return coeff * (part_one - part_two)

def v(t):
    value = I + (beta / (1 - beta)) * Delta * e**(-t) * (2*pi)**0.5 * (1/c) \
            * (part_v(A, a, t) - part_v(B, b, t))
    return value

def part_s(Z, z, t):
    coeff = Z*z * e**((z*beta / (c * sqrt2))**2)
    value = 0.5 * (1 + erf((c / (z * sqrt2)) * (t - z**2 * beta / c**2)))
    return coeff * value

def s(t):
    value = beta * Delta * e**(-beta * t) * (2*pi)**0.5 * (1/c) * \
            (part_s(A, a, t) - part_s(B, b, t))
    return value

def part_connect(Z, z, t):
    #value = (Z / ((2*pi)**0.5 * z)) * e**(-0.5 * (c*t / z)**2)
    value = Z * e**(-0.5 * (c*t / z)**2)
    return value

def connect(t):
    return part_connect(A, a, t) - part_connect(B, b, t)    

points = 500
lower_bound = -2
step = abs(lower_bound)/points

t_values = numpy.linspace(lower_bound, 0, points)
v_analytic = numpy.zeros(points)
v_numeric = numpy.zeros(points)
v_very_numeric = numpy.zeros(points)
s_temp = 0
for k in range(points):
    #v_analytic
    v_analytic[k] = v(t_values[k])
    #v_numeric
    if k == 0:
        v_numeric[k] = I
    else:
        v_numeric[k] = v_numeric[k - 1] \
                       + step * (I - v_numeric[k - 1] + s(t_values[k - 1]))
    #v_very_numeric
    if k == 0:
        v_very_numeric[k] = I
    else:
        #s_temp *= e**(-beta * step)
        #s_temp += step * beta * connect(t_values[k])
        s_temp += step * beta * (connect(t_values[k]) - s_temp)
        v_very_numeric[k] = v_very_numeric[k - 1] \
                            + step * (I - v_numeric[k - 1] + s_temp)
    

plt.figure()
plt.plot(t_values, v_analytic)
plt.plot(t_values, v_numeric)
plt.plot(t_values, v_very_numeric)
plt.axhline(y = 0)
plt.axhline(y = 1)
plt.axvline(x = 0)
plt.xlabel("t (= x/c)")
plt.ylabel("v(t)")
plt.show()
