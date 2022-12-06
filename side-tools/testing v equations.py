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
c_v = 3.686
c_s = 3.686

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

points = 50
lower_bound = -2
t_values = numpy.linspace(lower_bound, 0, points)
v_analytic = numpy.zeros(points)
v_numeric = numpy.zeros(points)
for k in range(points):
    v_analytic[k] = v(t_values[k])
    if k == 0:
        v_numeric[k] = I
    else:
        v_numeric[k] = v_numeric[k - 1] \
                       + abs(lower_bound)/points \
                       * (I - v_numeric[k - 1] + s(t_values[k - 1]))
    

plt.figure()
plt.plot(t_values, v_analytic)
plt.plot(t_values, v_numeric)
plt.axhline(y = 0)
plt.axhline(y = 1)
plt.axvline(x = 0)
plt.show()
