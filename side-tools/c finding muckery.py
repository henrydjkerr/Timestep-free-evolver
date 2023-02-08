import numpy
import matplotlib.pyplot as plt

from math import pi, e, erf

beta = 1.001
A = 1
a = 1
B = 0
b = 1
Delta = 1
sqrt2 = 2**0.5

def part(Z, z, c):
    group_1 = z / (c * sqrt2)
    group_2 = group_1 * beta
    coeff = Z*z * e**(group_2**2)
    part_1 = 0.5 * (1 + erf(-group_2))
    part_2 = e**(0.5 * (1 - beta**2) * z**2 / c**2) \
             * 0.5 * (1 + erf(-group_1))
    return coeff * (part_1 - part_2)

def v_thI(c):
    value = (beta/(1 - beta)) * ((2*pi)**0.5) \
            * (part(A, a, c) - part(B, b, c))
    return value




points = 20
x_values = numpy.linspace(8, 9, points)
y_values = numpy.zeros(points)
for key in range(points):
    print(x_values[key])
    y_values[key] = v_thI(x_values[key])

plt.figure()
plt.plot(x_values, y_values)
plt.axhline(y = 0.1)
plt.show()
