import numpy
import matplotlib.pyplot as plt

from math import pi, e, erf

beta = 2
A = 1
a = 1
B = 0 #7
b = 0 #3.5
Delta = 1
sqrt2 = 2**0.5

def part(Z, z, c):
    part_value = Z*z * e**((z*beta / (c * sqrt2))**2)
    part_value *= (1 - e**(0.5 * (1 - beta**2) * z**2 / c**2))
    part_value *= 0.5 * (1 + erf(-z / (c * sqrt2)))
    print(Z, z, c, part_value)
    return part_value

def v_thI(c):
    value = (beta/(1 - beta)) * ((2*pi)**0.5 / c) \
            * (part(A, a, c) - part(B, b, c))
    return value


points = 20
x_values = numpy.linspace(-3.685627, -3.685626, points)
y_values = numpy.zeros(points)
for key in range(points):
    print(x_values[key])
    y_values[key] = v_thI(x_values[key])

plt.figure()
plt.plot(x_values, y_values)
plt.axhline(y = 0.1)
plt.show()
