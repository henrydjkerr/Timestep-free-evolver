import numpy
import matplotlib.pyplot as plt

from math import pi, e, erf

I = 0.9
beta = 1.001
A = 2
a = 1
B = 0 #7
b = 1 #3.5
Delta = 1
sqrt2 = 2**0.5
c = 3.82

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


x_dim = 20
y_dim = 1
x_points = 500
y_points = 1

outfile = open("test_wave.csv", "w")
for j in range(x_points):
    x = x_dim *(0.5 + (j - x_points)/x_points)
    t = x/c
    if t <= 0:
        new_v = v(t)
        new_s = s(t)
        print("t={}, v={}, s={}".format(t, new_v, new_s))
    else:
        new_v = I * (1 - e**(-x))
        new_s = 0
    for k in range(y_points):
        y = y_dim *(0.5 + (k - y_points)/y_points)
        outfile.write("{},{},{},{}\n".format(x, y, new_v, new_s))
outfile.close()

