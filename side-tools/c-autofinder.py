import numpy
import matplotlib.pyplot as plt

from math import e, erf

#Shouldn't be 1 for this program, just use 1.0001 or similar
beta = 2
A = 2
a = 1
B = 1.8
b = 2

target = 0.1    #v_th - I

#The follows is based on formula 8.17 in running review
#May differ from past versions in factors of (2*pi)**0.5 for Z

def sub_part(value):
    return e**(value**2) * 0.5 * (1 + erf(-value))

def part(Z, z, c):
    gamma = z / (c * 2**0.5)
    coeff = (beta / (1 - beta)) * Z
    return coeff * (sub_part(beta * gamma) - sub_part(gamma))

def whole(c):
    return part(A, a, c) - part(B, b, c)

#Assume the function is decreasing
#This is probably going to be a bit of a hack for now

c = 1
step = 1
margin = 0.0001
for count in range(100):
    test = whole(c)
    if test > target:
        if test - target < margin:
            break
        c += step
    else:
        step *= 0.5
        c -= step

if count == 99:
    print("Operation timed out")
print(count)
print(c)

points = 20
x_values = numpy.linspace(c - 0.2, c + 0.2, points)
y_values = numpy.zeros(points)
for key in range(points):
    y_values[key] = whole(x_values[key])

plt.figure()
plt.plot(x_values, y_values)
plt.axhline(y = target)
plt.show()
        
