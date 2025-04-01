import numpy as np
import matplotlib.pyplot as plt

from math import pi, e, erf


#from matplotlib import rc
#rc("font", **{"family":"serif", "serif":["Times New Roman"]})
#rc("text", usetex=True)

def part(Z, z, x):
    value = Z/(z*(2*pi)**0.5) * e**(-(1/(2*z*z)) * x**2)
    return value

def hat(x):
    return part(A, a, x) - part(B, b, x)

#------------------------------------------------------------------------------

blue1 = np.array((0.2, 0.8, 0.9))
blue2 = np.array((0.15, 0.55, 0.7))
blue3 = np.array((0.1, 0.2, 0.3))
gold = np.array((0.85, 0.75, 0.2))
white = np.array((1.0, 1.0, 1.0))
fade = 0.67

black = np.array((0.0, 0.1, 0.3))
green = np.array((0.1, 0.5, 0.1))
pink  = np.array((0.9, 0.5, 0.75))

beta = 2

A = 2
a = 1
B = 2
b = 2

print_mode = False
if print_mode:
    plt.figure(figsize=(5, 4), dpi=400)
else:
    plt.figure(figsize=(5,4))

points = 651
d_values = np.linspace(-6.5, 6.5, points)
w_values = np.zeros(points)

flag = True
n = 0
for k in range(0, points):
    w_values[k] = hat(d_values[k])
    if flag:
        if w_values[k] < 0:
            n = k
        else:
            flag = False

        
#tnr = "Times New Roman"

plt.axhline(y=0, color="#999999", linestyle= "dotted")
plt.axvline(x=0, color="#999999",linestyle= "dotted")

plt.plot(d_values, w_values, color = black)

#plt.title("Mexican Hat function")
plt.ylabel("$w(d)$")
plt.xlabel("$d$ (distance between neurons)")

plt.fill_between(d_values[n : points - n], w_values[n : points - n],
                 hatch="///", edgecolor="white",
                 facecolor = green * (1 - fade) + white * fade)
plt.fill_between(d_values[:n+1], w_values[:n+1],
                 hatch="///", edgecolor="white",
                 facecolor = pink * (1 - fade) + white * fade)
plt.fill_between(d_values[points - n - 1:], w_values[points - n - 1:],
                 hatch="///", edgecolor="white",
                 facecolor = pink * (1 - fade) + white * fade)
plt.margins(x=0, y=0.2)

if print_mode:
    plt.savefig("mexican_hat.pdf")
    plt.savefig("mexican_hat.png")
else:
    plt.show()
