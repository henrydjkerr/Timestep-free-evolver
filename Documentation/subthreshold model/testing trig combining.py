import numpy as np
import matplotlib.pyplot as plt
from math import e, cos, sin, cosh, sinh, atan


steps = 5000
length = 5

delta = length/steps
t_values = np.arange(0, length, delta)
sin_values = np.zeros(steps)
cos_values = np.zeros(steps)
added = np.zeros(steps)
synth = np.zeros(steps)

sin_coeff = -3
cos_coeff = 5
rate = 4

combo_coeff = (sin_coeff**2 + cos_coeff**2) ** 0.5
if cos_coeff < 0:
    combo_coeff *= -1
phase_offset = atan(-sin_coeff / cos_coeff)


for x in range(steps):
    sin_values[x] = sin_coeff * sin(rate * t_values[x])
    cos_values[x] = cos_coeff * cos(rate * t_values[x])
    added[x] = sin_values[x] + cos_values[x]
    synth[x] = combo_coeff * cos(rate * t_values[x] + phase_offset)

plt.figure()
x_axis = t_values
plt.plot(x_axis, added, c="#ff0000")
plt.scatter(x_axis, synth, c="#000000")
plt.show()
