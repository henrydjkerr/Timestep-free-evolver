import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from numpy import cos, sin
from math import pi, e

R = 2
D = 1
#u_0 = 2.24356837426285
#freq = 0.8631161426652929 #T = 1.159

p = 0
q2 = 0
abs_q = 0

delay = 10
sigma = 1
speed = 1

def change_RD(new_R, new_D):
    global R, D, p, q2, abs_q
    R = new_R
    D = new_D
    p = 0.5*(D+1)
    q2 = 0.25*( (D-1)**2 - 4*R )
    abs_q = (-q2)**0.5
change_RD(R, D)

def gauss(x):
    return (1 / (sigma * (2*pi)**0.5)) * e**(-x**2 / (2*sigma**2))

def drive(t):
    return gauss(speed*(t-delay) - sigma ) - gauss(speed*(t-delay) + sigma)

#------------------------------------------------------------------------------

dt = 0.001
end = delay * 2
t_array = np.arange(0, end, dt)
timepoints = len(t_array)

v = np.zeros(timepoints)
u = np.zeros(timepoints)
i = np.zeros(timepoints)

points = 30
R_start = 0.0
R_length = 10
R_array = np.linspace(R_start, R_start + R_length, points)
D_start = 0.0
D_length = 10
D_array = np.linspace(D_start, D_start + D_length, points)

zero_offset = np.ndarray((points, points))

for x in range(points):
    for y in range(points):
        try:
            change_RD(R_array[y], D_array[x])
            zero_time = None
            for tp in range(timepoints):
                t = t_array[tp]
                i[tp] = drive(t)
                v[tp] = v[tp-1] + dt * (-v[tp-1] - u[tp-1] + drive(t))
                u[tp] = u[tp-1] + dt * (R * v[tp-1] - D * u[tp-1])

                if t >= delay-1 and v[tp] >= 0 and zero_time == None:
                    zero_time = t
                    break
            zero_offset[y, x] = zero_time - delay
        except TypeError:
            pass

level_select = np.arange(-1, 1.05, 0.1)
#level_select = np.arange(-1.5, 1.55, 0.1)

fig, ax = plt.subplots(figsize=(5,4), dpi=200)
#img = ax.imshow(zero_offset, origin="lower",
#                extent=(D_start, D_start + D_length,
#                        R_start, R_start + R_length),
#                aspect=D_length/R_length, cmap="PiYG",
#                norm=matplotlib.colors.CenteredNorm(vcenter=0))
img = ax.contour(D_array, R_array, zero_offset, level_select,
                 cmap="PiYG",
                 extent=(D_start, D_start + D_length,
                         R_start, R_start + R_length),
                 norm=matplotlib.colors.CenteredNorm(vcenter=0),
                 aspect=D_length/R_length)
fig.colorbar(img, ax=ax)

plt.xlabel("$D$ (Ion channel activity decay rate)")
plt.ylabel("$R$ (Ion channel response rate)")

plt.title("Delay between $s(t) = 0$ and $v(t) = 0$")
plt.show()

