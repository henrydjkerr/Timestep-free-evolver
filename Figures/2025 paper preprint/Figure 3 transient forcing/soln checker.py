import numpy as np
from matplotlib import pyplot as plt
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
    return gauss((t-delay) - sigma ) - gauss((t-delay) + sigma)


dt = 0.0001
end = delay * 2
t_array = np.arange(0, end, dt)
points = len(t_array)
v = np.zeros(points)
u = np.zeros(points)
i = np.zeros(points)
#u[0] = u_0

for point in range(1, points):
    t = t_array[point]
    i[point] = drive(t)
    v[point] = v[point-1] + dt * (-v[point-1] - u[point-1] + drive(t))
    #if v[point] >= 1:
    #    v[point] = 0
    u[point] = u[point-1] + dt * (R * v[point-1] - D * u[point-1])

fig = plt.figure(figsize=(4,3), dpi=200)

plt.axhline(y=0, c="gray", linestyle="dashed")

#plt.plot(t_array, i, label="External forcing I(t)", c="#ff7f41")
#plt.plot(t_array, u, label="$u$", c="#9569be")
#plt.plot(t_array, v, label="$v$", c="#007d69")
plt.plot(t_array, i, label="Forcing $s(t)$", c=(0.9, 0.5, 0.75))
plt.plot(t_array, u, label="$u$", c=(0.1, 0.5, 0.1))
plt.plot(t_array, v, label="$v$", c=(0.0, 0.1, 0.3))

#plt.axhline(u_0, c="pink", label="$u_0$")
#plt.axhline(R/D, c="purple", label="$u$ steady state")
#plt.axhline(1, c="gray", linestyle="dotted")
n = 0
#while True:
#    n += 1
#    this_T = (1/freq) * n
#    if this_T > end:
#        break
#    else:
#        plt.axvline(this_T, c="gray", linestyle="dotted")
plt.ylim(-0.4, 0.4)
plt.legend(loc="upper left", reverse=True)
plt.xlim(0, end)
plt.xticks(np.arange(0, 20.1, 5))
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Example transient forcing response")#; $R$ = {}, $D = {}, $σ$ = {}$".format(R, D, sigma))
plt.show()

