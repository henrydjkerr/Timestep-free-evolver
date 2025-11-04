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

delay = 40
sigma = 1

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

#------------------------------------------------------------------------------

dt = 0.001
end = delay * 1.5
t_array = np.arange(0, end, dt)
points = len(t_array)


R_runs = 5
R_array = [0.5, 1, 2, 4, 8]
runs = 40
c_array = np.linspace(0.01, 3, runs)
zero_offset = np.zeros((R_runs, runs))

v = np.zeros(points)
u = np.zeros(points)
i = np.zeros(points)

change_RD(2, 1)

for meta in range(R_runs):
    change_RD(R_array[meta], 1)
    for run in range(runs):
        #change_RD(2*R_array[run], R_array[run])
        #change_RD(R_array[run], D)
        c = c_array[run]
        sigma = 1/c
        try:
            zero_time = None
            for point in range(points):    
                t = t_array[point]
                i[point] = drive(t)
                v[point] = v[point-1] + dt * (-v[point-1] - u[point-1] + drive(t))
                u[point] = u[point-1] + dt * (R * v[point-1] - D * u[point-1])
            
                if t >= delay-1 and v[point] >= 0 and zero_time == None:
                    zero_time = t
                    break
            zero_offset[meta, run] = (zero_time - delay) * c
        except TypeError:
            pass


fig = plt.figure(figsize=(4, 3), dpi=200)
#for run in range(runs):
colour = np.array((0.8, 0.2, 0.4))
for k in range(R_runs):
    plt.plot(c_array, zero_offset[k], c=colour * (k + 1) / R_runs,
             label="$R$ = {}".format(R_array[k]))
    
plt.xlim(0, 3)
plt.ylim(-0.5, 1.25)
plt.xlabel("$c$ (Wave speed)")
plt.ylabel("$c(t - τ)$ (Relative delay)")
#plt.ylabel("Absolute delay $t - τ$")
plt.title("Response of delay to wave speed, $D = 1$")
plt.legend()
plt.show()

