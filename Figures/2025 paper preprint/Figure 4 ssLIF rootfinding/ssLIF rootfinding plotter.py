import numpy as np
import matplotlib.pyplot as plt

from math import pi, e, erf, log

import v_calcs_ssLIF as v_calcs













#------------------------------------------------------------------------------

black = np.array((0, 0.1, 0.3))
green = np.array((0.1, 0.5, 0.1))
pink = np.array((0.9, 0.5, 0.75))
orange = np.array((1.0, 0.75, 0.55))

points = 500
length = 3.5
delta = length/points

t_values = np.linspace(0, length, points)
s_values = np.zeros(points)
u_values = np.zeros(points)
v_values = np.zeros(points)

v_th = 1
v_r = 0

R = 3
D = 1
beta = 2
I = 0.8 * (R + D) / D

firing = False
inset = True
if firing:
    v_0 = 0.5
    s_0 = -2.5
    u_0 = 4
else:
    v_0 = 0.5
    s_0 = -1.9
    u_0 = 4



#Derived values
p = 0.5*(D+1)
q2 = 0.25 * ((D - 1)**2 - 4*R)
if q2 >= 0:
    abs_q = q2**0.5
else:
    abs_q = (-q2)**0.5

A = v_calcs.coeff_trig(v_0, s_0, u_0, I, beta, R, D)
B = v_calcs.coeff_synapse(s_0, beta, R, D)
K = v_calcs.coeff_const(I, R, D)

start_time = 0
end_time = -(1/min(p, beta)) * log(abs(v_th - K) / (abs(A) + abs(B)))

for k in range(0, points):
    t = t_values[k]
    v_values[k] = v_calcs.get_vt(t, v_0, s_0, u_0, I, beta, R, D)


rootfind_x = [start_time]
rootfind_y = [v_0]

error_bound = 0.00001
#Start iterations
t_old = start_time
N = 0
for count in range(100):
    N += 1
    #Calculate upper bounds on derivatives
    #You need fewer iterations if you do it inside the loop
####    Mvelo = abs(A * (p**2 + abs_q**2)**0.5) * e**(-p * start_time) \
####            + max(-beta * B * e**(-beta * start_time),
####                  -beta * B * e**(-beta * end_time))
####    Maccel = max(abs(A * (p**4 + abs_q**4)**0.5) * e**(-p * start_time) \
####                 + max(beta**2 * B * e**(-beta * start_time),
####                       beta**2 * B * e**(-beta * end_time)),
####                 0)
    Mvelo = (p + abs_q) * abs(A) * e**(-p * t_old) \
            + max(-beta * B * e**(-beta * t_old),
                  -beta * B * e**(-beta * end_time))
    Maccel = max((p**2 + abs_q**2) * abs(A) * e**(-p * t_old) \
                 + max(beta**2 * B * e**(-beta * t_old),
                       beta**2 * B * e**(-beta * end_time)),
                 0)
##    Mvelo = (p + abs_q) * abs(A) * e**(-p * t_old) \
##            + beta * abs(B) * e**(-beta * t_old)
##    Maccel = max((p**2 + abs_q**2) * abs(A) * e**(-p * t_old) \
##                 + beta**2 * abs(B) * e**(-beta * t_old),
##                 0)
    if Mvelo <= 0:
        #Gradient is negative from here on, so you'll never cross
        print("M_n <= 0, halting, N =", N)
    #Perform modified Newton-Raphson method
    v_test = v_calcs.get_vt(t_old, v_0, s_0, u_0, I, beta, R, D)
    v_deriv = v_calcs.get_dvdt(t_old, v_test, v_0, s_0, u_0, I,
                                 beta, R, D)
    m = min(Mvelo,
            0.5*(v_deriv \
                 + (v_deriv**2 + 4*Maccel*(v_th - v_test))**0.5))
    #print(N, v_deriv, m, Maccel)
    #if N == 9: break
    if m <= 0:
        print("m <= 0, halting, N =", N)
        break
    t_new = t_old + (v_th - v_test) / m
    rootfind_x.append(t_old)
    rootfind_y.append(v_test)
    rootfind_x.append(t_new)
    rootfind_y.append(v_th)
    if abs(t_new - t_old) <= error_bound:
        print("converged, halting, N =", N)
        break
    elif t_new > end_time:
        print("Left interval, halting, N =", N)
        break
    else:
        t_old = t_new
    #Currently silently failing if it takes too many iterations




#plt.figure(figsize=(4, 4), dpi=800)
if not inset:
    plt.figure(figsize=(5,4))
else:
    plt.figure(figsize=(2.5,2))
#plt.xlim((0, end_time))

plt.axvline(x=end_time, color="#999999", linestyle=":", label="T")
plt.axhline(y=v_th, color="#999999", linestyle="--", label="$v_\\text{th}$")


#plt.plot(t_values, s_values, color=pink, label="$s$")
plt.plot(t_values, v_values, c=black, label="$v$")
plt.plot(rootfind_x, rootfind_y, c=orange, label="Iterations")

if inset:
    plt.ylim((0.9, 1.02))
    plt.xlim((1.7, 2.7))
    plt.yticks([0.9, 0.96, 1.02])
else:
    plt.ylim((-0.4, 1.2))
    plt.yticks([-0.4, 0.0, 0.4, 0.8, 1.2])

#if firing:
#    plt.title("2-variable LIF with regular input, " + r"$\beta$ = " + str(beta))
plt.ylabel("$v$")
plt.xlabel("$t$")
#plt.title("Time graph of firing neuron")

if firing:
    plt.legend(loc="lower right", reverse = True)

plt.margins(x=0, y=0.01)

if not inset:
    plt.savefig("ssLIF_{}.pdf".format(firing))
else:
    plt.savefig("ssLIF_inset.pdf")
plt.show()
