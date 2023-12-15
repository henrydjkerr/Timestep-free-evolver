import numpy as np
import matplotlib.pyplot as plt
from math import e, cos, sin, cosh, sinh

steps = 5000
length = 5

delta = length/steps
t_values = np.arange(0, length, delta)
v_values = np.zeros(steps)
u_values = np.zeros(steps)

v_0 = 1
u_0 = -5
v_values[0] = v_0
u_values[0] = u_0

s_0 = 5
I = 5
beta = 2

C = 20
D = 0.1

def dv(v, u, t):
    return I - v - u + (s_0 * e**(-beta * t))

def du(v, u, t):
    return C*v - D*u

for x in range(1, steps):
    v = v_values[x-1]
    u = u_values[x-1]
    t = t_values[x-1]
    v_values[x] = v_values[x-1] + delta * dv(v, u, t)
    u_values[x] = u_values[x-1] + delta * du(v, u, t)

#------------------------------------------------------------------------------
    
v_analytic = np.zeros(steps)
u_analytic = np.zeros(steps)

p = 0.5*(D+1)
q_proto = (D - 1)**2 - 4*C
if q_proto < 0:
    abs_q = 0.5*(-q_proto)**0.5
else:
    abs_q = 0.5*q_proto**0.5
q2 = 0.25*q_proto


L1 = - p - abs_q
L2 = - p + abs_q

def ana_v_lambda(t):
    part_1 = (L2 + 1) * (I/L1 + s_0/(L1 + beta) + v_0) + u_0
    part_1 *= -e**(L1 * t)
    part_2 = (L1 + 1) * (I/L2 + s_0/(L2 + beta) + v_0) + u_0
    part_2 *= e**(L2 * t)
    part_beta = e**(-beta * t) * s_0
    part_beta *= (L2+1)/(L1+beta) - (L1+1)/(L2+beta)
    part_I = I * ((L2+1)/L1 - (L1+1)/L2)
    return (part_1 + part_2 + part_beta + part_I) / (L1 - L2)

def ana_u_lambda(t):
    #Something wrong here
    part_1 = (I/L1 + s_0/(L1 + beta) + v_0) + (L1 + 1) * u_0 / C
    part_1 *= e**(L1 * t)
    part_2 = (I/L2 + s_0/(L2 + beta) + v_0) + (L2 + 1) * u_0 / C
    part_2 *= -e**(L2 * t)
    part_beta = -e**(-beta * t) * s_0
    part_beta *= 1/(L1+beta) - 1/(L2+beta)
    part_I = -I * (1/L1 - 1/L2)
    return C * (part_1 + part_2 + part_beta + part_I) / (L1 - L2)
    

def c(x):
    if q2 < 0:
        return cos(x)
    else:
        return cosh(x)

def s(x):
    if q2 < 0:
        return sin(x)
    else:
        return sinh(x)

def ana_v_pq(t):
    s_cluster = s_0 * (2*p - beta - 1) / (p**2 - q2 -2*p*beta + beta**2)
    I_cluster = I * (2*p - 1) / (p**2 - q2)
    part_c = v_0 - s_cluster - I_cluster
    part_c *= e**(-p * t) * c(abs_q * t)

    part_s = s_0 * (p**2 + q2 - p*(beta + 1) + beta) / (p**2 - q2 -2*p*beta + beta**2)
    part_s += I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0
    part_s *= e**(-p * t) * s(abs_q * t) / abs_q    #suspect

    part_beta = e**(-beta * t) * s_cluster
    return part_c - part_s + part_beta + I_cluster


def ana_u_pq(t):
    s_cluster = C * s_0 / (p**2 - q2 - 2*p*beta + beta**2)
    I_cluster = C * I / (p**2 - q2)
    part_c = -(s_cluster + I_cluster - u_0)
    part_c *= e**(-p * t) * c(abs_q * t)

    part_s = s_cluster * p + I_cluster * (p - beta) - C * v_0 + (p - 1) * u_0
    part_s *= -e**(-p * t) * s(abs_q * t) / abs_q

    part_beta = e**(-beta * t) * s_cluster
    return part_c + part_s + part_beta + I_cluster

    
for x in range(0, steps):
    v_analytic[x] = ana_v_pq(t_values[x])
    u_analytic[x] = ana_u_pq(t_values[x])

#------------------------------------------------------------------------------

plt.figure()
x_axis = t_values
y1_axis = v_values
y2_axis = u_values
y3_axis = v_analytic
y4_axis = u_analytic
plt.plot(x_axis, y1_axis, c="#ff0000")
plt.plot(x_axis, y2_axis, c="#009900")
plt.scatter(x_axis, y3_axis, c="#000000")
plt.scatter(x_axis, y4_axis, c="#000000")
plt.show()
