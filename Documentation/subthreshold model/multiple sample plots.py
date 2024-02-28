import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import e, cos, sin, cosh, sinh, atan


#------------------------------------------------------------------------------    

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

#Equations for v, u using the p, q formulation
#These ones work fine
def ana_v_pq(t):
    s_cluster = s_0 * (2*p - beta - 1) / (p**2 - q2 -2*p*beta + beta**2)
    I_cluster = I * (2*p - 1) / (p**2 - q2)
    part_c = v_0 - s_cluster - I_cluster
    part_c *= e**(-p * t) * c(abs_q * t)

    part_s = s_0 * (p**2 + q2 - p*(beta + 1) + beta) / (p**2 - q2 -2*p*beta + beta**2)
    part_s += I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0
    part_s *= e**(-p * t) * s(abs_q * t) / abs_q

    part_beta = e**(-beta * t) * s_cluster
    return part_c - part_s + part_beta + I_cluster

#This had an error but I've fixed it
def ana_u_pq(t):
    s_cluster = C * s_0 / ((p - beta)**2 - q2)
    I_cluster = C * I / (p**2 - q2)
    
    part_c = u_0 - s_cluster - I_cluster
    part_c *= e**(-p * t) * cos(abs_q * t)

    part_s = (s_cluster * (p - beta)) + (I_cluster * p) + ((p - 1)* u_0) - C*v_0
    part_s *= e**(-p * t) * sin(abs_q * t) / abs_q

    part_beta = e**(-beta * t) * s_cluster
    return part_c - part_s + part_beta + I_cluster

#------------------------------------------------------------------------------

#Testing the version where we combine cos and sin together

#Functions for deriving coefficients
def part_c(v_0, s_0, u_0, I):
    return v_0 - coeff_synapse(s_0) - coeff_const(I)

#def part_s(v_0, s_0, u_0, I):
#    return -((s_0 * (p**2 + q2 - p*(beta + 1) + beta)) \
#           + (I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0) / abs_q)

    #There was some error in the original form of this
def part_s(v_0, s_0, u_0, I):
    value = s_0 * (p**2 + q2 - p*(beta + 1) + beta) / ((p - beta)**2 - q2)
    value += I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0
    value /= -abs_q
    return value
    
def coeff_trig(v_0, s_0, u_0, I):
    #Coefficient of the "trigonometric" terms in the true voltage equation
    #Though it's the same if they're not trigonometric, actually
    c_part = part_c(v_0, s_0, u_0, I)
    s_part = part_s(v_0, s_0, u_0, I)
    if (c_part >= 0):
        sign = 1
    else:
        sign = -1
    return sign * ((c_part**2 + s_part**2)**0.5)

def trig_phase(v_0, s_0, u_0, I):
    #Calculates the phase offset of the trigonometric/hyperbolic terms
    c_part = part_c(v_0, s_0, u_0, I)
    s_part = part_s(v_0, s_0, u_0, I)
    if q2 < 0:
        #Trigonometric case
        return atan(-s_part/c_part)
    else:
        #Hyperbolic case
        return atanh(s_part/c_part)
    #TODO: what if c_part = 0?

def coeff_synapse(s_0):
    #Coefficient of the synapse decay term in the true voltage equation
    #Same as ana_pq as long as brackets expand OK
    return s_0 * (2*p - beta - 1) / ((p - beta)**2 - q2)

def coeff_const(I):
    #Long term limit in the true voltage equation
    #Same as ana_pq
    return I * (2*p - 1) / (p**2 - q2)

#Functions for the actual value of v and u
def get_vt(t, v_0, s_0, u_0, I):
    """Calculates v(t) from initial conditions"""
    T = coeff_trig(v_0, s_0, u_0, I)
    theta = trig_phase(v_0, s_0, u_0, I)
    B = coeff_synapse(s_0)
    K = coeff_const(I)
    return T*(e**(-p * t))*cos(abs_q*t + theta) + B*e**(-beta * t) + K
    #Seems like it has to be cos(q(t + theta)) rather than cos(qt + theta)?
    #Or not?

def get_ut(t, v_0, s_0, u_0, I):
    """Calculates u(t) from initial conditions"""
    s_cluster = C * s_0 / ((p - beta)**2 - q2)
    I_cluster = C * I / (p**2 - q2)
    
    part_c = u_0 - s_cluster - I_cluster
    part_c *= e**(-p * t) * cos(abs_q * t)

    part_s = (s_cluster * (p - beta)) + (I_cluster * p) + ((p - 1)* u_0) - C*v_0
    part_s *= e**(-p * t) * sin(abs_q * t) / abs_q

    part_beta = e**(-beta * t) * s_cluster
    return part_c - part_s + part_beta + I_cluster

#------------------------------------------------------------------------------

steps = 5000
length = 5

v_0 = 0.5
s_0 = 2
u_0 = -1
I = 0.9
C = 4
D = 1
beta = 1


delta = length/steps
t_values = np.arange(0, length, delta)

v_values = []
u_values = []

n = 4
for k in range(n):
    p = 0.5*(D+1)
    q_proto = (D - 1)**2 - 4*C
    if q_proto < 0:
        abs_q = 0.5*(-q_proto)**0.5
    else:
        abs_q = 0.5*q_proto**0.5
    q2 = 0.25*q_proto
    if q2 > 0:
        print("Outside oscillatory regime")
        raise ValueError
    
    v_values.append(np.zeros(steps))
    u_values.append(np.zeros(steps))
    for x in range(steps):
        t = t_values[x]
        v_values[-1][x] = get_vt(t, v_0, s_0, u_0, I)
        u_values[-1][x] = get_ut(t, v_0, s_0, u_0, I)
    C *= 2
    D *= 2


cmap = matplotlib.colormaps["YlOrRd_r"]
#cmap = matplotlib.colormaps["Wistia_r"]


plt.figure()
x_axis = t_values
for k in range(n):
    #plt.plot(t_values, v_values[k], c = (1.0, k/n, 1 - k/n))
    #plt.plot(t_values, u_values[k], c = (k/n, 1 - k/n, 1.0))
    #plt.plot(t_values, v_values[k], c = (1.0, k/n, 0))
    #plt.plot(t_values, u_values[k], c = (1.0, k/n, 0), linestyle = "dashed")
    plt.plot(t_values, v_values[k], c = cmap(0.8*(k+1)/n),
             label = "Rate = {}".format(2**k))
    

plt.title("Single-neuron voltage response for varying ion channel time constants")
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.axhline(1, linestyle="dotted", c="#0022cc", label = "Firing threshold")
#norm = matplotlib.colors.LogNorm(vmin = 1, vmax = 2**n)
#plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
plt.legend()
plt.show()
