import numpy as np
import matplotlib.pyplot as plt
from math import e, cos, sin, cosh, sinh, atan

steps = 50000
length = 5

delta = length/steps
t_values = np.arange(0, length, delta)
v_values = np.zeros(steps)
u_values = np.zeros(steps)

#Initial conditions, constants
v_0 = 5
u_0 = 0
s_0 = -4
I = 3
beta = 2
C = 10
D = 5

v_0 = -0.18
u_0 = 0.766
s_0 = 2.05
I = 0.718
beta = 1.995
C = 2.85
D = 2.077

v_0 = -0.45
s_0 = 2.201
u_0 = -0.83
I = 0.990
beta = 1.560
C = 2.183
D = 1.529

beta = 2.554
C = 2.729
D = 3.864
v_0 = 0.601
s_0 = 1.258
u_0 = -1.37
I = 0.737

v_0 = -0.7366913762715817
s_0 = 1.43877533028168
u_0 = 0.49549990146727874
I = 1.0666990972531414
beta = 3.454704939258241
C = 1.3194081275570249
D = 1.8612391512003994

v_0 = -0.12358751876809682
s_0 = -2.6534389873966973
u_0 = -1.640959313776784
I = 1.1929969370369464
C = 0.30041492031562556
D = 1.9667230932557858
beta = 1.21306163732562

#Now testing for why some things don't converge
v_0 = -0.5180116594868698
s_0 = -1.7183102990180616
u_0 = 1.4833584215619702
I = 0.8058996560587008
beta = 2.8515476604235435
C = 1.622501913545989
D = 3.3747412316766234

v_0 = -0.5344216674595554
s_0 = 2.6320532631604756
u_0 = 1.8376515726025984
I = 1.1992006037229164
C = 0.21701322110312685
D = 1.2876136046959845
beta = 1.4181985425471741

v_values[0] = v_0
u_values[0] = u_0

#------------------------------------------------------------------------------

#Forward-Euler derivative calculation (can't mess that one up, but numerics can)
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

#------------------------------------------------------------------------------

#Equations for v, u using the lambda formulation
#Something's wrong with these
#Not accounting for imaginary parts properly?
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
    
for x in range(0, steps):
    #v_analytic[x] = ana_v_pq(t_values[x])
    #u_analytic[x] = ana_u_pq(t_values[x])
    if q2 >= 0:
        print("Error: not in oscillatory regime")
    v_analytic[x] = get_vt(t_values[x], v_0, s_0, u_0, I)
    u_analytic[x] = get_ut(t_values[x], v_0, s_0, u_0, I)

plt.figure()
x_axis = t_values
y1_axis = v_values
y2_axis = u_values
y3_axis = v_analytic
y4_axis = u_analytic
plt.plot(x_axis, y1_axis, c="#ff0000")  #  Red is v(t), Euler
plt.plot(x_axis, y2_axis, c="#009900")  #Green is u(t), Euler
plt.scatter(x_axis, y3_axis, c="#000000")
plt.scatter(x_axis, y4_axis, c="#000000")
plt.axhline(1, linestyle="dashed", c="#0022cc")
plt.show()
