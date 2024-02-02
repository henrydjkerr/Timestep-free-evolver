"""
Device functions for calculating the voltage and its derivative at given times.
"""

from numba import cuda
from math import pi, e, cos, sin, cosh, sinh, atan, atanh

from modules.general.ParamPlus import lookup

beta = lookup["synapse_decay"]
C = lookup["C"]
D = lookup["D"]

#------------------------------------------------------------------------------
#Basic definitions and functions

p = 0.5*(D+1)
q_proto = (D - 1)**2 - 4*C
if q_proto < 0:
    abs_q = 0.5*(-q_proto)**0.5
else:
    abs_q = 0.5*q_proto**0.5
q2 = 0.25*q_proto
period = 2*pi / abs_q

@cuda.jit(device = True)
def c(x):
    if q2 < 0:
        return cos(x)
    else:
        return cosh(x)

@cuda.jit(device = True)
def s(x):
    if q2 < 0:
        return sin(x)
    else:
        return sinh(x)


#Functions for deriving coefficients
@cuda.jit(device = True)
def part_c(v_0, s_0, u_0, I):
    return v_0 - coeff_synapse(s_0) - coeff_const(I)

@cuda.jit(device = True)
def part_s(v_0, s_0, u_0, I):
    return -((s_0 * (p**2 + q2 - p*(beta + 1) + beta)) \
           + (I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0) / abs_q)

@cuda.jit(device = True)
def coeff_trig(v_0, s_0, u_0, I):
    #Coefficient of the "trigonometric" terms in the true voltage equation
    #Though it's the same if they're not trigonometric, actually
    c_part = part_c(v_0, s_0, u_0, I)
    s_part = part_s(v_0, s_0, u_0, I)
    if (c_part >= 0):
        sign = 1
    else:
        sign = -1
    return sign * (c_part**2 + s_part**2)**0.5

@cuda.jit(device = True)
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

@cuda.jit(device = True)
def coeff_synapse(s_0):
    #Coefficient of the synapse decay term in the true voltage equation
    return s_0 * (2*p - beta - 1) / ((p - beta)**2 - q2)

@cuda.jit(device = True)
def coeff_const(I):
    #Long term limit in the true voltage equation
    return I * (2*p - 1) / (p*2 - q2)

#------------------------------------------------------------------------------


#Functions for the actual value of v and u
@cuda.jit(device = True)
def get_vt(t, v_0, s_0, u_0, I):
    """Calculates v(t) from initial conditions"""
    T = coeff_trig(v_0, s_0, u_0, I)
    theta = trig_phase(v_0, s_0, u_0, I)
    B = coeff_synapse(s_0)
    K = coeff_const(I)
    return T*e**(-p * t)*c(abs_q*t + theta) + B*e**(-beta * t) + K

@cuda.jit(device = True)
def get_dvdt(t, v_actual, v_0, u_0, I):
    """Calculates the derivative of v(t) given its current value"""
    u_actual = get_ut(t, v_0, s_0, u_0, I)
    return I - v_actual - u_actual + s_0*e**(-beta * t) + I

@cuda.jit(device = True)
def get_ut(t, v_0, s_0, u_0, I):
    """Calculates u(t) from initial conditions"""
    s_cluster = C * s_0 / (p**2 - q2 - 2*p*beta + beta**2)
    I_cluster = C * I / (p**2 - q2)
    part_c = -(s_cluster + I_cluster - u_0)
    part_c *= e**(-p * t) * c(abs_q * t)

    part_s = s_cluster * p + I_cluster * (p - beta) - C * v_0 + (p - 1) * u_0
    part_s *= -e**(-p * t) * s(abs_q * t) / abs_q

    part_beta = e**(-beta * t) * s_cluster
    return part_c + part_s + part_beta + I_cluster

#For the upper bound of v
@cuda.jit(device = True)
def get_vt_upper(t, v_0, s_0, u_0, I):
    """Calculates the upper bound on v(t) given by setting cos = 1"""
    T = coeff_trig(v_0, s_0, u_0, I)
    B = coeff_synapse(s_0)
    K = coeff_const(I)
    return T*e**(-p * t) + B*e**(-beta * t) + K

@cuda.jit(device = True)
def get_dvdt_upper(t, v_0, s_0, u_0, I):
    """Calculates the derivative of the upper bound on v(t)"""
    T = coeff_trig(v_0, s_0, u_0, I)
    B = coeff_synapse(s_0)
    return -(p * T*e**(-p * t) + beta * B*e**(-beta * t))
    
