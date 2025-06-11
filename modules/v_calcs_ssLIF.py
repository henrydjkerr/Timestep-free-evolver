"""
Device functions for calculating the voltage and its derivative at given times.
"""

from numba import cuda
from math import pi, e, cos, sin, cosh, sinh, atan, atanh

from modules.general.ParamPlus import lookup

#------------------------------------------------------------------------------
#Basic definitions and functions

@cuda.jit(device = True)
def deriv_pq(R, D):
    p = 0.5*(D+1)
    q2 = 0.25 * ((D - 1)**2 - 4*R)
    if q2 >= 0:
        abs_q = q2**0.5
    else:
        abs_q = (-q2)**0.5
    return p, abs_q, q2

@cuda.jit(device = True)
def c(x, q2):
    if q2 < 0:
        return cos(x)
    else:
        return cosh(x)

@cuda.jit(device = True)
def s(x, q2):
    if q2 < 0:
        return sin(x)
    else:
        return sinh(x)


#Functions for deriving coefficients
@cuda.jit(device = True)
def part_c(v_0, s_0, u_0, I, beta, R, D):
    return v_0 - coeff_synapse(s_0, beta, R, D) - coeff_const(I, R, D)

@cuda.jit(device = True)
def part_s(v_0, s_0, u_0, I, beta, R, D):
    p, abs_q, q2 = deriv_pq(R, D)
    value = s_0 * (p**2 + q2 - p*(beta + 1) + beta) / ((p - beta)**2 - q2)
    value += I * (p**2 + q2 - p) / (p**2 - q2) + v_0 * (1 - p) + u_0
    value /= -abs_q
    return value

@cuda.jit(device = True)
def coeff_trig(v_0, s_0, u_0, I, beta, R, D):
    #Coefficient of the "trigonometric" terms in the true voltage equation
    #Though it's the same if they're not trigonometric, actually
    c_part = part_c(v_0, s_0, u_0, I, beta, R, D)
    s_part = part_s(v_0, s_0, u_0, I, beta, R, D)
    if (c_part >= 0):
        sign = 1
    else:
        sign = -1
    return sign * (c_part**2 + s_part**2)**0.5

@cuda.jit(device = True)
def trig_phase(v_0, s_0, u_0, I, beta, R, D):
    #Calculates the phase offset of the trigonometric/hyperbolic terms
    p, abs_q, q2 = deriv_pq(R, D)
    c_part = part_c(v_0, s_0, u_0, I, beta, R, D)
    s_part = part_s(v_0, s_0, u_0, I, beta, R, D)
    if c_part == 0:
        return pi/2
    elif q2 < 0:
        #Trigonometric case
        return atan(-s_part/c_part)
    else:
        #Hyperbolic case
        return atanh(s_part/c_part)
    #TODO: what if c_part = 0?

@cuda.jit(device = True)
def coeff_synapse(s_0, beta, R, D):
    #Coefficient of the synapse decay term in the true voltage equation
    p, abs_q, q2 = deriv_pq(R, D)
    return s_0 * (2*p - beta - 1) / ((p - beta)**2 - q2)

@cuda.jit(device = True)
def coeff_const(I, R, D):
    #Long term limit in the true voltage equation
    p, abs_q, q2 = deriv_pq(R, D)
    return I * (2*p - 1) / (p**2 - q2)

#------------------------------------------------------------------------------


#Functions for the actual value of v and u
@cuda.jit(device = True)
def get_vt(t, v_0, s_0, u_0, I, beta, R, D):
    """Calculates v(t) from initial conditions"""
    p, abs_q, q2 = deriv_pq(R, D)
    T = coeff_trig(v_0, s_0, u_0, I, beta, R, D)
    theta = trig_phase(v_0, s_0, u_0, I, beta, R, D)
    B = coeff_synapse(s_0, beta, R, D)
    K = coeff_const(I, R, D)
    return T*(e**(-p * t))*c(abs_q*t + theta, q2) + B*e**(-beta * t) + K

@cuda.jit(device = True)
def get_dvdt(t, v_actual, v_0, s_0, u_0, I, beta, R, D):
    """Calculates the derivative of v(t) given its current value"""
    u_actual = get_ut(t, v_0, s_0, u_0, I, beta, R, D)
    return I - v_actual - u_actual + s_0*e**(-beta * t)

@cuda.jit(device = True)
def get_ut(t, v_0, s_0, u_0, I, beta, R, D):
    """Calculates u(t) from initial conditions"""
    p, abs_q, q2 = deriv_pq(R, D)
    s_cluster = R * s_0 / ((p - beta)**2 - q2)
    I_cluster = R * I / (p**2 - q2)
    
    part_c = u_0 - s_cluster - I_cluster
    part_c *= e**(-p * t) * c(abs_q * t, q2)

    part_s = (s_cluster * (p - beta)) + (I_cluster * p) + ((p - 1)* u_0) - R*v_0
    part_s *= e**(-p * t) * s(abs_q * t, q2) / abs_q

    part_beta = e**(-beta * t) * s_cluster
    return part_c - part_s + part_beta + I_cluster

#For the upper bound of v
@cuda.jit(device = True)
def get_vt_upper(t, v_0, s_0, u_0, I, beta, R, D):
    """Calculates the upper bound on v(t) given by setting cos = 1"""
    p, abs_q, q2 = deriv_pq(R, D)
    A = abs(coeff_trig(v_0, s_0, u_0, I, beta, R, D))
    B = coeff_synapse(s_0, beta, R, D)
    K = coeff_const(I, R, D)
    return A*e**(-p * t) + B*e**(-beta * t) + K

@cuda.jit(device = True)
def get_dvdt_upper(t, v_0, s_0, u_0, I, beta, R, D):
    """Calculates the derivative of the upper bound on v(t)"""
    p, abs_q, q2 = deriv_pq(R, D)
    A = abs(coeff_trig(v_0, s_0, u_0, I), beta, R, D)
    B = coeff_synapse(s_0, beta, R, D)
    return -(p * A*e**(-p * t) + beta * B*e**(-beta * t))
    
