import numpy
from math import pi, e, erf
sqrt2 = 2**0.5

#Rewritten for 1D, I hope

def part_v(Z, z, beta, c, t):
    coeff = Z * e**((z*beta / (c * sqrt2))**2)
    part_one = e**((1 - beta)*t) * 0.5 \
               * (1 + erf((c/(z*sqrt2)) * (t - (z**2 * beta / c**2))))
    part_two = e**(0.5*(1 - beta**2) * (z/c)**2) * 0.5 \
               * (1 + erf((c/(z*sqrt2)) * (t - (z**2 / c**2))))
    return coeff * (part_one - part_two)

def v(I, A, a, beta, c, t):
    value = I + (beta / (1 - beta)) * e**(-t) * (1/c) \
            * part_v(A, a, beta, c, t)
    return value

def part_s(Z, z, beta, c, t):
    coeff = Z * e**((z*beta / (c * sqrt2))**2)
    value = 0.5 * (1 + erf((c / (z * sqrt2)) * (t - z**2 * beta / c**2)))
    return coeff * value

def s(A, a, beta, c, t):
    value = beta * e**(-beta * t) * (1/c) * \
            part_s(A, a, beta, c, t)
    return value

def make(I, A, a, beta, c, x_dim, x_points):
    #Generates initial profile for the travelling wave
    coordinates = numpy.arange(-x_dim/2, x_dim/2, x_dim/x_points)
    voltage = numpy.zeros(x_points)
    synapse = numpy.zeros(x_points)

    for k, x in enumerate(coordinates):
        t = x/c
        if t <= 0:
            voltage[k] = v(I, A, a, beta, c, t)
            synapse[k] = s(A, a, beta, c, t)
        else:
            voltage[k] = I * (1 - e**(-x))
            synapse[k] = 0
    #return coordinates, voltage, synapse
    return voltage, synapse
    
