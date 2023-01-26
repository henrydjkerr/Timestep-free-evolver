import numpy
import matplotlib.pyplot as plt

from math import pi, e, erf

import data_reader

#-------------------------------------------------------------------------------

def data_importer(filename):
    """
    Read data from file into numpy arrays then return the arrays.
    Assumes formatting of coord, (coord,) voltage, synapse
    """
    coordinates = numpy.zeros((neurons_number, dimension))
    voltage = numpy.zeros(neurons_number)
    synapse = numpy.zeros(neurons_number)

    infile = open(filename, "r")
    for n, line in enumerate(infile):
        if n >= neurons_number:
            raise IndexError("The source data file {} has too many entries \
to fit in the supplied arrays.".format(filename))
        entries = line.strip("\n").split(",")
        if len(entries) == 3:
            coordinates[n, 0] = float(entries[0])
            voltage[n] = float(entries[1])
            synapse[n] = float(entries[2])
        elif len(entries) == 4:
            coordinates[n, 0] = float(entries[0])
            #coordinates[n, 1] = float(entries[1])
            voltage[n] = float(entries[2])
            synapse[n] = float(entries[3])
    return coordinates, voltage, synapse

#-------------------------------------------------------------------------------

def gaussian(distance, sigma):
    """Generates a Gaussian curve"""
    value = (1 / (sigma * (2 * pi) ** 0.5)) * e**(-0.5 * (distance / sigma)**2)
    #value = 1 - distance/sigma
    #if value < 0: value = 0
    return value

def get_vt(t, v_0, s_0, I):
    """Calculates v(t) from initial conditions"""
    value =  I + (s_0 / (1 - synapse_decay)) * e**(-synapse_decay * t) \
            + (v_0 - I - (s_0 / (1 - synapse_decay))) * e**(-t)
    return value

def get_dvdt(t, v_actual, s_0, I):
    """Calculates the derivative of v(t) given v(t), s(0) and I"""
    value = I - v_actual + s_0 * e**(-synapse_decay * t)
    return value
    

#-------------------------------------------------------------------------------

data = data_reader.get("output-20230116000529-500N-1000sp")

neurons_number = int(data.lookup["neurons_number"])
dimension = int(data.lookup["dimension"])
signal_strength = float(data.lookup["normal_strength_e"])
signal_sigma = float(data.lookup["normal_sigma_e"])
synapse_decay = float(data.lookup["synapse_decay"])
I = float(data.lookup["input_base_before"])
signal_flattening_factor = 1 / (2 * pi) ** 0.5

assert dimension == 1

coordinates, voltage, synapse = data_importer("test_wave.csv")

print(data.data[0])
print(data.data[1])
signal = signal_flattening_factor * signal_strength \
         * abs(gaussian(data.data["coord"][1] - data.data["coord"][0],
                        signal_strength))
print(synapse[249], "+", signal, "=", synapse[249] + signal)
print(voltage[250], voltage[249])
print(data.data["time"][1])
print(get_vt(data.data["time"][1],
             voltage[249], (synapse[249] + signal)[0], I))

##plt.figure()
##plt.plot(t_values, v_analytic)
##plt.plot(t_values, v_numeric)
##plt.plot(t_values, v_very_numeric)
##plt.plot(t_values, v_REALLY_numeric)
##plt.axhline(y = 0)
##plt.axhline(y = 1)
##plt.axvline(x = 0)
##plt.xlabel("t (= x/c)")
##plt.ylabel("v(t)")
##plt.show()
