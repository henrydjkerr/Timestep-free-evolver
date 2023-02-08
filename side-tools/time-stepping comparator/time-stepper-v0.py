from numba import cuda
import numpy
from numpy.random import default_rng
import matplotlib.pyplot as plt
from math import pi, e, log
from random import random

from modules import Control
lookup = Control.lookup
from modules.data_importer import data_importer

import time
stopwatch = time.time()

#------------------------------------------------------------------------------
#Constants

neurons_number = lookup["neurons_number"]
dimension = lookup["dimension"]

threads = lookup["threads"]
blocks = lookup["blocks"]

spikes_sought = lookup["spikes_sought"]

leniency_threshold = lookup["leniency_threshold"]


#-------------------------------------------------------------------------------

def gaussian(x, sigma):
    return (1/sigma) * ((1/(2*pi))**0.5) * (e**(-0.5 * (x/sigma)**2))
    

#-------------------------------------------------------------------------------


#Set up variable arrays
#voltage = default_rng().random(neurons_number)  #Should be v_r if nothing else
voltage = numpy.zeros(neurons_number)
synapse = numpy.zeros(neurons_number)
coordinates = numpy.zeros((neurons_number, dimension))
I = numpy.zeros(neurons_number)

coordinates, voltage, synapse = data_importer("test_wave.csv")

for i in range(neurons_number):
    I[i] = lookup["input_base_before"]

#Set up host-side recording for firing events
spike_count = 0
spike_id = []
spike_time = []
simulation_time = 0

time_step = 0.0001

synapse_decay = lookup["synapse_decay"]
v_th = lookup["v_th"]
v_r = lookup["v_r"]
signal_strength = lookup["normal_strength_e"]
signal_sigma = lookup["normal_sigma_e"]
dx = lookup["dx"]
print("beta = {}, v_th = {}, v_r = {}, strength = {}, sigma = {}, dx = {}".format(
    synapse_decay, v_th, v_r, signal_strength, signal_sigma, dx))

print("I[0]:", I[0])

while simulation_time < 2:
    simulation_time += time_step
    for n in range(neurons_number):
        voltage[n] += time_step * (I[n] - voltage[n] + synapse[n])
        synapse[n] -= time_step * synapse_decay * synapse[n]
    for n in range(neurons_number):
        if voltage[n] >= v_th:
            voltage[n] = v_r
            spike_count += 1
            spike_id.append(n)
            spike_time.append(simulation_time)
            for m in range(neurons_number):
                if n != m:
                    #distance = dx * abs(n - m)
                    distance = Control.d.get_distance_between(coordinates, n, m)
                    strength = synapse_decay * dx * signal_strength \
                               * gaussian(distance, signal_sigma)
                    synapse[m] += strength
                    #synapse[m] += Control.c.connection_weight(distance)




#--------------------

#Console output
print("Time taken:", time.time() - stopwatch)
print("Spikes found:", spike_count)
print("Saving to file...")

#Save data to file
timestamp = time.strftime("%Y%m%d%H%M%S")
identifier = "{}-{}N-{}sp".format(timestamp, neurons_number, spikes_sought)
filename = "output/output-{}.csv".format(identifier)

outfile = open(filename, "w")
outfile.write("PARAMETERS:\n")
for key in lookup:
    line = "PAR,{},{}\n".format(str(key), str(lookup[key]))
    outfile.write(line)
outfile.write("MODULES:\n")
for key in Control.names:
    line = "MOD,{},{}\n".format(str(key), str(Control.names[key]))
    outfile.write(line)
outfile.write("DATA:\n")
#d_coordinates.copy_to_host(coordinates)
for k in range(len(spike_id)):
    n = spike_id[k]
    line = str(n) + "," + str(spike_time[k])
    for d in range(dimension):
        line += "," + str(coordinates[n, d])
    outfile.write(line + "\n")
outfile.close()
print("Save file closed.")


#Plot figure
plt.figure(figsize=(8, 8))#, dpi=1000)
x_axis = numpy.array([coordinates[n, 0] for n in spike_id])
y_axis = numpy.array(spike_time)
plt.scatter(x_axis, y_axis, s=0.2, c="#a0a070")
plt.title("Euler version")
plt.xlabel("Neuron id/position")
plt.ylabel("Time")
plt.margins(x=0, y=0.01)

plt.savefig("output/figure-{}.png".format(identifier))
plt.show()   
