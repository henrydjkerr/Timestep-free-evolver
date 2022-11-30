"""
CUDA implementation of the 'timestep-free' method of finding firing times.

Supports variant modules for different approaches to different steps.
E.g. choice of how to initialise arrays, choice of numerical methods.
These are set in import-profiles.txt, and managed via the Control module.
Said module also holds parameter values stored in parameters.txt.

Most operations occur on the GPU, though focused optimisation has not
been performed.

There's a singularity in the solutions when the synaptic decay rate = 1.
As such it's necessary to handle the case close to 1 with separate code.
This code has not yet been written, but could be switched using the
architecture mentioned for selecting between modules.

TODO:
 - Test in 2D
 - Figure out what unit testing looks like on a GPU
 - Look for a nice way to suppress all the under-occupancy warnings
"""

import time
stopwatch = time.time()
#Not a scientifically accurate way of benchmarking, but...
#If you need high precision to tell the GPU is an improvement over the CPU,
#then it probably wasn't worth the money. 

from numba import cuda
import numpy
from numpy.random import default_rng
import matplotlib.pyplot as plt
from math import pi, e, log
from random import random

from modules import Control
lookup = Control.lookup
from modules import generic

from modules.data_importer import data_importer


#-------------------------------------------------------------------------------
#Constants

neurons_number = lookup["neurons_number"]
dimension = lookup["dimension"]

threads = lookup["threads"]
blocks = lookup["blocks"]

spikes_sought = lookup["spikes_sought"]

leniency_threshold = lookup["leniency_threshold"]


#-------------------------------------------------------------------------------


#Set up variable arrays
voltage = default_rng().random(neurons_number)  #Should be v_r if nothing else
#voltage = numpy.zeros(neurons_number)
synapse = numpy.zeros(neurons_number)
lower_bound = numpy.zeros(neurons_number)
upper_bound = numpy.zeros(neurons_number)
firing_time = numpy.zeros(neurons_number)
fire_flag = numpy.zeros(neurons_number, dtype = numpy.dtype(int))
coordinates = numpy.zeros((neurons_number, dimension))
input_strength = numpy.zeros(neurons_number)
firing_neurons = numpy.zeros(neurons_number, dtype = numpy.dtype(int))

smallish_array = numpy.zeros(blocks)
d_smallish_array = cuda.to_device(smallish_array)

#This wouls be where to put any data importing system
#data_importer("TW5_500N_doubled.csv", coordinates, voltage, synapse)
#print(synapse)

d_voltage = cuda.to_device(voltage)
d_synapse = cuda.to_device(synapse)
d_lower_bound = cuda.to_device(lower_bound)
d_upper_bound = cuda.to_device(upper_bound)
d_firing_time = cuda.to_device(firing_time)
d_fire_flag = cuda.to_device(fire_flag)
d_coordinates = cuda.to_device(coordinates)
d_input_strength = cuda.to_device(input_strength)
d_firing_neurons = cuda.to_device(firing_neurons)

#Set up host-side recording for firing events
spike_count = 0
spike_id = []
spike_time = []
simulation_time = 0

#Initialise voltage, input_strength via CUDA
Control.xi.coordinate_init[blocks, threads](d_coordinates)
Control.vi.voltage_init[blocks, threads](d_voltage, d_coordinates)
Control.ii.input_init[blocks, threads](d_input_strength, d_coordinates)


while spike_count < spikes_sought:
    """Check if each neuron can fire, create estimated firing times"""
    Control.check.fire_check[blocks, threads](d_voltage, d_synapse,
                                              d_input_strength, d_fire_flag,
                                              d_lower_bound, d_upper_bound,
                                              d_firing_time)
        
    """Find out the earliest latest bound on firing"""    
    best_worst_time = generic.find_smallest(d_upper_bound, d_fire_flag,
                                            d_smallish_array, smallish_array)

    """Cull neurons that have no chance of firing before that bound"""
    generic.cull_larger[blocks, threads](d_lower_bound, d_fire_flag,
                                     best_worst_time * (1 + leniency_threshold))

    """Produce accurate estimates of remaining neurons' firing times"""
    Control.solve.find_firing_time[blocks, threads](d_voltage, d_synapse,
                                                    d_input_strength,
                                                    d_fire_flag, d_lower_bound,
                                                    d_upper_bound,
                                                    d_firing_time)
    
    """Check which neuron has the fastest firing time"""
    fastest_time = generic.find_smallest(d_firing_time, d_fire_flag,
                                         d_smallish_array, smallish_array)

    """Check which neurons are fast enough to fire"""
    generic.did_fire[blocks, threads](d_fire_flag, d_firing_time, fastest_time)

    d_fire_flag.copy_to_host(fire_flag)
    """Record new firing times"""
    simulation_time += fastest_time
    new_spikes = 0
    for n in range(neurons_number):
        if fire_flag[n] == 1:
            spike_id.append(n)
            spike_time.append(simulation_time)
            firing_neurons[new_spikes] = n
            new_spikes += 1
    firing_neurons[new_spikes] = neurons_number + 1
    d_firing_neurons = cuda.to_device(firing_neurons)

    #Quit if no new neurons can fire
    if new_spikes == 0:
        print("No new neurons fired...")
        #print(fire_flag)
        break

    """Update values"""
    generic.postclean[blocks, threads](d_voltage, d_synapse, d_input_strength,
                                       d_fire_flag, d_lower_bound,
                                       d_upper_bound, d_firing_time,
                                       d_coordinates, d_firing_neurons,
                                       fastest_time)

    if Control.i.input_control(simulation_time):
        Control.i.input_update[blocks, threads](d_input_strength,
                                                simulation_time, fastest_time)

    spike_count += new_spikes


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
d_coordinates.copy_to_host(coordinates)
for k in range(len(spike_id)):
    n = spike_id[k]
    line = str(n) + "," + str(spike_time[k])
    for d in range(dimension):
        line += "," + str(coordinates[n, d])
    outfile.write(line + "\n")
outfile.close()
print("Save file closed.")

#Plot graph, if desired
plt.figure(figsize=(8, 8))#, dpi=1000)
x_axis = numpy.array([coordinates[n, 0] for n in spike_id])
y_axis = numpy.array(spike_time)
plt.scatter(x_axis, y_axis, s=0.2, c="#a0a070")
plt.title("Neuron firing times")
plt.xlabel("Neuron id/position")
plt.ylabel("Time")
plt.margins(x=0, y=0.01)

plt.savefig("output/figure-{}.png".format(identifier))
plt.show()   

    
