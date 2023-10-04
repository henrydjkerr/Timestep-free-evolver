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

from modules.general import locator
from modules.general import Control
lookup = Control.lookup
from modules.general import generic
from modules.general import array_manager
from modules.general import data_loader
from modules.general import save

#from modules.general.data_importer import data_importer


#-------------------------------------------------------------------------------
#Constants

neurons_number = lookup["neurons_number"]
dimension = lookup["dimension"]

threads = lookup["threads"]
blocks = lookup["blocks"]

spikes_sought = lookup["spikes_sought"]

leniency_threshold = lookup["leniency_threshold"]


#-------------------------------------------------------------------------------


smallish_array = numpy.zeros(blocks)
d_smallish_array = cuda.to_device(smallish_array)

#Sends all arrays to the GPU,
array_manager.send_all_to_device()
#and fills in their values.
Control.fill_arrays(array_manager.device_arrays)
#You want to use empty functions here if you're importing data

#Shorthanding certain arrays
d_upper_bound = array_manager.device_arrays["upper_bound"]
d_lower_bound = array_manager.device_arrays["lower_bound"]
d_fire_flag = array_manager.device_arrays["fire_flag"]
d_firing_time = array_manager.device_arrays["firing_time"]
d_firing_neurons = array_manager.device_arrays["firing_neurons"]

fire_flag = array_manager.host_arrays["fire_flag"]
firing_neurons = array_manager.host_arrays["firing_neurons"]

#Set up host-side recording for firing events
spike_count = 0
spike_id = []
spike_time = []
simulation_time = 0

while spike_count < spikes_sought:
    """Check if each neuron can fire, create estimated firing times"""
    Control.check.fire_check(array_manager.device_arrays)
        
    """Find out the earliest latest bound on firing"""
    best_worst_time = generic.find_smallest(d_upper_bound, d_fire_flag,
                                            d_smallish_array, smallish_array)

    """Cull neurons that have no chance of firing before that bound"""
    generic.cull_larger[blocks, threads](d_lower_bound, d_fire_flag,
                                    best_worst_time * (1 + leniency_threshold))

    """Produce accurate estimates of remaining neurons' firing times"""
    Control.solve.find_firing_time(array_manager.device_arrays)
    
    """Check which neuron has the fastest firing time"""
    fastest_time = generic.find_smallest(d_firing_time, d_fire_flag,
                                         d_smallish_array, smallish_array)

    """Check which neurons are fast enough to fire"""
    generic.did_fire[blocks, threads](d_fire_flag, d_firing_time, fastest_time)

    """Record new firing times"""
    array_manager.retrieve("fire_flag")
    simulation_time += fastest_time
    new_spikes = 0
    for n in range(neurons_number):
        if fire_flag[n] == 1:
            spike_id.append(n)
            spike_time.append(simulation_time)
            firing_neurons[new_spikes] = n
            new_spikes += 1
    firing_neurons[new_spikes] = neurons_number + 1
    array_manager.send("firing_neurons")

    #Quit if no new neurons can fire
    if new_spikes == 0:
        print("No new neurons fired...")
        #print(fire_flag)
        break

    """Update values"""
    generic.postclean(array_manager.device_arrays, fastest_time)

    if Control.i.input_control(simulation_time):
        Control.i.input_update(array_manager.device_arrays,
                               simulation_time, fastest_time)

    spike_count += new_spikes


#Console output
print("Time taken:", time.time() - stopwatch)
print("Spikes found:", spike_count)
print("Saving to file...")

#Save data to file
array_manager.retrieve("coordinates")
save.save_data(spike_id, spike_time, array_manager.host_arrays["coordinates"])
print("Save file closed.")

#Plot graph, if desired
#Should probably file this off
coordinates = array_manager.host_arrays["coordinates"]
plt.figure(figsize=(8, 8))#, dpi=1000)
x_axis = numpy.array([coordinates[n, 0] for n in spike_id])
y_axis = numpy.array(spike_time)
plt.scatter(x_axis, y_axis, s=0.2, c="#9569be")
plt.title("Neuron firing times")
plt.xlabel("Neuron position")
plt.ylabel("Time")
plt.margins(x=0, y=0.01)

#plt.savefig("output/figure-{}.png".format(identifier))
plt.show()   

    
