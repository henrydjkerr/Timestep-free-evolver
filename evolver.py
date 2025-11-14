"""
CUDA implementation of the 'timestep-free' method of finding firing times.

Supports variant modules for different approaches to different steps.
E.g. choice of how to initialise arrays, choice of numerical methods.
These are set in import-profiles.txt, and managed via the Control module.
Said module also holds parameter values stored in parameters.txt.

Most operations occur on the GPU, though focused optimisation has not
been performed.

Reads from the file "settings_selection.txt" as a primary input file, which then
points to various secondary input files.

TODO:
 - Implement a proper toggle for the post-fire-check culling of non-candidate
   neurons
 - Genericise the saving of variable profiles at the end of the simulation
"""

import time
stopwatch = time.time()
#Not a scientifically accurate way of benchmarking, but...
#If you need high precision to tell the GPU is an improvement over the CPU,
#then it probably wasn't worth the money. 

from numba import cuda
import numpy
import matplotlib.pyplot as plt
from math import pi, e, log

from modules.general import locator
from modules.general import Control
lookup = Control.lookup
from modules.general import generic
from modules.general import array_manager
from modules.general import save

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

#Additional counters for debugging
NN = 0
minute_count = 0

while spike_count < spikes_sought:    
    """Check if each neuron can fire, create estimated firing times"""
    Control.check.fire_check(array_manager.device_arrays)        

    """Find out the earliest latest bound on firing"""
    best_worst_time = generic.find_smallest(d_upper_bound, d_fire_flag,
                                            d_smallish_array, smallish_array)

    """Cull neurons that have no chance of firing before that bound"""
    #This is only perfect if you you're certain which neurons will fire
    #If you're not sure, it's only heuristic
##    generic.cull_larger[blocks, threads](d_lower_bound, d_fire_flag,
##                                    best_worst_time * (1 + leniency_threshold))

    """Produce accurate estimates of remaining neurons' firing times"""
    Control.solve.find_firing_time(array_manager.device_arrays)

    """Check which neuron has the fastest firing time"""
    fastest_time = generic.find_smallest(d_firing_time, d_fire_flag,
                                         d_smallish_array, smallish_array)

    """Check which neurons are fast enough to fire"""
    generic.cull_larger[blocks, threads](d_firing_time, d_fire_flag,
                                         fastest_time * (1 + leniency_threshold))

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
        break
    
    NN += 1

    """Update values"""
    Control.paramchange.update(fastest_time)
    Control.clean.postclean(array_manager.device_arrays, fastest_time)
    
    if Control.i.input_control(simulation_time):
        Control.i.input_update(array_manager.device_arrays,
                               simulation_time, fastest_time)

    spike_count += new_spikes
    if (time.time() - stopwatch) / 60 - 1 > minute_count:
        minute_count += 1
        print(minute_count, "minutes passed,")
        print("{}/{} spikes detected\n".format(spike_count, spikes_sought))


#Console output
print("Time taken:", time.time() - stopwatch)
print("Spikes found:", spike_count)
print("Saving to file...")

#Save data to file
array_manager.retrieve("coordinates")
coordinates = array_manager.host_arrays["coordinates"]
save.save_data(spike_id, spike_time, coordinates)
print("Save file closed.")

#Also save the final profile of the variables in separate files
#Need to fix this up better so it's agnostic of sLIF/ssLIF/etc.
#But for now...
try:
    array_manager.retrieve("voltage")
    save.save_profile(array_manager.host_arrays["voltage"], "v")
except:
    pass
try:
    array_manager.retrieve("wigglage")
    save.save_profile(array_manager.host_arrays["voltage"], "u")
except:
    pass
try:
    array_manager.retrieve("synapse")
    save.save_profile(array_manager.host_arrays["voltage"], "s")
except:
    pass

#Save a graph so you can see what happened
save.save_figure(coordinates)



    
