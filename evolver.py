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
import matplotlib.pyplot as plt
from math import pi, e, log

from modules.general import locator
from modules.general import Control
lookup = Control.lookup
from modules.general import generic
from modules.general import array_manager
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

#print_time = False
NN = 0
minute_count = 0

while spike_count < spikes_sought:
    #if spike_count > 1500: print_time = True
    #if spike_count > 2200: print_time = False

##    if NN < 4:
##        print("\n")
##        print(NN)
##        array_manager.retrieve_all_from_device()
##        print("I", array_manager.host_arrays["input_strength"][995:1005])
##        print("v", array_manager.host_arrays["voltage"][995:1005])
##        print("u", array_manager.host_arrays["wigglage"][995:1005])
##        print("s", array_manager.host_arrays["synapse"][995:1005])
##        print("fire_flag A", array_manager.host_arrays["fire_flag"][995:1005])
##        
##        print("I", array_manager.host_arrays["input_strength"][705:715])
##        print("v", array_manager.host_arrays["voltage"][705:715])
##        print("u", array_manager.host_arrays["wigglage"][705:715])
##        print("s", array_manager.host_arrays["synapse"][705:715])
##        print("fire_flag A", array_manager.host_arrays["fire_flag"][705:715])

    
    """Check if each neuron can fire, create estimated firing times"""
    Control.check.fire_check(array_manager.device_arrays)        

##    if NN < 4:
##        array_manager.retrieve_all_from_device()
##        print("fire_flag B", array_manager.host_arrays["fire_flag"][995:1005])
##        print("lower_bound", array_manager.host_arrays["lower_bound"][995:1005])
##        print("upper_bound", array_manager.host_arrays["upper_bound"][995:1005])
##        
##        print("fire_flag B", array_manager.host_arrays["fire_flag"][705:715])
##        print("lower_bound", array_manager.host_arrays["lower_bound"][705:715])
##        print("upper_bound", array_manager.host_arrays["upper_bound"][705:715])

    #array_manager.retrieve("upper_bound")
    """Find out the earliest latest bound on firing"""
    best_worst_time = generic.find_smallest(d_upper_bound, d_fire_flag,
                                            d_smallish_array, smallish_array)
##    if NN == 0:
##        print(best_worst_time)
    """Cull neurons that have no chance of firing before that bound"""
    #This is only perfect if you you're certain which neurons will fire
    #If you're not sure, it's only heuristic
    #generic.cull_larger[blocks, threads](d_lower_bound, d_fire_flag,
    #                                best_worst_time * (1 + leniency_threshold))

    """Produce accurate estimates of remaining neurons' firing times"""
    Control.solve.find_firing_time(array_manager.device_arrays)

##    if NN < 4:
##        array_manager.retrieve_all_from_device()
##        print("fire_flag C", array_manager.host_arrays["fire_flag"][995:1005])
##        print("Firing times", array_manager.host_arrays["firing_time"][995:1005])
##        print("lower_bound", array_manager.host_arrays["lower_bound"][995:1005])
##        print("upper_bound", array_manager.host_arrays["upper_bound"][995:1005])
##
##        print("fire_flag C", array_manager.host_arrays["fire_flag"][705:715])
##        print("Firing times", array_manager.host_arrays["firing_time"][705:715])
##        print("lower_bound", array_manager.host_arrays["lower_bound"][705:715])
##        print("upper_bound", array_manager.host_arrays["upper_bound"][705:715])

    """Check which neuron has the fastest firing time"""
    fastest_time = generic.find_smallest(d_firing_time, d_fire_flag,
                                         d_smallish_array, smallish_array)

    #print(fastest_time)
    #array_manager.retrieve("firing_time")
    #print("Firing times", array_manager.host_arrays["firing_time"][1830:1840])
    
    """Check which neurons are fast enough to fire"""
    generic.cull_larger[blocks, threads](d_firing_time, d_fire_flag,
                                         fastest_time * (1 + leniency_threshold))

    #array_manager.retrieve("fire_flag")
    #print("fire_flag D", array_manager.host_arrays["fire_flag"][1830:1840])

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
    
    NN += 1

##    if print_time:
##        array_manager.retrieve_all_from_device()
##        print("\n")
##        print(spike_count)
##        #print(fastest_time)
##        #print(firing_neurons[0:5])
##        #print(array_manager.host_arrays["fire_flag"][1836:1840])
##        #print("v", array_manager.host_arrays["voltage"][1836:1840])
##        #print("u", array_manager.host_arrays["wigglage"][1836:1840])
##        print("s", array_manager.host_arrays["synapse"][1836:1840])
##        #temp_s = array_manager.host_arrays["synapse"][1838]
##        #temp_x = array_manager.host_arrays["coordinates"][1838]
##        #print("s:", temp_s, "x:", temp_x)
##        #print(temp_s * e**(-6 * fastest_time))
##        #print("Firing:", firing_neurons[0])
##        #print(abs(temp_x - (firing_neurons[0] - 1000) * 0.02))
##        #print("I", array_manager.host_arrays["input_strength"][1836:1840])
##        #print("x", array_manager.host_arrays["coordinates"][1836:1840])

    """Update values"""
    Control.paramchange.update(fastest_time)
    Control.clean.postclean(array_manager.device_arrays, fastest_time)
    
##    if print_time:
##        array_manager.retrieve_all_from_device()
##        #print(spike_count)
##        #print(fastest_time)
##        #print(firing_neurons[0:5])
##        print(array_manager.host_arrays["fire_flag"][1836:1840])
##        #print("v", array_manager.host_arrays["voltage"][1836:1840])
##        #print("u", array_manager.host_arrays["wigglage"][1836:1840])
##        print("s", array_manager.host_arrays["synapse"][1836:1840])
##        #print("I", array_manager.host_arrays["input_strength"][1836:1840])
##        #print("x", array_manager.host_arrays["coordinates"][1836:1840])

    if Control.i.input_control(simulation_time):
        #print("Hi", lookup["R"])
        Control.i.input_update(array_manager.device_arrays,
                               simulation_time, fastest_time)

    spike_count += new_spikes
    if (time.time() - stopwatch) / 60 - 1 > minute_count:
        minute_count += 1
        print(minute_count, "minutes passed,")
        print("{}/{} spikes detected\n".format(spike_count, spikes_sought))
    
    #print(spike_count)


#Console output
print("Time taken:", time.time() - stopwatch)
print("Spikes found:", spike_count)
print("Saving to file...")

#Save data to file
array_manager.retrieve("coordinates")
save.save_data(spike_id, spike_time, array_manager.host_arrays["coordinates"])
print("Save file closed.")

#Also save the final profile of the variables in separate files
#Need to fix this up better so it's agnostic of sLIF/ssLIF/etc.
#But for now...
array_manager.retrieve("voltage")
array_manager.retrieve("wigglage")
array_manager.retrieve("synapse")
save.save_profile(array_manager.host_arrays["voltage"], "v")
save.save_profile(array_manager.host_arrays["voltage"], "u")
save.save_profile(array_manager.host_arrays["voltage"], "s")

#Plot graph, if desired
#Should probably file this off
coordinates = array_manager.host_arrays["coordinates"]
#plt.figure(figsize=(4, 4), dpi=200)#, dpi=1000)

#plt.figure(figsize=(2, 4), dpi=2000)
#plt.figure(figsize=(10, 24), dpi=400)
plt.figure()
#plt.xlim(-2, 2)
#plt.ylim(0, 6)

x_axis = numpy.array([coordinates[n, 0] for n in spike_id])
y_axis = numpy.array(spike_time)
#plt.scatter(x_axis, y_axis, s=0.2, c="#9569be")

plt.scatter(x_axis, y_axis, s=1, c="#000000", marker="8")
#plt.scatter(x_axis, y_axis, s=32, c="#000000", marker="o")
plt.title("Neuron firing times")
plt.xlabel("Neuron position")
plt.ylabel("Time")
plt.margins(x=0, y=0.01)

#plt.savefig("output/figure-{}.png".format(identifier))
#plt.savefig("sim_R={}.png".format(str(lookup["C"])[:4]))
plt.savefig("sample.png")
plt.savefig("sim-{}.png".format(time.strftime("%Y%m%d-%H%M%S")))
plt.show()


    
