This document describes the variety of modules that can be swapped between to manage certain functionality.

In general, modules that are to execute device-side have host-side wrappers.  The reason for this is that Numba does not support **kwargs or dictionary arguments with the device-side functions, which means that if a piece of code wants to call a device-side function, it has to know the exact arguments it wants.  I considered this undesirable, because it requires explicitly numbering and naming all the arrays needed in the main program file.  Different models need different number of variables, and I don't want you to have to swap or alter `evolver.py` for each model you use.  So instead, you call a wrapper function that takes a dictionary of all device-side arrays as an argument, then picks the ones it wants to call the device-side function with.

This has the bonus advantage of removing the repeated `[blocks, threads]` notation from the function calls, which makes things less cramped in my opinion.

Modules are grouped by the name they are bound to in existing modules.  At present, some of these are still referenced in non-swappable code (so the program is not fully generic yet), meaning you should expect to conform to these names even if you create an all-new set of swap-in modules.



# Modules `fire_check`
These modules perform the task of determining whether a given neuron will attain its firing condition without further input.  If the neuron will fire, the associated position in the flag array is set to true, and upper and lower bounds on the firing time are calculated. Otherwise, the flag is set to false.


## fire_check_maybe1.py
Wrapper module for use with the synaptic LIF (sLIF) model.  The equations have a singularity when the paramater `synapse_decay = 1`, so it's necessary to have separate code to avoid a division-by-zero error in this special case.

Imports and binds the `fire_check` function from `fire_check_is1` if `synapse_decay` is in the interval (0.999, 1.001), or from `fire_check_not1` otherwise.


## fire_check_not1.py
For use with the sLIF model in the case `synapse_decay != 1`.

### Functions:
`fire_check(dict arrays)`: take a dictionary of device-side arrays, select the ones required for running `fire_check_device`, then call that function with the appropriate arguments.

The arrays required are named by the dictionary keys: voltage, synapse, input_strength, fire_flag, lower_bound, upper_bound, firing_time.

`fire_check_device(voltage, synapse, input_strength, fire_flag, lower_bound, upper_bound, firing_time)`: for the given neuron index, determine whether there exists any time t >= 0 at which v(t) = v_th, taking the present model time to be t = 0.  If such a time exists, estimate correct bounds, and set the corresponding position in the `fire_flag` array to `True` and calculate bounds on the firing time, storing them in the arrays `lower_bound` and `upper_bound`.  Otherwise, set the flag to `False`.

Additionally, the firing_time array is used to store a converging initial condition for the Newton-Raphson root-finding method.

TODO: the full determination of the correct bounds requires working through a large number of cases.  This would better be explained in a separate file.


## fire_check_is1.py
For use in the sLIF model in the case `synapse_decay == 1`, or is sufficiently close to prompt concerns that dividing by `synapse_decay - 1` may cause precision errors.

The description is otherwise identical to `fire_check_not1` as far as this document is concerned. Contains the functions `fire_check` and `fire_check_device`.

# Modules `root_finder`

## newton_raphson.py
A conventional implementation of the Newton-Raphson algorithm.  For simplicity's sake, it is hard-coded to solve the sLIF equations, determining an accurate and precise value of t that satisfies v(t) = v_th.

### Functions
`find_firing_time(dict arrays)`: wrapper function to select the required device-side arrays and use them to call the device-side function `find_firing_time_device`.

The arrays required are named by the dictionary keys: voltage, synapse, input_strength, fire_flag, lower_bound, upper_bound, firing_time.

`find_firing_time_device(voltage, synapse, input_strength, fire_flag, lower_bound, upper_bound, firing_time)`: for the given neuron index `n`, if `fire_flag[n] == True`, execute the Newton-Raphson scheme to numerically find a solution to v(t) = v_th for t > 0.  Use the pre-existing value `firing_time[n]` as the initial value, as this has previously been selected to assure convergence.


