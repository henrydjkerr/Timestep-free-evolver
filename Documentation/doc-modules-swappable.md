This document describes the variety of modules that can be swapped between to manage certain functionality.

In general, modules that are to execute device-side have host-side wrappers.  The reason for this is that Numba does not support **kwargs or dictionary arguments with the device-side functions, which means that if a piece of code wants to call a device-side function, it has to know the exact arguments it wants.  I considered this undesirable, because it requires explicitly numbering and naming all the arrays needed in the main program file.  Different models need different number of variables, and I don't want you to have to swap or alter `evolver.py` for each model you use.  So instead, you call a wrapper function that takes a dictionary of all device-side arrays as an argument, then picks the ones it wants to call the device-side function with.

This has the bonus advantage of removing the repeated `[blocks, threads]` notation from the function calls, which makes things less cramped in my opinion.

Modules are grouped by the name they are bound to in existing modules.  At present, some of these are still referenced in non-swappable code (so the program is not fully generic yet), meaning you should expect to conform to these names even if you create an all-new set of swap-in modules.



# Modules `fire_check`
These modules perform the task of determining whether a given neuron will attain its firing condition without further input.  If the neuron will fire, the associated position in the flag array is set to true, and upper and lower bounds on the firing time are calculated. Otherwise, the flag is set to false.


## fire_check_sLIF.py
For use with the synaptic LIF (sLIF) model.  

The desired outcome is to find an interval in which the first time $t > 0$ at which $v(t) \geq v_{th}$ is achieved, and to find a point from which the Newton-Raphson root-finding method will converge to that value of $t$.  This is achieved by characterising the form of $v$ by its initial value, long-term limit and any extreme or inflection points to produce a comprehensive set of cases.

The equation for $v$ is
$$v(t) = I + \frac{s_0}{1 - \beta}e^{-\beta t} + \bigg(v_0 - I - \frac{s_0}{1 - \beta}\bigg)e^{-t},$$
where $v_0 = v(0)$, $s_0 = s(0)$ and $\beta =$ `synapse_decay`.  This has a singularity when $\beta = 1$, instead giving equation
$$v(t) = I + s_0 t e^{-t} + (v_0 - I)e^{-t},$$
but the equation has the same shape and characteristics we need.  Additionally, in certain cases the extreme points occur at a large negative value of $t$; to avoid overflow errors when calculating the exponentials, we restrict calculation of $v(t)$ to $[0, \infty)$.

### Functions:
`fire_check(dict arrays)`: take a dictionary of device-side arrays, select the ones required for running `fire_check_device`, then call that function with the appropriate arguments.

The arrays required are named by the dictionary keys: voltage, synapse, input_strength, fire_flag, lower_bound, upper_bound, firing_time.

`fire_check_device(voltage, synapse, input_strength, fire_flag, lower_bound, upper_bound, firing_time)`: for the given neuron index, determine whether there exists any time $t \geq 0$ at which $v(t) =$ `v_th`, taking the present model time to be $t = 0.  If such a time exists, set the corresponding position in the fire_flag array to `True` and calculate bounds on the firing time, storing them in the arrays lower_bound and upper_bound.  Otherwise, set the flag to `False`.  Additionally, set the value of the firing_time array to an initial $t$ value from which the Newton-Raphson root-finding method will converge to the firing time.

TODO: supply auxiliary document running through the case-selection rationale.



## fire_check_ssLIF.py
For use with the subthreshold-synaptic LIF (sLIF) model in the case where the system of governing ODEs has complex eigenvalues leading to oscillatory terms.  

The desired outcome is to find an interval in which the first time $t > 0$ at which $v(t) \geq v_{th}$ is achieved.  However, demonstrating the existence of said $t$ is no longer simple, so we use an interval in which said $t$ will be present if it exists.  This is achieved by considering the upper envelope of the oscillation of $v$, and treating it in a similar manner to $v$ in `fire_check_sLIF`, producing a number of cases.

The equation for $v$ is overlong when expressed in basic constants, so it is instead simplified to
$$v(t) = A e^{-p t} \cos(|q|t + \theta) + Be^{-\beta t} + K,$$
where $p > 0$, $|q| > 0$.  The upper envelope of oscillation is
$$\tilde{v}(t) = |A| e^{-p t} + Be^{-\beta t} + K,$$
which is of comparable form to the $v$ of the sLIF neuron.  By considering the times at which $\tilde{v} \geq v_{th}$ and the period of $v$'s oscillation, we can construct an interval that must contain any existing firing time.  As with the sLIF, in certain cases the extreme points of $\tilde{v}$ occur at a large negative value of $t$; to avoid overflow errors when calculating the exponentials, we restrict calculation of $v(t)$ to $[0, \infty)$.

As the Newton-Raphson method is not a good fit for solving this model's equation, a different numerical method is used which does not require initial conditions other than the interval.

### Functions:
`fire_check(dict arrays)`: take a dictionary of device-side arrays, select the ones required for running `fire_check_device`, then call that function with the appropriate arguments.

The arrays required are named by the dictionary keys: voltage, synapse, wigglage, input_strength, fire_flag, lower_bound, upper_bound.

TODO: think up a variable name that's more publishable than "wigglage"

`fire_check_device(voltage, synapse, wigglage, input_strength, fire_flag, lower_bound, upper_bound)`: for the given neuron index, determine whether there exists any time $t \geq 0$ at which $v(t) =$ `v_th`, taking the present model time to be $t = 0.  If such a time exists, set the corresponding position in the `fire_flag` array to `True` and calculate bounds on the firing time, storing them in the arrays `lower_bound` and `upper_bound`.  Otherwise, set the flag to `False`.  Additionally, set the value of the `firing_time` array to an initial $t$ value from which the Newton-Raphson root-finding method will converge to the firing time.

TODO: supply auxiliary document running through the case-selection rationale.



## newton_raphson.py
A conventional implementation of the Newton-Raphson algorithm.  For simplicity's sake, it is hard-coded to solve the sLIF equations, determining an accurate and precise value of t that satisfies v(t) = v_th.

### Functions
`find_firing_time(dict arrays)`: wrapper function to select the required device-side arrays and use them to call the device-side function `find_firing_time_device`.

The arrays required are named by the dictionary keys: voltage, synapse, input_strength, fire_flag, lower_bound, upper_bound, firing_time.

`find_firing_time_device(voltage, synapse, input_strength, fire_flag, lower_bound, upper_bound, firing_time)`: for the given neuron index `n`, if `fire_flag[n] == True`, execute the Newton-Raphson scheme to numerically find a solution to $v(t) = v_{th}$ for $t > 0$ and store it in the array `firing_time[n]`.  Use the pre-existing value `firing_time[n]` as the initial value, as this has previously been selected to assure convergence.


## newton_raphson_ssLIF.py
A modified Newton-Raphson scheme for robustly finding roots of arbitrary twice-differentiable functions on a given interval. This example is hard-coded to the ssLIF equations, determining an accurate and precise value of t that satisfies v(t) = v_th, if such a value exists. If no such value exists, it reports as such by setting the corresponding value in the `fire_flag` array to `False`.

In brief, rather than extrapolating the derivative of the current point each iteration, this method uses upper bounds on the first and second derivatives to determine a steeper line that is guaranteed to undershoot any roots.  By initialising the algorithm on the left bound of the interval, it marches across monotonically until it either converges or passes the right bound and terminates with no root found.

### Functions
`find_firing_time(dict arrays)`: wrapper function to select the required device-side arrays and use them to call the device-side function `find_firing_time_device`.

The arrays required are named by the dictionary keys: voltage, synapse, wigglage, input_strength, fire_flag, lower_bound, upper_bound, firing_time.

`find_firing_time_device(voltage, synapse, wigglage, input_strength, fire_flag, lower_bound, upper_bound, firing_time)`: for the given neuron index `n`, if `fire_flag[n] == True`, execute the modifiied Newton-Raphson scheme to numerically find a solution to $v(t) = v_{th}$ for $t > 0$ and store it in the array `firing_time[n]`.  If no such value is found between `lower_bound[n]` and `upper_bound[n]`, set `fire_flag[n] = False` instead.
