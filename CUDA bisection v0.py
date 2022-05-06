"""
Attempt to rewrite timestep-free evolver without using Objects.
Starting with the bisection method for simplicity.
Also ignoring the beta ~= 1 case.
"""

import time
stopwatch = time.time()

from numba import cuda
import numpy
from numpy.random import default_rng
import matplotlib.pyplot as plt
from math import pi, e, log
from random import random

#-------------------------------------------------------------------------------
#Space for cuda kernels
@cuda.jit()
def voltage_init(voltage_d):
    #Initialises voltage
    n = cuda.grid(1)
    if n < neurons_number:
        voltage_d[n] = v_r + (v_th - v_r) * voltage_d[n]
        #voltage[n] = 0

@cuda.jit()
def input_init(input_strength_d):
    #Initialises input_strength
    n = cuda.grid(1)
    if n < neurons_number:
        position = dx * (n - (neurons_number/2) + 0.5)
        input_strength_d[n] = bellcurve_cuda(position, sigma)

#Main loop kernels

##@cuda.jit()
##def preclean(fire_flag, lower_bound, upper_bound):
##    n = cuda.grid(1)
##    if n < neurons_number:
##        fire_flag[n] = 1
##        lower_bound[n] = 0
##        upper_bound[n] = 0

@cuda.jit()
def fire_check(voltage_d, synapse_d, input_strength_d,
               fire_flag_d, lower_bound_d, upper_bound_d):
    n = cuda.grid(1)
    if n < neurons_number:
        fire_flag_d[n] = 1
        v_0 = voltage_d[n]
        s_0 = synapse_d[n]
        I = input_strength_d[n]
        condition = 0
        
        lower_bound_type = 0
        upper_bound_type = 0
        
        if v_0 > v_th:
            #"Error"-catching case
            lower_bound_d[n] = 0
            upper_bound_d[n] = 0
            lower_bound_type = -1
            upper_bound_type = -1
        elif (I < v_th) and (s_0 <= 0):
            #Easy no-firing case
            fire_flag_d[n] = 0
        else:
            #Assuming synapse decay rate != 1
            #Also assuming synapse decay rate != 0
            if s_0 == 0:
                extreme_exists = False
                #condition = 1
            else:
                #Only know it's not either trivial case
                e_temp = (1 - synapse_decay)/synapse_decay \
                         * (I - v_0 + s_0/(1 - synapse_decay))
                e_gradient = e_temp / s_0
                e_inflect = e_temp / (s_0 * synapse_decay)
                if e_gradient <= 0:
                    extreme_exists = False
                    #condition = 2
                else:
                    extreme_time = 1/(1 - synapse_decay) \
                                   * cuda.libdevice.log(e_gradient)
                    if extreme_time >= 0:
                        extreme_exists = True
                        #condition = 3
                    else:
                        extreme_exists = False
                        #condition = 4
            if extreme_exists:
                #Still need to check solution exists!
                extreme_v = get_vt_cuda(extreme_time, v_0, s_0, I)
                if extreme_v == v_th:
                    lower_bound_d[n] = extreme_time
                    upper_bound_d[n] = extreme_time
                    fire_flag_d[n] = 1
                    #condition += 0.1
                elif extreme_v > v_th:
                    lower_bound_type = 1
                    upper_bound_d[n] = extreme_time
                    upper_bound_type = -1
                    fire_flag_d[n] = 1
                    #condition += 0.2
                elif (extreme_v < v_0) and (I > v_th):
                    lower_bound_type = 2
                    upper_bound_type = 1
                    fire_flag_d[n] = 1
                    #condition += 0.3
                else:
                    fire_flag_d[n] = 0
                    #condition += 0.4
            elif (I > v_th):
                #No extreme but long-term limit takes you over
                lower_bound_type = 1
                upper_bound_type = 1
                fire_flag_d[n] = 1
                #condition += 0.5
            else:
                #No extreme, and long-term limit is below threshold
                fire_flag_d[n] = 0
                #condition += 0.6
                
            if lower_bound_type == 1:
                lower_bound_d[n] = (v_th - v_0) / get_dvdt_cuda(0, v_0, s_0, I)
            elif lower_bound_type == 2:
                inflect_time = 1/(1 - synapse_decay) \
                               * cuda.libdevice.log(e_inflect)
                inflect_v = get_vt_cuda(inflect_time, v_0, s_0, I)
                lower_bound_d[n] = inflect_time + (v_th - inflect_v) \
                                   / get_dvdt_cuda(inflect_time,
                                                   inflect_v, s_0, I)
            if upper_bound_type == 1:
                m = 0
                while True:
                    test_t = 2**m
                    if test_t > lower_bound_d[n]:                    
                        temp_v = get_vt_cuda(test_t, v_0, s_0, I)
                        if temp_v > v_th:
                            upper_bound_d[n] = test_t
                            break
                        #Could try to also improve the lower bound if I'm lucky
                        #But I don't record the value there actively
                    m += 1
                    #No failure condition?
            #if n == 50: print(lower_bound[n], upper_bound[n])
            #Something feels ugly here but eh.

@cuda.jit()
def bisection(voltage_d, synapse_d, input_strength_d,
              fire_flag_d, lower_bound_d, upper_bound_d, firing_time_d):
    n = cuda.grid(1)
    if n < neurons_number:
        if fire_flag_d[n]:
            t_a = lower_bound_d[n]
            t_b = upper_bound_d[n]
            while True:
                t_current = 0.5*(t_b + t_a)
                if t_b - t_a < error_bound:
                    firing_time_d[n] = t_current
                    break
                test_v = get_vt_cuda(t_current, voltage_d[n], synapse_d[n],
                               input_strength_d[n])
                if test_v == v_th:
                    firing_time_d[n] = t_current
                    break
                elif test_v > v_th:
                    t_b = t_current
                else:
                    t_a = t_current

@cuda.jit()
def did_fire(fire_flag_d, firing_time_d, fastest_time_d):
    n = cuda.grid(1)
    if n < neurons_number:
        if fire_flag_d[n]:
            #Will these conditionals cause more trouble than they're worth?
            if firing_time_d[n] > fastest_time_d * (1 + leniency_threshold):
                fire_flag_d[n] = 0

@cuda.jit()
def postclean(voltage_d, synapse_d, input_strength_d, 
              fire_flag_d, lower_bound_d, upper_bound_d,
              firing_time_d, fastest_time_d):
    n = cuda.grid(1)
    if n < neurons_number:
        #Calculate updated voltage
        if fire_flag_d[n] == 1:   #If firing
            voltage_d[n] = v_r
        else:                   #If not firing
            voltage_d[n] = get_vt_cuda(fastest_time_d, voltage_d[n],
                                       synapse_d[n], input_strength_d[n])
        #Update synapse value wrt time evolution
        synapse_d[n] *= e**(-synapse_decay * fastest_time_d)
        #Update synapse value wrt other neurons firing
        m = 0
        while m < neurons_number:
            if n != m:
                if fire_flag_d[m] == 1:
                    distance = float(abs(m - n)) * dx
                    synapse_d[n] += connection_weight_cuda(distance)
            m += 1

        lower_bound_d[n] = 0
        upper_bound_d[n] = 0
        firing_time_d[n] = 0

#-------------------------------------------------------------------------------
#Space for cuda device functions
@cuda.jit(device = True)
def bellcurve_cuda(x, sigma):
    #Do I need to load pi, e on to the device specifically?
    value = 10 * (1 / (sigma * (2 * pi) ** 0.5)) * e**(-0.5 * (x/sigma)**2)
    #value = 10 * (1 / (sigma * 2.507)) * 2.72**(-0.5 * (x/sigma)**2)
    return value

@cuda.jit(device = True)
def connection_weight_cuda(diff):
    #Need c_sigma_1, c_sigma_2, connection_strength on device?
    signal = bellcurve_cuda(diff, c_sigma_1) - bellcurve_cuda(diff, c_sigma_2)
    return connection_strength * signal
    #return 0.1 * signal

@cuda.jit(device = True)
def get_vt_cuda(t, v_0, s_0, I):
    value =  I + (s_0 / (1 - synapse_decay)) * e**(-synapse_decay * t) \
            + (v_0 - I - (s_0 / (1 - synapse_decay))) * e**(-t)
    #value =  I + (s_0 / (1 - 2)) * 2.72**(-2 * t) \
    #        + (v_0 - I - (s_0 / (1 - 2))) * 2.72**(-t)
    return value

@cuda.jit(device = True)
def get_dvdt_cuda(t, v_actual, s_0, I):
    #If I only calculate dvdt when I already know v(t) then this is easier
    value = I - v_actual + s_0 * e**(-synapse_decay * t)
    #value = I - v_actual + s_0 * 2.72**(-2 * t)
    return value


#-------------------------------------------------------------------------------
###Space for functions
##def bellcurve(x, sigma):
##    return 10 * (1 / (sigma * (2 * pi) ** 0.5)) * e**(-0.5 * (x/sigma)**2)
##
##def connection_weight(diff):
##    signal = bellcurve(diff, c_sigma_1) - bellcurve(diff, c_sigma_2)
##    return connection_strength * signal
##
##
##def get_vt(t, v_0, s_0, I):
##    return I + (s_0 / (1 - synapse_decay)) * e**(-synapse_decay * t) \
##           + (v_0 - I - (s_0 / (1 - synapse_decay))) * e**(-t)
##
##def get_dvdt(t, v_actual, s_0, I):
##    #If I only calculate dvdt when I already know v(t) then this is easier
##    return I - v_actual + s_0 * e**(-synapse_decay * t)



#-------------------------------------------------------------------------------
#Copy/pasting over some constants

neurons_number = 1001
spikes_sought = 1250


threads = 256
blocks = int((neurons_number+threads-1)/threads)


v_r = 0
v_th = 0.1
dx = 0.1
synapse_decay = 2   #Don't let this be 1 for now
sigma = 10
connection_strength = 0.01
c_sigma_1 = 5
c_sigma_2 = 10
leniency_threshold = 0.001 #Removing choice between absolute/relative for now
error_bound = 0.0000001

#-------------------------------------------------------------------------------


#Set up variable arrays
voltage = default_rng().random(neurons_number)  #Should be v_r if nothing else
#voltage = numpy.zeros(neurons_number)
synapse = numpy.zeros(neurons_number)
lower_bound = numpy.zeros(neurons_number)
upper_bound = numpy.zeros(neurons_number)
firing_time = numpy.zeros(neurons_number) #Could reuse one of the bounds but eh
fire_flag = numpy.zeros(neurons_number, dtype = numpy.dtype(int))
input_strength = numpy.zeros(neurons_number) #This one won't be modified

d_voltage = cuda.to_device(voltage)
d_synapse = cuda.to_device(synapse)
d_lower_bound = cuda.to_device(lower_bound)
d_upper_bound = cuda.to_device(upper_bound)
d_firing_time = cuda.to_device(firing_time)
d_fire_flag = cuda.to_device(fire_flag)
d_input_strength = cuda.to_device(input_strength)

#Set up recording for firing events
#While list appending is probably not usable in C/CUDA, I'll use it here for
# simplicity and to not worry about how to handle the exceptions
spike_count = 0
spike_id = []
spike_time = []
simulation_time = 0

#Initialise voltage, input_strength via CUDA
voltage_init[blocks, threads](d_voltage)
input_init[blocks, threads](d_input_strength)
#print(input_strength)

check = -20000

while spike_count < spikes_sought:
    check += 1
    """Check if each neuron can fire, create estimated firing times"""
    cuda.synchronize()
    fire_check[blocks, threads](d_voltage, d_synapse, d_input_strength,
                                d_fire_flag, d_lower_bound, d_upper_bound)
    cuda.synchronize()
              
    d_fire_flag.copy_to_host(fire_flag)
    d_lower_bound.copy_to_host(lower_bound)
    d_upper_bound.copy_to_host(upper_bound)


    if check == 1:
        d_voltage.copy_to_host(voltage)
        d_synapse.copy_to_host(synapse)
        temp_1 = 0
        temp_2 = 0
        temp_3 = 0
        temp_4 = 0
        temp_5 = 0
        for n in range(neurons_number):
            temp_1 += fire_flag[n]
            temp_2 += lower_bound[n]
            temp_3 += upper_bound[n]
            temp_4 += voltage[n]
            temp_5 += synapse[n]
        print(temp_1, temp_2, temp_3)
        print("v sum:", temp_4)
        print("s sum:", temp_5)
        print(synapse)
        
    """Find out the earliest latest bound on firing"""                    
    best_worst_time = 0
    best_worst_flag = False
    for n in range(neurons_number):
        if fire_flag[n]:
            if not best_worst_flag:
                best_worst_flag = True
                best_worst_time = upper_bound[n]
            if upper_bound[n] < best_worst_time:
                best_worst_time = upper_bound[n]

    """Cull neurons that have no chance of firing before that bound"""
    for n in range(neurons_number):
        #Precursor cull of too-slow candidates
        if fire_flag[n]:
            if lower_bound[n] > best_worst_time * (1 + leniency_threshold):
                #Using relative leniency
                fire_flag[n] = 0
    #print("ff50", fire_flag[50])

    d_fire_flag = cuda.to_device(fire_flag)
    """Produce accurate estimates of remaining neurons' firing times"""
    cuda.synchronize()
    bisection[blocks, threads](d_voltage, d_synapse, d_input_strength,
                               d_fire_flag, d_lower_bound, d_upper_bound,
                               d_firing_time)
    cuda.synchronize()

    d_firing_time.copy_to_host(firing_time)

    if check == 1:
        temp_1 = 0
        for n in range(neurons_number):
            temp_1 += firing_time[n]
        print(temp_1)
        print(firing_time)
    
    """Check which neuron has the fastest firing time"""
    fastest_time = 0
    fastest_flag = False
    for n in range(neurons_number):
        if fire_flag[n]:
            if not fastest_flag:
                fastest_time = firing_time[n]
                fastest_flag = True
            elif firing_time[n] < fastest_time:
                fastest_time = firing_time[n]

    """Check which neurons are fast enough to fire"""
    cuda.synchronize()
    did_fire[blocks, threads](d_fire_flag, d_firing_time, fastest_time)
    cuda.synchronize()

    d_fire_flag.copy_to_host(fire_flag)
    """Record new firing times"""
    simulation_time += fastest_time
    new_spikes = 0
    for n in range(neurons_number):
        if fire_flag[n] == 1:
            spike_id.append(n)
            spike_time.append(simulation_time)
            new_spikes += 1

    """Update values"""
    cuda.synchronize()
    postclean[blocks, threads](d_voltage, d_synapse, d_input_strength,
                               d_fire_flag, d_lower_bound, d_upper_bound,
                               d_firing_time, fastest_time)
    cuda.synchronize()

    #Quit if no new neurons can fire
    if new_spikes == 0:
        print("No new neurons fired...")
        #print(fire_flag)
        break
    spike_count += new_spikes

    #Can go back later and see how many of these need to be in distinct loops
    #Related to how many distinct GPU kernel calls to make

    #print(check, spike_count)

        
print("Time taken:", time.time() - stopwatch)
print(spike_count)

plt.figure(figsize=(4.5, 4.5), dpi=1000)
x_axis = numpy.array(spike_id)
y_axis = numpy.array(spike_time)
plt.scatter(x_axis, y_axis, s=5, c="#a0a070")
plt.title("Neuron firing times")# (Bisection CUDA)")
plt.xlabel("Neuron id/position")
plt.ylabel("Time")
plt.margins(x=0, y=0.01)
plt.xlim(300, 700)

#plt.set_size_inches(12, 12)

plt.savefig("saved_figure_LIF_untitled_v2.pdf")
#plt.show()   
            
                
            
        
    
