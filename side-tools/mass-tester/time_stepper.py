import numpy
from math import pi, e

#------------------------------------------------------------------------------

def gaussian(x, sigma):
    return (1/sigma) * ((1/(2*pi))**0.5) * (e**(-0.5 * (x/sigma)**2))
    
#Set up host-side recording for firing events
#spike_count = 0
#spike_id = []
#spike_time = []
#simulation_time = 0

time_step = 0.0001

def get_zero_time(v_th, v_r, I, signal_strength, signal_sigma, synapse_decay,
                  voltage, synapse, dx, limit):
    simulation_time = 0
    finished = False
    while (finished == False) and (simulation_time < limit):
        simulation_time += time_step
        for n in range(len(voltage)):
            voltage[n] += time_step * (I - voltage[n] + synapse[n])
            synapse[n] -= time_step * synapse_decay * synapse[n]
        for n in range(len(voltage)):
            if voltage[n] >= v_th:
                voltage[n] = v_r
                #spike_count += 1
                #spike_id.append(n)
                #spike_time.append(simulation_time)
                for m in range(len(voltage)):
                    if n != m:
                        distance = dx * abs(n - m)
                        strength = synapse_decay * dx * signal_strength \
                                   * gaussian(distance, signal_sigma)
                        synapse[m] += strength
                if n == 0:
                    finished = True
                    time_taken = simulation_time
    return simulation_time
                    




