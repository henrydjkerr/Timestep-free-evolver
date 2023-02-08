import numpy
from math import e
from time import time
import matplotlib.pyplot as plt

def s(t):
    return ((beta * A)/(beta + a*c)) * e**(a*c*t)

def v(t):
    return I + ((beta * A)/((beta + a*c)*(1 + a*c))) * e**(a*c*t)

A = 1
a = 1
beta = 1

v_th = 1
v_r = -10
I = 0.9

#As in overleaf running review for 1D exponential
c = (1/(2*(a**2))) * (-(beta + 1) +
                    ((beta + 1)**2 + 4*beta*(A*(a**2)/(v_th - I) - 1))**0.5)

print(c)

width = 20
neurons = 500
dx = width/neurons

coordinates = numpy.arange(0, width, dx)
voltage = numpy.zeros(neurons)
synapse = numpy.zeros(neurons)
spikes = numpy.zeros(neurons)

spike_times = []
spike_places = []

for n in range(neurons):
    voltage[n] = v(-coordinates[n]/c)
    synapse[n] = s(-coordinates[n]/c)

timestep = 0.001
finished = False
stopwatch = time()
time_limit = 30
model_time = 0

def H(x):
    if x > 0:
        return 1
    else:
        return 0

while finished == False:
    model_time += timestep
    for n in range(neurons):
        voltage[n] += timestep * (I - voltage[n] + synapse[n])
        
        cocoord = coordinates[n] - c * model_time
        #Options for synapse updating:
        #-1- Vanilla; keep the update-on-firing active.  Doesn't match theory
        synapse[n] -= timestep * beta * synapse[n]
        #For others you have to comment out the marked line below
        #-2- Timestep according to theory.  Matches theory.
        #synapse[n] += timestep * beta *(A*e**(-a*cocoord) - synapse[n])
        #synapse[n] *= H(cocoord)
        #-3- Entire value according to theory.  Matches theory.
        #synapse[n] = (beta * A)/(beta + a*c) * e**(-a*cocoord) * H(cocoord)
        #-4- Update from firing neurons according to theory.  Close enough.
        #synapse[n] -= timestep * beta * synapse[n]
        #synapse[n] += timestep * c * beta * A * e**(-a*cocoord) * H(cocoord)
        
    for n in range(neurons):
        if voltage[n] >= v_th:
            voltage[n] = v_r
            #Record stuff
            spikes[n] = model_time  #This only records the latest per neuron
            spike_times.append(model_time) #All firing times
            spike_places.append(dx*n)   #All firing positions
            
            for m in range(neurons):
                if m != n:
                    if m == 40 and m > n:
                        before = synapse[m]
                    #d = abs(coordinates[n] - coordinates[m])
                    d = dx*abs(n-m)
                    #Comment out this line if you're overriding synapse updating
                    synapse[m] += dx * beta * A * (e**(-a * d))
            if n == neurons - 1:
                finished = True
    if time() - stopwatch > time_limit:
        print("Ran out of time")
        break

if finished == True:
    print("Euler speed:", width/model_time)

    plt.figure()
    #x_axis = coordinates
    #y_axis = spikes
    x_axis = numpy.array(spike_places)
    y_axis = numpy.array(spike_times)
    plt.scatter(x_axis, y_axis, s=0.5)
    plt.show()

    
