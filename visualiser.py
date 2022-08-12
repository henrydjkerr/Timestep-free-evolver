"""
Program for visualising neuron data outputs.
"""

import numpy
import matplotlib.pyplot as plt


filename = "output-20220805003234-5000N-50000sp.csv"
filename = "output-KW-1000N-2x-fix.csv"

infile = open(filename, "r")

index = []
timestamp = []
position_x = []

for line in infile:
    if len(line) > 0:
        entries = line.strip("\n").split(",")
        index.append(float(entries[0]))
        timestamp.append(float(entries[1]))
        position_x.append(float(entries[2]))

infile.close()

plt.figure(figsize=(16, 16))
x_axis = numpy.array(position_x)
y_axis = numpy.array(timestamp)
plt.scatter(x_axis, y_axis, s=0.2, c="#a0a070")
plt.title("Neuron firing times")
plt.xlabel("Neuron position")
plt.ylabel("Time")
plt.margins(x=0, y=0.01)

plt.ylim(0, 200)
#plt.xlim(0, 6)

plt.savefig("1000N-wave.png")
plt.show()

##test_array_1 = numpy.arange(1000)
##test_array_2 = numpy.arange(1000)
##
##print(index[:100])
##
##plt.figure()#figsize=(4, 4))#, dpi=1000)
##x_axis = numpy.array(test_array_1) #index)
##y_axis = numpy.array(timestamp)
##plt.scatter(x_axis, y_axis)#, s=0.02, c="#a0a070")
##plt.title("Neuron firing times")
##plt.xlabel("Neuron id/position")
##plt.ylabel("Time")
###plt.margins(x=0, y=0.01)
###plt.xlim(0, 500)


  
            
