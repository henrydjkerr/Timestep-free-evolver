import time
from modules.general import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
spikes_sought = lookup["spikes_sought"]
dimension = lookup["dimension"]

def save_data(spike_id, spike_time, coordinates):
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
    for k in range(len(spike_id)):
        n = spike_id[k]
        line = str(n) + "," + str(spike_time[k])
        for d in range(dimension):
            line += "," + str(coordinates[n, d])
        outfile.write(line + "\n")
    outfile.close()

def save_profile(array, name):
    timestamp = time.strftime("%Y%m%d%H%M%S")
    identifier = "{}-{}N-{}sp".format(timestamp, neurons_number, spikes_sought)
    filename = "output/output-{}_{}.txt".format(identifier, name)
    outfile = open(filename, "w")
    for value in array:
        outfile.write(str(value) + "\n")
    outfile.close()
    
