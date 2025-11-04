"""
Tool for reading in data output by the spiking neuron simulator.

Reconstructs the lookup and module name dictionaries, and parses the firing
time data.
"""

import numpy

def get(filename):
    """Return an object holding the data in [filename].csv"""
    return Data(filename)

class Data:
    def __init__(self, filename):
        datafile = open(filename + ".csv", "r")

        self.lookup = {}
        self.module_names = {}
        raw_data = []
        mode = None
        for line in datafile:
            line = line.replace("\n", "")
            if ":" in line:
                if line[:-1] == "PARAMETERS":
                    mode = "PAR"
                elif line[:-1] == "MODULES":
                    mode = "MOD"
                elif line[:-1] == "DATA":
                    mode = "DATA"
            elif "," in line:
                entries = line.split(",")
                if mode == "PAR":
                    self.lookup[entries[1]] = entries[2]
                elif mode == "MOD":
                    self.module_names[entries[1]] = entries[2]
                elif mode == "DATA":
                    raw_data.append(entries)

        assert len(raw_data) > 0
        dimension = len(raw_data[0]) - 2
        assert dimension > 0
        self.data = numpy.zeros(len(raw_data),
                                dtype=[("index", int),
                                       ("time", float),
                                       ("coord", float, (dimension,))])
        for i, entry in enumerate(raw_data):
            self.data["index"][i] = int(entry[0])
            self.data["time"][i] = float(entry[1])
            for d in range(dimension):
                self.data["coord"][i][d] = float(entry[2 + d])

        
if __name__ == "__main__":
    test_data = get("2D_demo_newtype")

        
                    
                    
                    
        
