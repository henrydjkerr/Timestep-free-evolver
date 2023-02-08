"""
Reads data from a .csv file in format (coordinate, voltage, synapse)
and uses it to fill the supplied arrays with initial conditions.

Currently only set up to handle 1D coordinates.

Can handle filling only part of an array, but fails if the array is
too short.
"""

import numpy

from modules import Control
lookup = Control.lookup

neurons_number = lookup["neurons_number"]
dimension = lookup["dimension"]

def data_importer(filename):
    """
    Read data from file into numpy arrays then return the arrays.
    Assumes formatting of coord, (coord,) voltage, synapse
    """
    coordinates = numpy.zeros((neurons_number, dimension))
    voltage = numpy.zeros(neurons_number)
    synapse = numpy.zeros(neurons_number)

    infile = open(filename, "r")
    for n, line in enumerate(infile):
        if n >= neurons_number:
            raise IndexError("The source data file {} has too many entries \
to fit in the supplied arrays.".format(filename))
        entries = line.strip("\n").split(",")
        if len(entries) == 3:
            coordinates[n, 0] = float(entries[0])
            voltage[n] = float(entries[1])
            synapse[n] = float(entries[2])
        elif len(entries) == 4:
            coordinates[n, 0] = float(entries[0])
            #coordinates[n, 1] = float(entries[1])
            voltage[n] = float(entries[2])
            synapse[n] = float(entries[3])
    return coordinates, voltage, synapse    
    
    
