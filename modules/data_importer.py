"""
Reads data from a .csv file in format (coordinate, voltage, synapse)
and uses it to fill the supplied arrays with initial conditions.

Currently only set up to handle 1D coordinates.

Can handle filling only part of an array, but fails if the array is
too short.
"""

import numpy

def data_importer(filename, coordinates, voltage, synapse):
    """
    Reads data from file into the supplied arrays.
    Requires the arrays all be the same length, the file to supply
    values for all of them, and the arrays to be at least as long
    as the file.
    """
    length = len(coordinates)
    assert length == len(voltage)
    assert length == len(synapse)

    infile = open(filename, "r")
    n = 0
    for line in infile:
        if n >= length:
            raise IndexError("The source data file {} has too many entries \
to fit in the supplied arrays (length {}).".format(filename, length))
        entries = line.strip("\n").split(",")
        if len(entries) == 3:
            coordinates[n, 0] = float(entries[0])
            voltage[n] = float(entries[1])
            synapse[n] = float(entries[2])
            n += 1

    
    
    
