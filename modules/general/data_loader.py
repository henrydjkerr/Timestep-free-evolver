import numpy

from modules.general import locator
#from modules.general import Control
#lookup = Control.lookup
from modules.general import array_manager

def csv_load(array, filename):
    file = open(filename)
    length = len(array)
    for n, line in enumerate(file):
        if n >= length:
            raise IndexError("The source data file {} has too many entries \
to fit in the supplied arrays.".format(filename))
        entries = line.strip("\n").split(",")
        if len(entries) == 1:
            array[n] = entries[0]
        else:
            array[n] = entries
            #Want some error-checking on this?


filename = locator.location["import"]
raw_data = locator.file_reader(filename)
        

for line in raw_data:
    key = line[0]
    source = line[1]
    csv_load(array_manager.host_arrays[key], source)
    
