import numpy as np
from numba import cuda

from modules.general import locator
from modules.general import Control
lookup = Control.lookup

dimension = lookup["dimension"]
neurons_number = lookup["neurons_number"]


#------------------------------------------------------------------------------

def send(key):
    #Push data from host memory to device memory
    device_arrays[key] = cuda.to_device(host_arrays[key])

def send_all_to_device():
    #Push all arrays to device memory
    for key in host_arrays:
        send(key)

def retrieve(key):
    #Pull data from device memory to host memory
    device_arrays[key].copy_to_host(host_arrays[key])

def file_to_array(filename, array):
    #Read data from file into an array host-side
    #The file should only have one entry per line (currently unchecked)
    data = open(filename)
    for n, line in enumerate(data):
        if n >= len(array):
            raise IndexError("The source data file {} has too many entries \
to fit in the supplied arrays.".format(filename))
        line = line.strip("\n")
        array[n] = line
    data.close()

#------------------------------------------------------------------------------

#Set up dictionaries for references to the host-side and device-side arrays
host_arrays = {}
device_arrays = {}
#and the data types of the arrays
data_types = {"bool": np.bool_, "int": np.int_,
              "single": np.single, "double": np.double}

#Read in the arrays requested, then create them (empty)
filename = locator.location["arrays"]
raw_info = locator.file_reader(filename)

for line in raw_info:
    try:
        key = line[0]
        length = line[1]
        data_type = line[2]
        host_arrays[key] = np.zeros(lookup[length], data_types[data_type])
    except ValueError:
        pass
        #Should probably have proper error reporting here

#Manually create the coordinates array (unfilled)
host_arrays["coordinates"] = np.zeros((neurons_number, dimension),
                                      dtype = data_types["double"])

#Read in which arrays have pre-existing data on file, and fill them in
filename_import = locator.location["import"]
import_info = locator.file_reader(filename_import)

for line in import_info:
    key = line[0]
    data_filename = line[1]
    array = host_arrays[key]
    file_to_array(data_filename, array)




