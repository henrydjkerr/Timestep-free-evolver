`modules/general` contains core functionality of the program that should not vary when altering the mathematical model.


## locator.py
Read and store where individual configuration files are to be found from `settings_selection.txt`, and enable reading their contents.

### Functions:
`file_reader(str filename)`: take the name of a file, then returns a list of lists produces as follows: for each line in the file, if the line contains a colon, create a list of strings from the line's contents by splitting at the position of each colon, then append this list as a single entry in the to-be-returned list.

### Constants:
`dict location`: stores the name of each configuration file in use, keyed by a string naming their purpose.  Currently used are:
 - `parameters` stores the general numerical constants and dimensions for the model.
 - `modules` determines which modules are imported to construct and service the model in code.
 - `arrays` specifies the name, length and data type of the variable arrays needed for the model.
 - `import` optionally specifies which of these arrays should be initialised with from-file data rather than a mathematical function.  (If nothing is needed, this file still must be specified, but can be left empty.)


## Param.py
Read in the parameters config file using `locator.py`, storing the results in a dictionary.  The first entry in each line is used as the key, while the second entry is used as the value.  The data type of the value is automatically detected, with the heirarchy `int` > `float` > `str`.  Further entries on each line are ignored.

Note that `Param.py` should not be referenced directly, as `ParamPlus.py` does necessary post-processing on the dictionary.  The dictionary is then pulled forward into `Control.py` without modification, so if you already need access to `Control.py` you don't need `ParamPlus.py` as well.

### Constants:
`dict lookup`: the aforementioned dictionary of parameters.


## ParamPlus.py
Run sanity checks on the existence, data type and range of certain compulsory parameters provided by `Param.lookup`.  This may be scaled back in future to make it more generic.

Additionally, generate a small number of additional parameters derived from the existing compulsory ones.

### Constants:
`dict lookup`: the dictionary is rebound here for ease of access.


## Control.py
Read in the modules config file using `locator.py`, and import the modules requested, binding them to shorthand aliases.  Liable to be rewritten for improved generic compatibility.

Additionally, as the manager for imported modules, also contains a function for initialising arrays, since the form initialisation takes is dependent on imported modules.

### Functions:
`fill_arrays(dict arrays)`: search through the dictionary of device-side arrays for specific keys, and run initialisation functions on those arrays.

### Constants:
Module bindings:
 - `d`: module for measuring sitances between points
 - `c`: module for giving connection weighting between points
 - `v`: module for supplying governing equations for neuron voltage (the primary variable)
 - `i`: module for enabling mid-simulation changes to the input current (external forcing variable)
 - `vi`: module for procedurally initialising the voltage array
 - `ii`: module for procedurally initialising the input current array
 - `xi`: module for procedurally initialising the array of neuron positional coordinates
 - `check`: module for determining whether a given neuron will fire, and generating bounds on the firing time
 - `solve`: module for numerically or analytically finding a precise firing time for a neuron

`dict lookup`: the dictionary of parameters is bound here as well.


## array_manager.py

Create and hold references to the host-side and device-side copies of arrays, and manage the transfer of data between the two. Also fill in arrays that are to be initialised with data from file.

### Constants:

`dict host_arrays`: dictionary of arrays held in RAM and directly accessible to your CPU.  Keyed according to their names in the arrays config file.

`dict device_arrays`: dictionary of arrays held in the GPU's memory. Keyed likewise.

Note that you cannot pass dictionaries to GPU functions using Numba; you have to first unpack what you need from the dictionary with a wrapper function.

### Functions:
`send(str key)`: copy an array from host memory to device memory.  `key` specifies the array in `host_arrays` and `device_arrays`.

`retrieve(str key)`: copy an array from device memory to host memory.

`send_all_to_device()`: copy all arrays in `host_arrays` to device memory.

`file_to_array(str filename, array array)`: read data from the specified file into an array.  The source format should be a plaintext file, one data value per line.


## data_loader.py

Hang on, isn't this just replicating functionality in `array_manager.py`?  Can't I just remove this?


## save.py

Write data on neuron firing times to file.

### Functions:

`save_data(list spike_id, list spike_time, array coordinates)`: write a .csv file with three sections:
  - All parameters used, both from file and derived, as stored in the `lookup` dictionary.
  - All modules specified in the module config file.
  - All firing event data, written in the format (neuron ID, firing time, neuron spatial coordinate).


## generic.py

A collection of smaller functions used in the mathematical treatment on the model, but independent of the actual form of the model.  For example, functions for sorting lists with the aid of the GPU.

At the moment, some functions are insufficiently generic to be places here, so they should be moved.

### Functions:

`did_fire`: nongeneric duplication of `cull_larger`, to be removed.

`cull_larger(array some_array, array flags, float threshold)`: for each index, set the flag to False if the array value is above the threshold value.

`postclean(dict arrays, float fastest_time)`: wrapper function to pull out the arrays needed for the parallel function `postclean_device`.  Since this requires specific knowledge of which arrays are used, and then what needs to be done to update them, this should be moved out of `modules/general`.

`postclean_device(...)`: using the previously determined time until the next firing event, update the value of all variables to the appropriate time, then propagate the signals from all firing events and reset any procedural arrays such as flags.  This needs to be moved out of `modules/general` for not being generic.

`find_small_qualified(array some_array, array flags, array results_array, float fill)`: utilising the parallel processing division of arrays into blocks, find the smallest value in each block of `some_array` that has a True flag, and write it to the corresponding block coordinate in `results_array`.  If all flags are false, write `fill` instead.

`results_array`'s length should be that of `some_array` divided by the block size specified in the parameter file, rounded up.  `fill` should usually be the maximal representable value.

If no flags are True, then the function will fill `results_array` with `fill`.

`find_smallest(array some_device_list, array flags, array partial_device_list, array partial_list)`: call `find_small_qualified(some_device_list, flags, partial_device_list, numpy.ma.minimum_fill_value(partial_device_list))`, then copy `partial_device_list` on the device to `partial_list` on the host, and return the lowest value in `partial_list`.

This function returns the lowest value in `some_device_list` that has a True flag.  If none of the flags are True, then the function erroneously returns the maximal fill value instead.  However, if you're calling this function with all the flags False then something's gone wrong already.

This two-stage process is used both to exploit the parallel capacity of the GPU for searching a large array, and to avoid transferring a full array from device memory to host memory.



