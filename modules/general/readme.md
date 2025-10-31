These are the core modules that provide basic functionality for the program, such as loading parameters and modules and saving data.

 - `array_manager.py` provides functions for moving arrays between the CPU ("host") and GPU ("device").  It also generates the (empty) arrays on startup.
 - `Control.py` imports the user-specified modules into itself, imports the dictionary of parameters from `ParamPlus.py` and fills in the model arrays with initial conditions.  It acts as a common access point for the rest of the program.
 - `generic.py` is the home for minor generic functions.  It presently contains functions for setting flags on neurons based on a threshold, and finding the smallest value of a variable across neurons with the appropriate flag.
 - `locator.py` handles the locating and basic reading of config files.
 - `Param.py` converts the parameter config file into a dictionary of correctly-typed parameter values.
 - `ParamPlus.py` is a wrapper around `Param.py` that does some sanity checks on a few parameter values (you may want to remove these if making a different model), and derives a few model parameters: the spatial dimension of the network, the total number of neurons and the total number of CUDA blocks needed.
 - `save.py` saves the program output to a `.csv` file.  These are named according to the time at which the simulation finished, and contains the modules loaded, the model parameters, and then the generated firing time data.
