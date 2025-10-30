This is a program intended for numerical simulation of a population of neurons, where each neuron is modelled individually.

In broad terms, the input is the description both of the single-neuron model and its arrangement into a population in terms of equations and parameters, as well as initial conditions, and the output is a list of IDs and times telling you which neuron fired when.

I have written this program with the intention of making it as modular as possible, so that different models can be used simply by changing which modules are imported via text file, rather than by rewriting the core loop.  However, it was designed with neuron models where an analytic solution to the governing differential equations can be found, so I make no promises about ease of use or efficiency when dealing with more complex models.

This program was written and tested on Windows 10, as is the case for most things I write, so I make no promises about cross-platform compatibility.  If I've made any heinous blasphemies against general good programming practices, I'd be interested to know, so please feel free to complain.


### Dependencies

This program is written in Python 3.13, utilising numpy 2.1.3, scipy 1.15.2 and Numba 0.61.0, as well as matplotlib 3.10.0 for visualisation.  The program makes use of GPU-based parallel processing via Numba's CUDA interface, so a compatible NVidia GPU is also necessary. You will also need to install the CUDA toolkit, along with any additional CUDA dependencies reduired in the Numba documentation.

## Program overview

The core program file is `evolver.py`.

The folder `config` stores `.txt` files that determine the parameters and arrays of the model, which modules to use to implement the model, and optional importing of initial conditions from file.  `settings_selection.txt` is contained in the same folder as `evolver.py`, and determines which files from `config` are read.
 - At present, initial conditions read from file are placed in the same top-level folder as `evolver.py` and `settings_selection.txt`.

The folder `modules` stores all the custom modules that may be used to customise the program, as determined by their inclusion in the active configuration files.  You may add your own modules, if you wish.

The folder `modules/general` contains general tools always used by the program, so they cannot be swapped out via config file.  That being said, `generic.py` still contains a number of assumptions about the model, so those functions need to be split off.

Program output is saved to the `output` folder.

The folder `side-tools` contains auxilliary programs that can support use of the main program, but are not called by the main program at any point.  

The folder `Documentation` contains documentation files.  This file is in that folder.

