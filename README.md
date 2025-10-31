This is a GPU-parallelised numerical simulator for networks of spiking neurons with all-to-all coupling (allowing zero-weighted coupling for *de facto* unconnected neurons).   For an input of a specified model, parameters and initial conditions, the output is a list of firing events, showing which neuron fires when.  This is saved in a .csv file along with the model parameters.  At this time the program does not support batch processing.

The program is designed to be modular, with models and their handling specified through module imports controlled via text configuration files.  Examples of customisable factors include the single-neuron governing equations, the solver for the single-neuron model, the connectivity weighting between neurons, and the network dimension and topology.  These model profiles can in turn be switched between through a top-level configuration file.  The code was designed in the expectation that it should be used with neuron models that admit an analytical solution to their governing ODEs between discrete inputs, allowing for the next firing time to be discovered through numerical root-finding; however, the root-finding code is self-contained and specified per-model so in principle it can work with more complex models that require timestepping (AKA numerical integration).

This program was written to support my work on my PhD thesis and attendant publications.  It was written under Python 3.13, making use of numpy 2.1.3, scipy 1.15.2 and Numba 0.61.0, as well as matplotlib 3.10.0 for visualisation.  The program makes use of GPU-based parallel processing via Numba's CUDA interface, so a compatible NVidia GPU is also necessary. You will also need to install the CUDA toolkit, along with any additional CUDA dependencies required in the Numba documentation.  The program was written and used on Windows 10; I have not checked for cross-platform compatibility.

Further details on the function of the program are given in the Documentation folder.  Auxilliary tools that help with generating initial conditions and interpreting model solutions are supplied in the side-tools folder.  The Figures folder contains the code and data used to produce figures for various publications using this model.

The top-level folders are thus:
 - "config" contains the .txt files used to specify the model and otherwise control inputs and module selection.
 - "Documentation" contains further information on the workings of this program.
 - "Figures" contains the code and data used to produce figures for various publications that I prepared using this program.
 - "modules" contains the various compulsory and optional modules imported by the main body, `evolver.py`.
 - "output" is the folder within which program output is saved.
 - "side-tools" contains a collection of auxilliary tools that help in generating initial conditions and interpreting model solutions, but are not accessed directly by the main program.
 - "visualisation" contains various ways to visualise the outputs of the model.
