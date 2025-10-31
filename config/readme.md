These are lists of the currently available config files.  They typically relate to one of two integrate-and-fire models I have worked on using this program, internally dubbed the sLIF and ssLIF models.  If you're coming here externally, it's probably from a work using the ssLIF model (with per-neuron variables v, u, s).

## Parameter files
 - `parameters.txt` is the only file provided at present.  The sLIF model parameters are a subset of the ssLIF parameters, typically holding similar values.

## Module files
 - `settings-sLIF-DEFAULT.txt` selects modules to run the sLIF model with the neuron voltage initialised at 0 and a transient period of external excitatory input in the centre of the domain.
 - `settings-sLIF-IMPORT.txt` selects modules to run the sLIF model with the neuron voltage (and synaptic buffer) initialised according to input files, and constant external input.
 - `settings-sLIF-KW.txt` was for internal use (don't worry about it).
 - `settings-ssLIF-DEFAULT.txt` selects modules to run the ssLIF model with the neuron voltage initialised at 0 and a transient period of external excitatory input in the centre of the domain.
 - `settings-ssLIF-IMPORT.txt` selects modules to run the ssLIF model with the neuron voltage (and other variables) initialised according to input files, and constant external input.
 - `settings-ssLIF-IMPORT-dynamic.txt`selects modules to run the ssLIF model with the neuron voltage (and other variables) initialised according to input files, and constant external input.  It additionally alters __PARAMETER-NAME__ according to settings in `parameters.txt`

## Array files
 - ``arrays-sLIF.txt` sets the arrays used in the sLIF model.
 - ``arrays-ssLIF.txt` sets the arrays used in the ssLIF model.

## Initial condition files
 - `data_import-NULL.txt` is an empty file used when you don't want to load any initial conditions from external files.
 - `data_import-wave.txt` names import files for the voltage and synaptic buffer variables, according to the sLIF model.
 - `data_import-wave-ssLIF.txt` names import files for the voltage, ion channel and synaptic buffer variables, according to the ssLIF model.
 - `data_import-wave-ssLIF-2D.txt` also names import files for the voltage, ion channel and synaptic buffer variables, according to the ssLIF model, but labelled differently so as to not confuse between inputs for 1D and 2D simulations.
