The "modules" folder contains all the modules that may be used by the main program.  The subfolder "modules/general" contains the mandatory modules used regardless of model specification.  Otherwise, there are multiple modules available to implement different functions from which a model may be assembled.

Several modules relate specifically to the sLIF two-variable neuron model, while others relate to its extension, the ssLIF three-variable model.  Other modules implement generic options that can be useful to a variety of models.

Modules are grouped below according to the name used in the module import config file; the headers are a plaintext description of the category followed by their alias in the config file.

## Distance-measuring functions (`distance`)
These modules specify how the distance between two neurons is calculated.
 - `distances_R123.py` calculates the Euclidean distance between points by assuming they are embedded within the standard 1D, 2D or 3D real number spaces.
 - `distances_ring_1.py` (is redundant; should be deleted by now)
 - `distances_ring_123.py` calculates the Euclidean distance between points by assuming they are embedded within the standard 1D, 2D or 3D toroidal spaces.  In other words, it's the same type of measurement as in `distances_R123.py`, but now the boundaries of the network are assumed to wrap around and touch each other to give a periodic domain, so there are additional possible shortest paths to test.

## Connection weighting functions (`connection_weight`)
These modules specify how the connection weights between two neurons is calculated, as a function of the distance between the neurons.
 - `connection_weight_difference_of_exponentials.py` gives the connection weighting as the difference between two exponentially-decaying functions.
 - `connection_weight_difference_of_normals.py` gives the connection weighting as the difference between two Gaussian functions.

## External input updating (`i_update`)
These modules control if and how the external input to the neurons change over time.  Unlike other parameters, this is expected to be set per-neuron.
 - `input_update_constant.py` gives a dummy function that causes the external input to be held at a constant level throughout the simulation.
 - `input_update_shutoff.py` causes the input to switch to a uniform value across the entire network at a specified time.  The name reflects the expectation that this simulates a short stimulus at the beginning to kickstart self-sustaining activity, but the input could just as easily be raised instead.
 - `input_update_ssLIF_dynamic.py` is used in the ssLIF model when dynamically varying the parameter R or D.  The resting voltage of an ssLIF neuron is a function of R, D and the external input, but usually we wish to keep the resting voltage constant as we vary R or D.  This module recalculates the external input to each neuron at each step to keep their rest state constant.

## Voltage array initialisation (`v_init`)
These functions set the voltage values for each neuron.  These are executed after any initial conditions are imported from file, so they can modify or overwrite imported data.
 - `voltage_init_null.py` gives a dummy function that does not edit the values of the voltage array.  This should only be used if initial conditions are being imported from file; otherwise, the array may contain garbage data as initial conditions that will cause model behaviour to vary between runs.
 - `voltage_init_primesine.py` causes the voltage to vary as the average of three coprime-period sine waves, taken as a function of the distance between a given neuron and the origin.  It's not a very serious module; it's just something that gives a "rough" yet deterministic and spatially-correlated variation to the neurons.
 - `voltage_init_random_uniform.py` sets the initial voltage of each neuron to independent uniform random variables distributed between the integrate-and-fire firing threshold and reset value.
 - `voltage_init_zero.py` sets the initial voltage of each neuron to 0.

## External input array initialisation (`i_init`)
These modules set the initial value for the external input to each neuron.
 - `input_init_constant.py`
 - `input_init_coshreciprocal.py`
 - `input_init_coshresciprocal_1D.py`
 - `input_init_gaussian.py`

## Coordinate array initialisation (`coord_init`)
These modules sets the spatial coordinates for each neuron.
 - `coordinate_init_R123.py` populates neurons on a regular grid in 1D, 2D or 3D.  By changing parameters this grid can be made square, rectangular or triangular/hexagonal.

## Model equations (`v_calcs`)
These modules give the solutions and derivatives of the single-neuron model equations.
 - `v_calcs_sLIF.py` gives the model equations for the sLIF model.
 - `v_calcs_ssLIF.py` gives the model equations for the ssLIF model.

## (`fire_check`)
 - `fire_check_sLIF.py`
 - `fire_check_ssLIF_trig.py`

## (`root_finder`)
 - `newton_raphson.py`
 - `newton_raphson_ssLIF.py`

## (`paramchange`)
 - `update_beta.py`
 - `update_D.py`
 - `update_R.py`
 - `update_null.py`

## (`cleanup`)
 - `cleanup_sLIF.py`
 - `cleanup_ssLIF.py`

## Miscellaneous
 - `device_gaussian.py` implements a device-side (GPU-side) Gaussian curve function for use in other modules.  Not specified through config files, and instead called directly by modules as needed.
