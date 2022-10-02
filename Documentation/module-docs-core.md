# Core modules

## Param.py

This is a generic module that scans through the file "parameters.txt" in the main directory, and parses any line containing a colon `:` as a dictionary key:value pair.  This is loaded into the dictionary `lookup`.  Values are saved as the first valid option of `int`, `float`, `str`.

## ParamPlus.py

Augments and enhances `Param.py`:
 - Runs some validation checks on expected data, and corrects certain bad inputs to avoid certain cases that would cause the program to misbehave.
 - Obligate constants:
     - `neuron_count_x` (`int` forced to be >= 1), number of neurons along the x-axis
     - `neuron_count_y` (`int` forced to be >= 1), number of neurons along the y-axis
     - `neuron_count_z` (`int` forced to be >= 1), number of neurons along the z-axis
     - `dx` (`int`/`float` > 0), spacing between neurons along the x-axis
     - `dy` (`int`/`float` > 0), spacing between neurons along the y-axis
     - `dz` (`int`/`float` > 0), spacing between neurons along the z-axis
     - `even_offset_yx` (`int`/`float` forced to be >= 0), offset applied to neurons in the x direction along even rows parallel to the y-axis (set to 0 for a regular grid)
     - `even_offset_zx` (`int`/`float` forced to be >= 0), offset applied to neurons in the x direction along even rows parallel to the z-axis (set to 0 for a regular grid)
     - `even_offset_zy` (`int`/`float` forced to be >= 0), offset applied to neurons in the y direction along even rows parallel to the z-axis (set to 0 for a regular grid)
     - `threads` (`int` forced to be >= 1), the number of threads running per CUDA block (stick to multiples of 32)
     - `spikes_sought` (`int` forced to be >= 1), the minimum number of firing events above which the program will finish its operation.
     - `v_r` (`int`/`float`), the voltage that neurons reset to after firing.
     - `v_th` (`int`/`float`), the voltage above which neurons are considered to fire.
     - `synapse_decay` (`int`/`float` > 0), an exponential scaling factor for how slowly neurons process incoming signals (Note: really, this probably shouldn't be considered obligatory since it depends on choice of model)
     - `leniency_threshold` (`int`/`float` >= 0), the relative amount by which a neuron can fire after the fastest neuron and still be considered to fire "at the same time".
     - `error_bound` (`int`/`float` > 0) , the absolute difference between iterations of a numerical root-finding scheme at which the scheme is considered to have converged.
 - Derives some useful secondary constants (such as dimensionality) from the primary constants read in by `Param.py`.
     - `dimension` (`int` 1, 2 or 3), determined by values of `neuron_count_x`/`y`/`z` to say whether the modelled population presents as a 1D line/curve, a 2D plane/surface or a 3D volume.
     - `neurons_number` (`int`), the total number of neurons modelled.
     - `blocks` (`int`), the number of CUDA blocks needed to house the number of neurons needed, given the already-requested number of threads per block.

Note: if I wished, it would be easy to recast `int`/`float` constants as pure `float`s, since the ambiguity in type comes from `Param.py` automatically selecting type when reading them in from file.

## Control.py

This is the primary access point for the main body of the program to access individual modules.  Control.py imports a custom selection of modules into itself, renaming them according to their purpose.  It also imports ParamPlus, which has already in turn imported Param, meaning Control provides access to the expanded `lookup` dictionary bound to `Control.lookup`.

Custom modules are selected by reading their names from the file `import-profiles.txt`.  This file uses the same read-in format as `parameters.txt`, with the 

Generic identities used for binding custom modules:
 - `d`: distance measuring functions, exposes:
     - Device function `float d.get_distance_from(array coordinates, int n, float x, float y, float z)`: return the distance from neuron `n` (indexing position recorded in `coordinates`) from the position (x, y, z).  1D or 2D cases would still require the values passed, but can ignore them.
     - Device-side `float d.get_distance_between(array coordinates, int n_1, int n_2)`: return the distance between the two neurons indexed `n_1` and `n_2` in `coordinates`.
     - Device-side `float d.get_distance_from_centre(array coordinates, int n) = float`: return the distance from neuron `n` (indexed as above) and the centre of the neuron population.  Note: I might change this to `get_distance_from_zero` and expect the user to sit the grid centred on zero instead.
 - `c`: connection weight functions
     - Device function `c.connection_weight(float diff)`: return the connection strength weighting for two neurons with a distance separation given by `diff`.
 - `v`: calculations related to the voltage equations
     - `float v.get_vt(float t, float v_0, float s_0, float I)`: return the value of _v(t)_ given constant input current _I_ and initial conditions for voltage _v(0) = v_0_ and for synaptic buffer _s(0) = s_0._  The current model time is considered as _t = 0._
     - Device function `float v.get_dvdt(float t, float v_actual, float s_0, float I)`: return the gradient of _v(t)_ given constant input current _I_ , initial synaptic buffer value _s(0) = s_0_ and the value of _v(t)_ (typically calculated using `v.get_vt` previously).
 - `i`: functions for updating the input current strength as the model runs
     - Host-side `bool input_control(float new_time)`: return `True` if the program should run code to modify the input current at the end of this loop, `False` otherwise.  `new_time` is the model-time the model will be updated to, with _t = 0_ taken as the point at which the entire model was initialised.
     - Device-side `void input_update(array input_strength_d, float new_time, float time_change)`: update the input current values in `input_strength_d`, potentially based on the information of the new global model-time `new_time` and the amount that time has been advanced on this particular loop, `time_change`.
 - `vi`: initialise voltage values
 - `ii`: initialise input current values
 - `xi`: initialise neuron coordinates
 - `check`: check whether a neuron will or will not fire, and generate appropriate bounds on the firing time
 - `solve`: find a precise firing time based on bounds generated by `check`


