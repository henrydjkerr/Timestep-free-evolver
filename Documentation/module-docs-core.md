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

This is the primary access point for the main body of the program to access individual modules.  Control.py imports a custom selection of modules into itself, renaming them according to their purpose.
