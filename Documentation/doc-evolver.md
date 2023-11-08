`evolver.py` is the primary code file from which this program is run.  As much as possible, this core is kept generic to allow for maximum flexibility by changing the parameters of the system and the modules it imports.

## Execution overview
As much as possible, the main loop is agnostic to how exactly each step is performed.

1. Import selected modules, and thus implicitly:
   1. Read in parameters and data.
   2. Create and initialise arrays.
2. Transfer arrays from host memory to device memory.
3. Create lists for holding generated firing event data.
4. Enter the main loop.  While the number of recorded firing events is less than the target number:
   1. (Device-side) For each neuron, check if the neuron will fire given its current conditions.  If it will fire, flag it as firing-capable, and find bounds on its firing time, taking the present time as the zero-point.
   2. (Host+Device) Of the firing time bounds that have been generated, find the smallest value for the upper bound.
   3. (Device-side) For each neuron flagged as firing-capable, check if its lower bound is higher than the lowest upper bound by a parameter-specified margin.  Remove the flag if so.
   4. (Device-side) For each neuron still flagged as firing-capable, find an accurate and precise value for the time at which the neuron will fire.
   5. (Host+Device) Of the accurate firing times that have been generated, find the smallest value.
   6. (Device-side) For each neuron flagged as firing-capable, check if its calculated firing time is higher than the smallest firing time by a (the same as before) parameter-specified margin.  Remove the flag if so.
   7. Copy the array of firing flags from the device to the host.
   8. (Host-side) Increment the current model time by the amount of the smallest firing time.
   9. (Host-side) Record each neuron with an active firing flag as having fired at the current model time.
   10. If no neurons have fired, exit the loop.
   11. (Device-side) Otherwise, update the state of each neuron to the current model time, including the effects upon them of any neurons that have fired.  For neurons that have fired, apply any reset conditions needed.
   12. (Device-side) If required, update the external input parameter for each neuron.
5. Save recorded firing event data to file.

Note 1: the purpose of softening the cut-off conditions by some threshold, and the rounding down of sufficiently close firing times, is to pre-empt certain numerical errors.  In certain situations, such as a plane wave in 2D or a symmetrically-expanding pulse in 1D, one may expect that the mathematical model will report multiple neurons firing simultaneously (a whole line, or two mirrored partners, respectively).  Firstly, we want the system to be able to deal with simultaneous firing events without needing to run the entire loop again, for efficiency reasons.  Secondly, since our present models do not include the delay of signal transmission, we do not want a situation where one neuron's firing interferes with the firing of a neuron that would otherwise fire nigh-instantaneously.  

Note 2: There are currently some issues with the later steps of updating and resetting neurons, as well as updating the external input, being insufficiently generic, so they are likely to be subject to change.
