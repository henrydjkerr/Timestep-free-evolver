This folder contains utility tools that don't interface directly with the main program while it's running, but can be used e.g. to generate input files, or find solutions to implicit equations.

## c-autofinder.py
When looking at planar travelling waves, the sLIF equations give an implicit equation for the wave speed *c*.  This program aims to find a solution to that implicit equation.

Set the model constants in the top of the program file.  The program calculates "for the given value of *c*, what does the value of *v<sub}th<\sub> - I* need to be to agree with this as a possible wave speed?", starting with *c = 1*.  It then compares to the actual specified value of *v<sub}th<\sub> - I*.  If its predicted value is higher than that demanded, it adds a flat increment to *c*.  If its predicted value is lower than that demanded, it halves all future increment sizes, then subtracts the new increment from *c*.  
 * The increment used in both directions is the same.
 * The function assumes the function is decreasing, which is not always the case.  Though it seems it usually is in the zones of interest.
If the step sizes becomes sufficiently small, the loop exits.  Otherwise it reports failure and terminates after 100 loops.  The program then prints out the value of *c* it settled on, and additionally generates a graph of the function in that area, so you can check visually that nothing weird is going on.
 * Since the equation involves multiplying very large and very small numbers together, it's quite possible to run into cases where the numerics severely lose precision.  The graph helps to show that.

**Additional:** please note the adjustment of *A*, *B* (the parameters controling excitatory and inhibitory strength) that depends on whether you're working in 1D or 2D.  Due to the 2D wave collapsing to the 1D case after one integration, the same equations solve the problem just fine, *but* only with adjustment for the extra coefficient the integration generates.  This has already been factored in to this program, so if you're working in 2D you should put in the values of *A*, *B* that you're using in `parameters.txt`.


## on-wave-generator.py
This program is for generating the travelling wave profiles of *v* and *s* for the sLIF system that you can then import into the evolver program as initial conditions.  Works for 1D simulations.

Set your parameters at the top of the file and it will write into the files `v-import.txt` and `s-import.txt`.  This requires you to know the wave speed already; if you don't know it, then use `c-autofinder.py` to find it.

Since `evolver.py` expects these import files to be found in its root directory, if you generate them in this folder you need to copy them over.  If you're doing that a lot, it may be faster to copy this program over into the root folder instead and run it from there, though it would be wise to clean up afterwards so you don't have two different copies of this program lying around.


## on-wave-generator-2D.py
This is a minor modification of `on-wave-generator.py` that generates initial conditions for 2D simulations instead.  It merely duplicates the entries, so every neuron with the same *x* coordinate has the same values of *v* and *s*.

The same notes on changes to *A*, *B* in 2D as mentioned in the `c-autofinder.py` description also apply here.


## on-wave-generator-2D-wobbly.py
A slightly hacky modification of `on-wave-generator-2D.py` to add rudimentary perturbations.  Each line of values parallel to the *x* axis is shifted proportional to *cos(ky)*.  You have control over the amplitude and the frequency of the perturbation.

This perturbation isn't created because it means anything, it's created because it's easy to create.
